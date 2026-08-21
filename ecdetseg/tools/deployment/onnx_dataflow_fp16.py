"""Build a calibrated, dataflow-aware explicit FP16 ONNX graph."""

from collections import Counter
from hashlib import sha256
from importlib import metadata
import json
from pathlib import Path
import tempfile
from typing import Callable, Dict, Optional, Sequence

import numpy as np

from tools.deployment.onnx_precision import PRECISION_POLICY_METADATA_KEY


DATAFLOW_FP16_POLICY = "explicit_fp16_dataflow_calibrated_v1"

# Keep only the operations whose result directly affects ranking, probability,
# or sub-pixel box decoding in FP32. ModelOpt's range analysis decides the rest.
DEFAULT_FP32_NODE_PATTERNS = (
    r"^/model/decoder/(?:Div|Log)$",
    r".*/integral(?:_\d+)?/(?:Softmax|MatMul)$",
    r".*/lqe_layers\.\d+/(?:Softmax|TopK|ReduceMean)$",
    r"^/Sigmoid$",
    r"^/TopK$",
)
# Use the physical finite FP16 range for measured activations. Values outside
# it remain FP32 automatically; precision-sensitive in-range operations are
# protected separately by DEFAULT_FP32_NODE_PATTERNS and full COCO parity.
DEFAULT_DATA_MAX = 65504.0
DEFAULT_INIT_MAX = 65504.0


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_value_info(value):
    import onnx

    copied = onnx.ValueInfoProto()
    copied.CopyFrom(value)
    return copied


def _set_static_batch(model, batch_size: int) -> None:
    """Give ModelOpt a concrete batch while it instruments every tensor."""
    for value in list(model.graph.input) + list(model.graph.output):
        tensor_type = value.type.tensor_type
        if not tensor_type.HasField("shape") or not tensor_type.shape.dim:
            continue
        first_dim = tensor_type.shape.dim[0]
        first_dim.ClearField("dim_param")
        first_dim.dim_value = batch_size


def _restore_io_contract(source_model, converted_model) -> None:
    """Restore dynamic I/O exactly and discard static calibration value_info."""
    converted_model.graph.ClearField("input")
    converted_model.graph.input.extend(
        [_copy_value_info(value) for value in source_model.graph.input]
    )
    converted_model.graph.ClearField("output")
    converted_model.graph.output.extend(
        [_copy_value_info(value) for value in source_model.graph.output]
    )
    converted_model.graph.ClearField("value_info")


def _set_metadata(model, values: Dict[str, str]) -> None:
    current = {entry.key: entry.value for entry in model.metadata_props}
    current.update(values)
    model.ClearField("metadata_props")
    for key, value in sorted(current.items()):
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = value


def _tensor_dtype_summary(model) -> Dict[str, Dict[str, int]]:
    import onnx

    count = Counter()
    elements = Counter()
    bytes_by_type = Counter()
    for initializer in model.graph.initializer:
        dtype = onnx.TensorProto.DataType.Name(initializer.data_type)
        count[dtype] += 1
        element_count = int(np.prod(initializer.dims, dtype=np.int64)) if initializer.dims else 1
        elements[dtype] += element_count
        try:
            bytes_by_type[dtype] += element_count * np.dtype(
                onnx.helper.tensor_dtype_to_np_dtype(initializer.data_type)
            ).itemsize
        except TypeError:
            pass
    return {
        "count": dict(sorted(count.items())),
        "elements": dict(sorted(elements.items())),
        "bytes": dict(sorted(bytes_by_type.items())),
    }


def _source_float_weight_stats(model, init_max: float) -> Dict[str, object]:
    import onnx
    from onnx import numpy_helper

    fp16_smallest_subnormal = float(np.nextafter(np.float16(0), np.float16(1)))
    total = 0
    above_fp16_max = 0
    below_fp16_subnormal = 0
    maximum = 0.0
    initializer_count = 0
    for initializer in model.graph.initializer:
        if initializer.data_type not in (onnx.TensorProto.FLOAT, onnx.TensorProto.DOUBLE):
            continue
        values = np.asarray(numpy_helper.to_array(initializer), dtype=np.float64)
        absolute = np.abs(values)
        initializer_count += 1
        total += int(values.size)
        above_fp16_max += int(np.count_nonzero(absolute > init_max))
        below_fp16_subnormal += int(
            np.count_nonzero((absolute > 0.0) & (absolute < fp16_smallest_subnormal))
        )
        if absolute.size:
            maximum = max(maximum, float(np.max(absolute)))
    return {
        "initializer_count": initializer_count,
        "element_count": total,
        "max_abs": maximum,
        "above_fp16_max_count": above_fp16_max,
        "nonzero_below_fp16_subnormal_count": below_fp16_subnormal,
        "fp16_smallest_subnormal": fp16_smallest_subnormal,
    }


def _io_contract(model) -> Dict[str, object]:
    import onnx

    def describe(value):
        tensor_type = value.type.tensor_type
        shape = []
        for dim in tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                shape.append(int(dim.dim_value))
            elif dim.HasField("dim_param"):
                shape.append(dim.dim_param)
            else:
                shape.append(None)
        return {
            "name": value.name,
            "dtype": onnx.TensorProto.DataType.Name(tensor_type.elem_type),
            "shape": shape,
        }

    return {
        "inputs": [describe(value) for value in model.graph.input],
        "outputs": [describe(value) for value in model.graph.output],
    }


def _remaining_fp32_compute(model) -> Dict[str, object]:
    """Describe non-Cast nodes still connected to explicit FP32 tensors."""
    import onnx

    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=False, data_prop=False)
    dtypes = {
        value.name: value.type.tensor_type.elem_type
        for value in list(inferred.graph.input)
        + list(inferred.graph.value_info)
        + list(inferred.graph.output)
        if value.type.HasField("tensor_type")
    }
    dtypes.update(
        {initializer.name: initializer.data_type for initializer in inferred.graph.initializer}
    )
    nodes = []
    for index, node in enumerate(inferred.graph.node):
        if node.op_type == "Cast":
            continue
        values = [name for name in list(node.input) + list(node.output) if name]
        if any(dtypes.get(name) == onnx.TensorProto.FLOAT for name in values):
            nodes.append(
                {
                    "name": node.name or f"{node.op_type}_{index}",
                    "op_type": node.op_type,
                }
            )
    return {
        "count": len(nodes),
        "op_counts": dict(sorted(Counter(node["op_type"] for node in nodes).items())),
        "nodes": nodes,
    }


def _load_calibration(calibration_path: Path, input_names: Sequence[str]) -> Dict[str, np.ndarray]:
    with np.load(calibration_path, allow_pickle=False) as archive:
        calibration = {name: np.asarray(archive[name]) for name in archive.files}
    missing = sorted(set(input_names) - set(calibration))
    extra = sorted(set(calibration) - set(input_names))
    if missing or extra:
        raise ValueError(
            "Calibration NPZ keys must exactly match ONNX inputs: "
            f"missing={missing}, extra={extra}."
        )
    batch_sizes = {int(value.shape[0]) for value in calibration.values() if value.ndim}
    if len(batch_sizes) != 1:
        raise ValueError(f"Calibration inputs have inconsistent batch sizes: {sorted(batch_sizes)}")
    if not batch_sizes or next(iter(batch_sizes)) <= 0:
        raise ValueError("Calibration data must have a positive batch dimension.")
    return calibration


def _default_converter(**kwargs):
    from modelopt.onnx.autocast import convert_to_mixed_precision

    return convert_to_mixed_precision(**kwargs)


def apply_dataflow_fp16_precision(
    model_path: Path,
    calibration_path: Path,
    report_path: Optional[Path] = None,
    data_max: float = DEFAULT_DATA_MAX,
    init_max: float = DEFAULT_INIT_MAX,
    fp32_node_patterns: Sequence[str] = DEFAULT_FP32_NODE_PATTERNS,
    converter: Optional[Callable[..., object]] = None,
) -> Dict[str, object]:
    """Convert a model using real activation ranges and explicit FP32 islands."""
    import onnx

    model_path = Path(model_path)
    calibration_path = Path(calibration_path)
    if data_max <= 0.0 or init_max <= 0.0:
        raise ValueError("data_max and init_max must be positive.")

    source_sha256 = _sha256_file(model_path)
    source_model = onnx.load(str(model_path), load_external_data=False)
    input_names = [value.name for value in source_model.graph.input]
    calibration = _load_calibration(calibration_path, input_names)
    batch_size = next(iter(calibration.values())).shape[0]

    static_model = onnx.ModelProto()
    static_model.CopyFrom(source_model)
    _set_static_batch(static_model, int(batch_size))

    converter = converter or _default_converter
    with tempfile.TemporaryDirectory(prefix="edgecrafter_fp16_", dir=str(model_path.parent)) as temp_dir:
        static_path = Path(temp_dir) / "calibration_static.onnx"
        onnx.save(static_model, str(static_path), save_as_external_data=False)
        converted_model = converter(
            onnx_path=str(static_path),
            low_precision_type="fp16",
            nodes_to_exclude=list(fp32_node_patterns),
            data_max=float(data_max),
            init_max=float(init_max),
            keep_io_types=True,
            calibration_data={name: value for name, value in calibration.items()},
            providers=["cpu"],
            opset=int(source_model.opset_import[0].version),
        )

    _restore_io_contract(source_model, converted_model)
    _set_metadata(
        converted_model,
        {
            PRECISION_POLICY_METADATA_KEY: DATAFLOW_FP16_POLICY,
            "edgecrafter.precision_calibration_sha256": _sha256_file(calibration_path),
            "edgecrafter.precision_data_max": str(float(data_max)),
            "edgecrafter.precision_init_max": str(float(init_max)),
        },
    )
    onnx.checker.check_model(converted_model)
    onnx.save(converted_model, str(model_path), save_as_external_data=False)

    activation_stats = {
        name: {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "min": float(np.min(value)),
            "max": float(np.max(value)),
            "max_abs": float(np.max(np.abs(value))),
        }
        for name, value in calibration.items()
    }
    report = {
        "schema_version": 1,
        "policy": DATAFLOW_FP16_POLICY,
        "converter": "NVIDIA ModelOpt AutoCast",
        "modelopt_version": (
            metadata.version("nvidia-modelopt")
            if converter is _default_converter
            else "test-or-custom-converter"
        ),
        "source_model_sha256": source_sha256,
        "output_model_sha256": _sha256_file(model_path),
        "output_model_bytes": model_path.stat().st_size,
        "calibration": {
            "path": str(calibration_path.resolve()),
            "sha256": _sha256_file(calibration_path),
            "batch_size": int(batch_size),
            "inputs": activation_stats,
        },
        "thresholds": {"data_max": float(data_max), "init_max": float(init_max)},
        "fp32_node_patterns": list(fp32_node_patterns),
        "source_float_weights": _source_float_weight_stats(source_model, float(init_max)),
        "source_initializers": _tensor_dtype_summary(source_model),
        "converted_initializers": _tensor_dtype_summary(converted_model),
        "graph": {
            "source_node_count": len(source_model.graph.node),
            "converted_node_count": len(converted_model.graph.node),
            "cast_node_count": sum(node.op_type == "Cast" for node in converted_model.graph.node),
            "remaining_fp32_compute": _remaining_fp32_compute(converted_model),
        },
        "io_contract": _io_contract(converted_model),
        "validation": {
            "onnx_checker": "passed",
            "trt_build_required": True,
            "coco_parity_required": True,
        },
    }
    report_path = Path(report_path) if report_path else model_path.with_suffix(".fp16-report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report["report"] = str(report_path)
    return report
