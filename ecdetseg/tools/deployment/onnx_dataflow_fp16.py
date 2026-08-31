"""Build a calibrated, dataflow-aware explicit FP16 ONNX graph."""

from collections import Counter, OrderedDict
import copy
from hashlib import sha256
from importlib import import_module
from importlib import metadata
import json
from pathlib import Path
import tempfile
import threading
from typing import Callable, Dict, List, Optional, Sequence

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

_MODELOPT_REFERENCE_RUNNER_LOCK = threading.Lock()


def modelopt_gpu_first_providers(gpu: int) -> List[str]:
    """Return ModelOpt's ORT provider priority for one visible CUDA device."""
    if gpu < 0:
        raise ValueError(f"ModelOpt calibration GPU index must be non-negative, got {gpu}.")
    return [f"cuda:{gpu}", "cpu"]


def _require_cuda_execution_provider(providers: Sequence[str]) -> None:
    cuda_requested = any(provider == "cuda" or provider.startswith("cuda:") for provider in providers)
    if not cuda_requested:
        return

    import onnxruntime as ort

    available = list(ort.get_available_providers())
    if "CUDAExecutionProvider" not in available:
        raise RuntimeError(
            "GPU-first ModelOpt calibration requires CUDAExecutionProvider, but this Python "
            f"environment exposes only {available}. Install a compatible onnxruntime-gpu "
            "package in the same environment."
        )


class _CalibrationBatchStream(dict):
    """Keep one calibration corpus while exposing bounded inference batches."""

    def __init__(self, values: Dict[str, np.ndarray], batch_size: int) -> None:
        super().__init__(values)
        self.batch_size = int(batch_size)
        self.sample_count = int(next(iter(values.values())).shape[0])
        if self.batch_size <= 0:
            raise ValueError("Calibration batch size must be positive.")
        if self.sample_count % self.batch_size != 0:
            raise ValueError(
                "Calibration sample count must be divisible by calibration batch size: "
                f"samples={self.sample_count}, batch_size={self.batch_size}."
            )

    @property
    def batch_count(self) -> int:
        return self.sample_count // self.batch_size

    def iter_batches(self):
        for start in range(0, self.sample_count, self.batch_size):
            stop = start + self.batch_size
            yield OrderedDict(
                (name, np.ascontiguousarray(value[start:stop]))
                for name, value in self.items()
            )


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


def _update_streaming_stats(aggregated, values, tensor_stats_type) -> None:
    for name, value in values.items():
        array = np.asarray(value)
        if not (
            np.issubdtype(array.dtype, np.number)
            or np.issubdtype(array.dtype, np.bool_)
        ):
            continue
        if array.size:
            with np.errstate(over="ignore", invalid="ignore"):
                batch_absmax = float(np.max(np.abs(array)))
                batch_min = float(np.min(array))
                batch_max = float(np.max(array))
        else:
            batch_absmax = batch_min = batch_max = 0.0

        current = aggregated.get(name)
        if current is None:
            aggregated[name] = tensor_stats_type(
                absmax=batch_absmax,
                min_val=batch_min,
                max_val=batch_max,
                shape=tuple(array.shape),
            )
        else:
            current.absmax = max(current.absmax, batch_absmax)
            current.min_val = min(current.min_val, batch_min)
            current.max_val = max(current.max_val, batch_max)


def _run_modelopt_reference_streaming(reference_runner, calibration: _CalibrationBatchStream):
    """Run ModelOpt's all-output ORT model without retaining every batch output."""
    import onnxruntime as ort
    from modelopt.onnx import utils as onnx_utils
    from modelopt.onnx.autocast.referencerunner import TensorStats
    from polygraphy import constants
    from polygraphy.backend.onnx import ModifyOutputs as ModifyOnnxOutputs

    ort.set_default_logger_severity(3)
    model_copy = copy.deepcopy(reference_runner.model)
    onnx_utils.clear_stale_value_info(model_copy)
    modified_model = ModifyOnnxOutputs(model_copy, outputs=constants.MARK_ALL)
    runners = reference_runner._get_ort_runner(modified_model)
    if len(runners) != 1:
        raise RuntimeError(
            f"ModelOpt streaming calibration expected one ORT runner, got {len(runners)}."
        )

    print(
        "ModelOpt AutoCast streaming calibration: "
        f"{calibration.sample_count} samples, {calibration.batch_count} batches, "
        f"batch_size={calibration.batch_size}."
    )
    aggregated = OrderedDict()
    single_batch_data = None
    runner = runners[0]
    runner.activate()
    try:
        for batch_index, feed_dict in enumerate(calibration.iter_batches()):
            if batch_index == 0:
                reference_runner._validate_inputs([feed_dict])
            outputs = runner.infer(feed_dict, check_inputs=batch_index == 0)
            combined = OrderedDict(feed_dict)
            combined.update(outputs)
            _update_streaming_stats(aggregated, combined, TensorStats)
            if calibration.batch_count == 1:
                single_batch_data = combined
            else:
                del combined
                del outputs
    finally:
        runner.deactivate()

    if single_batch_data is not None:
        return single_batch_data
    return aggregated


def _default_converter(**kwargs):
    calibration_data = kwargs.get("calibration_data")
    if not isinstance(calibration_data, _CalibrationBatchStream):
        from modelopt.onnx.autocast import convert_to_mixed_precision

        return convert_to_mixed_precision(**kwargs)

    # ModelOpt 0.46 supports multiple calibration batches, but its reference
    # runner first retains every marked intermediate output and only aggregates
    # ranges after all batches finish. For detector graphs that is effectively
    # O(samples * all_intermediate_tensors) RAM. Replace only the converter's
    # private runner class for this call and aggregate each batch immediately.
    convert_module = import_module("modelopt.onnx.autocast.convert")
    reference_runner_type = convert_module.ReferenceRunner
    required_methods = ("_get_ort_runner", "_validate_inputs")
    missing = [name for name in required_methods if not hasattr(reference_runner_type, name)]
    if missing:
        raise RuntimeError(
            "Installed ModelOpt ReferenceRunner is incompatible with bounded-memory "
            f"calibration; missing methods: {missing}."
        )

    class StreamingReferenceRunner(reference_runner_type):
        def run(self, inputs=None):
            if isinstance(inputs, _CalibrationBatchStream):
                return _run_modelopt_reference_streaming(self, inputs)
            return super().run(inputs)

    with _MODELOPT_REFERENCE_RUNNER_LOCK:
        convert_module.ReferenceRunner = StreamingReferenceRunner
        try:
            return convert_module.convert_to_mixed_precision(**kwargs)
        finally:
            convert_module.ReferenceRunner = reference_runner_type


def apply_dataflow_fp16_precision(
    model_path: Path,
    calibration_path: Path,
    report_path: Optional[Path] = None,
    data_max: float = DEFAULT_DATA_MAX,
    init_max: float = DEFAULT_INIT_MAX,
    calibration_batch_size: int = 1,
    fp32_node_patterns: Sequence[str] = DEFAULT_FP32_NODE_PATTERNS,
    providers: Optional[Sequence[str]] = None,
    converter: Optional[Callable[..., object]] = None,
) -> Dict[str, object]:
    """Convert a model using real activation ranges and explicit FP32 islands."""
    import onnx

    model_path = Path(model_path)
    calibration_path = Path(calibration_path)
    if data_max <= 0.0 or init_max <= 0.0:
        raise ValueError("data_max and init_max must be positive.")
    if isinstance(providers, str):
        raise TypeError("ModelOpt providers must be a sequence of provider names, not one string.")
    requested_providers = list(providers) if providers is not None else modelopt_gpu_first_providers(0)
    if not requested_providers:
        raise ValueError("At least one ModelOpt ONNX Runtime provider is required.")

    source_sha256 = _sha256_file(model_path)
    source_model = onnx.load(str(model_path), load_external_data=False)
    input_names = [value.name for value in source_model.graph.input]
    calibration = _load_calibration(calibration_path, input_names)
    calibration_stream = _CalibrationBatchStream(calibration, calibration_batch_size)

    static_model = onnx.ModelProto()
    static_model.CopyFrom(source_model)
    _set_static_batch(static_model, calibration_stream.batch_size)

    if converter is None:
        _require_cuda_execution_provider(requested_providers)
    converter = converter or _default_converter
    print(f"ModelOpt AutoCast ORT provider priority: {requested_providers}")
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
            calibration_data=calibration_stream,
            providers=requested_providers,
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
            "requested_providers": requested_providers,
            "sample_count": calibration_stream.sample_count,
            "batch_size": calibration_stream.batch_size,
            "batch_count": calibration_stream.batch_count,
            "streaming": converter is _default_converter,
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
