"""Create and validate an explicit FP8 Q/DQ ONNX model with NVIDIA ModelOpt."""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
from importlib import metadata
import json
from pathlib import Path
import sys
from typing import Dict, Optional, Sequence

import numpy as np
from packaging.version import InvalidVersion, Version


MODEL_OPT_QDQ_POLICY = "modelopt_fp8_qdq_autotune_v1"
MODEL_OPT_QDQ_METADATA_KEY = "edgecrafter.quantization_policy"
MODEL_OPT_FP8_SUPPORTED_OP_TYPES = frozenset({"Add", "Conv", "Gemm", "MatMul"})
MODEL_OPT_AUTOTUNE_VERSION = (0, 46)


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _shape(value) -> tuple[int, ...]:
    return tuple(
        int(dim.dim_value) if dim.dim_value > 0 else -1
        for dim in value.type.tensor_type.shape.dim
    )


def _io_contract(onnx, model) -> Dict[str, object]:
    def collect(values):
        return {
            value.name: {
                "dtype": onnx.TensorProto.DataType.Name(value.type.tensor_type.elem_type),
                "shape": _shape(value),
            }
            for value in values
        }

    return {"inputs": collect(model.graph.input), "outputs": collect(model.graph.output)}


def _restore_external_value_info(source_model, converted_model) -> None:
    """Restore the source I/O ABI metadata after ModelOpt shape inference.

    ModelOpt FP16 autocast may replace known output dimensions with symbolic
    dimensions even though the graph computation is unchanged. TensorRT and
    downstream callers rely on the exported source contract, so preserve its
    input/output ValueInfo verbatim.
    """
    for field in ("input", "output"):
        source_values = {value.name: value for value in getattr(source_model.graph, field)}
        converted_values = {value.name: value for value in getattr(converted_model.graph, field)}
        if converted_values.keys() != source_values.keys():
            raise ValueError(
                f"ModelOpt changed external {field} names: "
                f"got {sorted(converted_values)}, expected {sorted(source_values)}"
            )
        for name, source_value in source_values.items():
            converted_values[name].type.CopyFrom(source_value.type)


def summarize_qdq_graph(model) -> Dict[str, object]:
    import onnx

    producers = {
        output: node
        for node in model.graph.node
        for output in node.output
        if output
    }
    initializers = {initializer.name: initializer for initializer in model.graph.initializer}
    quantize_nodes = [node for node in model.graph.node if node.op_type == "QuantizeLinear"]
    dequantize_nodes = [node for node in model.graph.node if node.op_type == "DequantizeLinear"]
    qdq_pairs = sum(
        1
        for node in dequantize_nodes
        if node.input and producers.get(node.input[0]) is not None
        and producers[node.input[0]].op_type == "QuantizeLinear"
    )
    quantized_types: Counter[str] = Counter()
    for node in quantize_nodes + dequantize_nodes:
        if len(node.input) < 3:
            continue
        zero_point = initializers.get(node.input[2])
        if zero_point is not None:
            quantized_types[onnx.TensorProto.DataType.Name(zero_point.data_type)] += 1

    consumers = {}
    for node in model.graph.node:
        for input_name in node.input:
            consumers.setdefault(input_name, []).append(node)
    qdq_adjacent_compute_types: Counter[str] = Counter()
    quantized_weight_compute_types: Counter[str] = Counter()
    for node in model.graph.node:
        if node.op_type not in MODEL_OPT_FP8_SUPPORTED_OP_TYPES:
            continue
        has_dequantized_input = any(
            producers.get(input_name) is not None
            and producers[input_name].op_type == "DequantizeLinear"
            for input_name in node.input
        )
        has_quantized_output = any(
            consumer.op_type == "QuantizeLinear"
            for output_name in node.output
            for consumer in consumers.get(output_name, ())
        )
        if has_dequantized_input or has_quantized_output:
            qdq_adjacent_compute_types[node.op_type] += 1
        has_quantized_weight = any(
            producers.get(input_name) is not None
            and producers[input_name].op_type == "DequantizeLinear"
            and producers.get(producers[input_name].input[0]) is not None
            and producers[producers[input_name].input[0]].op_type == "QuantizeLinear"
            and producers[producers[input_name].input[0]].input[0] in initializers
            for input_name in node.input
        )
        if has_quantized_weight:
            quantized_weight_compute_types[node.op_type] += 1

    default_opset = next(
        (int(opset.version) for opset in model.opset_import if opset.domain == ""),
        0,
    )
    return {
        "quantize_linear_nodes": len(quantize_nodes),
        "dequantize_linear_nodes": len(dequantize_nodes),
        "adjacent_qdq_pairs": qdq_pairs,
        "quantized_zero_point_types": dict(sorted(quantized_types.items())),
        "quantized_weight_compute_nodes": dict(sorted(quantized_weight_compute_types.items())),
        "qdq_adjacent_compute_nodes": dict(sorted(qdq_adjacent_compute_types.items())),
        "default_opset": default_opset,
        "total_nodes": len(model.graph.node),
    }


def _set_metadata(model, values: Dict[str, str]) -> None:
    current = {entry.key: entry.value for entry in model.metadata_props}
    current.update(values)
    model.ClearField("metadata_props")
    for key, value in sorted(current.items()):
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = value


def _calibration_summary(calibration_data: Dict[str, np.ndarray]) -> Dict[str, object]:
    return {
        name: {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "min": float(np.min(array)),
            "max": float(np.max(array)),
        }
        for name, array in calibration_data.items()
    }


def require_modelopt_autotune_version() -> str:
    """Require the ModelOpt release whose Autotune API this adapter targets."""
    try:
        raw_version = metadata.version("nvidia-modelopt")
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "nvidia-modelopt is not installed; install nvidia-modelopt[onnx]==0.46.*."
        ) from exc
    try:
        parsed = Version(raw_version)
    except InvalidVersion as exc:
        raise RuntimeError(f"Invalid nvidia-modelopt version: {raw_version!r}") from exc
    if parsed.release[:2] != MODEL_OPT_AUTOTUNE_VERSION:
        raise RuntimeError(
            "This exporter requires nvidia-modelopt 0.46.x because it relies on that "
            f"release's ONNX Autotune contract; found {raw_version}."
        )
    return raw_version


def _configure_windows_utf8_logging() -> None:
    """Prevent ModelOpt's Unicode log messages from failing on cp1251 consoles."""
    if sys.platform != "win32":
        return
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def _autotune_summary(output_dir: Path, state_file: Path) -> Dict[str, object]:
    import yaml

    state = {}
    if state_file.is_file():
        loaded = yaml.safe_load(state_file.read_text(encoding="utf-8"))
        state = loaded if isinstance(loaded, dict) else {}
    patterns = state.get("patterns") if isinstance(state.get("patterns"), list) else []
    schemes = [
        scheme
        for pattern in patterns
        if isinstance(pattern, dict)
        for scheme in pattern.get("schemes", [])
        if isinstance(scheme, dict)
    ]
    successful_latencies = [
        float(scheme["latency_ms"])
        for scheme in schemes
        if scheme.get("latency_ms") is not None
        and np.isfinite(float(scheme["latency_ms"]))
    ]
    region_models = output_dir / "region_models"
    logs = output_dir / "logs"
    return {
        "output_dir": str(output_dir.resolve()),
        "state_file": str(state_file.resolve()),
        "baseline_model": str((output_dir / "baseline.onnx").resolve()),
        "optimized_pattern_model": str((output_dir / "optimized_final.onnx").resolve()),
        "baseline_latency_ms": state.get("baseline_latency_ms"),
        "profiled_patterns": len(patterns),
        "tested_schemes": len(schemes),
        "successful_schemes": len(successful_latencies),
        "best_scheme_latency_ms": min(successful_latencies) if successful_latencies else None,
        "region_models": len(list(region_models.glob("*.onnx"))) if region_models.is_dir() else 0,
        "log_files": len(list(logs.glob("*.log"))) if logs.is_dir() else 0,
    }


def apply_modelopt_fp8_qdq(
    source_path: Path,
    output_path: Path,
    calibration_path: Path,
    *,
    calibration_shapes: str,
    calibration_method: str = "entropy",
    calibration_eps: Sequence[str] = ("cpu",),
    high_precision_dtype: str = "fp32",
    mha_accumulation_dtype: str = "fp32",
    nodes_to_exclude: Sequence[str] = (),
    nodes_to_quantize: Sequence[str] = (),
    op_types_to_quantize: Sequence[str] = (),
    op_types_to_exclude: Sequence[str] = (),
    op_types_to_exclude_fp16: Sequence[str] = (),
    disable_mha_qdq: bool = False,
    keep_intermediate_files: bool = False,
    enable_gemv_detection_for_trt: bool = False,
    autotune: bool = True,
    autotune_output_dir: Optional[Path] = None,
    autotune_num_schemes_per_region: int = 30,
    autotune_pattern_cache_file: Optional[Path] = None,
    autotune_state_file: Optional[Path] = None,
    autotune_qdq_baseline: Optional[Path] = None,
    autotune_node_filter_list: Sequence[str] = (),
    autotune_verbose: bool = False,
    autotune_use_trtexec: bool = False,
    autotune_timing_cache: Optional[Path] = None,
    autotune_warmup_runs: int = 10,
    autotune_timing_runs: int = 50,
    autotune_trtexec_args: Optional[str] = None,
    require_fp8_qdq: bool = False,
    log_level: str = "INFO",
    report_path: Optional[Path] = None,
) -> Dict[str, object]:
    """Quantize a source ONNX model to FP8 with explicit Q/DQ pairs.

    ``dq_only=False`` is intentional: the deployment contract requires explicit
    QuantizeLinear/DequantizeLinear pairs. In Autotune mode ModelOpt discovers
    the node set; a manual node allow-list is rejected because it overrides that
    set. An op-type list may still constrain the discovered nodes to ModelOpt's
    documented FP8-supported operators.
    """
    import onnx
    from modelopt.onnx.quantization import quantize

    _configure_windows_utf8_logging()
    modelopt_version = require_modelopt_autotune_version()

    source_path = Path(source_path)
    output_path = Path(output_path)
    calibration_path = Path(calibration_path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Source ONNX does not exist: {source_path}")
    if not calibration_path.is_file():
        raise FileNotFoundError(f"Calibration data does not exist: {calibration_path}")
    if calibration_method not in {"entropy", "max"}:
        raise ValueError(f"Unsupported FP8 calibration method: {calibration_method}")
    if high_precision_dtype not in {"fp32", "fp16"}:
        raise ValueError(f"Unsupported high precision dtype: {high_precision_dtype}")
    unsupported_op_types = set(op_types_to_quantize) - MODEL_OPT_FP8_SUPPORTED_OP_TYPES
    if unsupported_op_types:
        raise ValueError(f"Unsupported ModelOpt FP8 op types: {sorted(unsupported_op_types)}")
    if autotune and nodes_to_quantize:
        raise ValueError(
            "ModelOpt Autotune must own nodes_to_quantize; use exclusions or "
            "autotune_node_filter_list to constrain the search."
        )
    if autotune_num_schemes_per_region <= 0:
        raise ValueError("Autotune schemes per region must be positive.")
    if autotune_warmup_runs < 0 or autotune_timing_runs <= 0:
        raise ValueError("Autotune warmup runs must be non-negative and timing runs positive.")

    autotune_output_path = Path(autotune_output_dir) if autotune_output_dir else output_path.parent / "autotune"
    state_path = (
        Path(autotune_state_file)
        if autotune_state_file
        else autotune_output_path / "autotuner_state.yaml"
    )
    for candidate, label in (
        (autotune_pattern_cache_file, "Autotune pattern cache"),
        (autotune_qdq_baseline, "Autotune Q/DQ baseline"),
    ):
        if candidate is not None and not Path(candidate).is_file():
            raise FileNotFoundError(f"{label} does not exist: {candidate}")
    if autotune:
        autotune_output_path.mkdir(parents=True, exist_ok=True)
        state_path.parent.mkdir(parents=True, exist_ok=True)
    if autotune_timing_cache is not None:
        Path(autotune_timing_cache).parent.mkdir(parents=True, exist_ok=True)

    source_model = onnx.load(str(source_path), load_external_data=False)
    source_contract = _io_contract(onnx, source_model)
    with np.load(calibration_path, allow_pickle=False) as archive:
        calibration_data = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
    if not calibration_data:
        raise ValueError(f"Calibration archive contains no tensors: {calibration_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    quantize(
        onnx_path=str(source_path),
        quantize_mode="fp8",
        calibration_data=calibration_data,
        calibration_method=calibration_method,
        calibration_shapes=calibration_shapes,
        calibration_eps=list(calibration_eps),
        nodes_to_quantize=list(nodes_to_quantize),
        nodes_to_exclude=list(nodes_to_exclude),
        op_types_to_quantize=list(op_types_to_quantize) or None,
        op_types_to_exclude=list(op_types_to_exclude),
        op_types_to_exclude_fp16=list(op_types_to_exclude_fp16),
        output_path=str(output_path),
        high_precision_dtype=high_precision_dtype,
        mha_accumulation_dtype=mha_accumulation_dtype,
        dq_only=False,
        keep_intermediate_files=keep_intermediate_files,
        direct_io_types=False,
        disable_mha_qdq=disable_mha_qdq,
        # ModelOpt runs this optional pre-scan before applying
        # nodes_to_exclude. Its shape inference rejects EdgeCrafter's dynamic
        # Integral MatMul, even though that node is explicitly excluded below.
        enable_gemv_detection_for_trt=enable_gemv_detection_for_trt,
        autotune=autotune,
        autotune_output_dir=str(autotune_output_path) if autotune else None,
        autotune_num_schemes_per_region=autotune_num_schemes_per_region,
        autotune_pattern_cache_file=(
            str(autotune_pattern_cache_file) if autotune_pattern_cache_file else None
        ),
        autotune_state_file=str(state_path) if autotune else None,
        autotune_qdq_baseline=str(autotune_qdq_baseline) if autotune_qdq_baseline else None,
        autotune_node_filter_list=list(autotune_node_filter_list) or None,
        autotune_verbose=autotune_verbose,
        autotune_use_trtexec=autotune_use_trtexec,
        autotune_timing_cache=str(autotune_timing_cache) if autotune_timing_cache else None,
        autotune_warmup_runs=autotune_warmup_runs,
        autotune_timing_runs=autotune_timing_runs,
        autotune_trtexec_args=autotune_trtexec_args,
        log_level=log_level,
    )
    if not output_path.is_file():
        raise RuntimeError(f"ModelOpt did not create the requested ONNX file: {output_path}")

    converted_model = onnx.load(str(output_path), load_external_data=False)
    _restore_external_value_info(source_model, converted_model)
    converted_contract = _io_contract(onnx, converted_model)
    if converted_contract != source_contract:
        raise ValueError(
            "ModelOpt changed the external ONNX ABI: "
            f"got {converted_contract}, expected {source_contract}"
        )

    summary = summarize_qdq_graph(converted_model)
    selected_fp8 = (
        summary["quantize_linear_nodes"] > 0
        and summary["dequantize_linear_nodes"] > 0
        and summary["quantized_zero_point_types"].get("FLOAT8E4M3FN", 0) > 0
    )
    if require_fp8_qdq and not selected_fp8:
        raise ValueError(
            "ModelOpt Autotune selected no FP8 Q/DQ regions on this target; "
            f"strict FP8 was requested: {summary}"
        )
    if selected_fp8 and summary["adjacent_qdq_pairs"] <= 0:
        raise ValueError(
            f"ModelOpt output has FP8 tensors but no adjacent QuantizeLinear/DequantizeLinear pair: {summary}"
        )
    if summary["default_opset"] < 19:
        raise ValueError(f"FP8 Q/DQ requires ONNX opset 19 or newer: {summary}")

    _set_metadata(
        converted_model,
        {
            MODEL_OPT_QDQ_METADATA_KEY: MODEL_OPT_QDQ_POLICY,
            "edgecrafter.modelopt_version": modelopt_version,
            "edgecrafter.quantization_mode": "fp8",
            "edgecrafter.high_precision_dtype": high_precision_dtype,
            "edgecrafter.autotune_selected_fp8": str(selected_fp8).lower(),
        },
    )
    onnx.checker.check_model(converted_model)
    onnx.save(converted_model, str(output_path), save_as_external_data=False)

    report = {
        "schema_version": 1,
        "policy": MODEL_OPT_QDQ_POLICY,
        "quantization_mode": "fp8",
        "selection": "FP8_SELECTED" if selected_fp8 else "NO_FP8_BENEFIT",
        "selected_fp8": selected_fp8,
        "require_fp8_qdq": require_fp8_qdq,
        "high_precision_dtype": high_precision_dtype,
        "mha_accumulation_dtype": mha_accumulation_dtype,
        "modelopt_version": modelopt_version,
        "calibration_method": calibration_method,
        "calibration_eps": list(calibration_eps),
        "calibration_shapes": calibration_shapes,
        "calibration": _calibration_summary(calibration_data),
        "nodes_to_exclude": list(nodes_to_exclude),
        "nodes_to_quantize": list(nodes_to_quantize),
        "op_types_to_quantize": list(op_types_to_quantize),
        "op_types_to_exclude": list(op_types_to_exclude),
        "op_types_to_exclude_fp16": list(op_types_to_exclude_fp16),
        "disable_mha_qdq": disable_mha_qdq,
        "enable_gemv_detection_for_trt": enable_gemv_detection_for_trt,
        "autotune": {
            "enabled": autotune,
            "num_schemes_per_region": autotune_num_schemes_per_region,
            "node_filter_list": list(autotune_node_filter_list),
            "pattern_cache_file": (
                str(Path(autotune_pattern_cache_file).resolve())
                if autotune_pattern_cache_file
                else None
            ),
            "qdq_baseline": (
                str(Path(autotune_qdq_baseline).resolve()) if autotune_qdq_baseline else None
            ),
            "use_trtexec": autotune_use_trtexec,
            "timing_cache": (
                str(Path(autotune_timing_cache).resolve()) if autotune_timing_cache else None
            ),
            "warmup_runs": autotune_warmup_runs,
            "timing_runs": autotune_timing_runs,
            "trtexec_args": autotune_trtexec_args,
            "artifacts": _autotune_summary(autotune_output_path, state_path) if autotune else None,
        },
        "source": {
            "path": str(source_path.resolve()),
            "bytes": source_path.stat().st_size,
            "sha256": _sha256_file(source_path),
            "io_contract": source_contract,
        },
        "output": {
            "path": str(output_path.resolve()),
            "bytes": output_path.stat().st_size,
            "sha256": _sha256_file(output_path),
            "io_contract": converted_contract,
        },
        "graph": summary,
    }
    report_path = Path(report_path) if report_path else output_path.with_suffix(".modelopt-report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report["report"] = str(report_path.resolve())
    return report
