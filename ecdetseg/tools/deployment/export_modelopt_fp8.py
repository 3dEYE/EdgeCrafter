"""One-command EdgeCrafter checkpoint -> ModelOpt FP8 Q/DQ ONNX -> TensorRT export."""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

SCRIPT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPT_ROOT))

import torch

from engine.core import YAMLConfig
from engine.core.yaml_utils import merge_dict
from tools.deployment.export_trt_eval import (
    TRTInference,
    _configured_num_top_queries,
    _parse_input_size,
    _read_engine_contract,
    _read_onnx_contract,
    _sha256_file,
    _validate_batch_request,
    _validate_engine_contract,
    _validate_engine_top_queries,
    _validate_onnx_contract,
    _write_engine_manifest,
    build_engine,
    evaluate_engine,
    export_onnx,
    resolve_yolo_data,
    write_modelopt_calibration_data,
)
from tools.deployment.onnx_dataflow_fp16 import (
    DATAFLOW_FP16_POLICY,
    DEFAULT_FP32_NODE_PATTERNS,
    apply_dataflow_fp16_precision,
    modelopt_gpu_first_providers,
)
from tools.deployment.onnx_modelopt_qdq import (
    MODEL_OPT_FP8_SUPPORTED_OP_TYPES,
    apply_modelopt_fp8_qdq,
    require_modelopt_autotune_version,
)
from tools.deployment.onnx_precision import SENSITIVE_FP32_OP_TYPES


COCO_BBOX_NAMES = (
    "map_50_95", "map_50", "map_75", "map_small", "map_medium", "map_large",
    "ar_1", "ar_10", "ar_100", "ar_small", "ar_medium", "ar_large",
)
MODEL_OPT_AUTOTUNE_PRESETS = {
    "quick": {"num_schemes_per_region": 30, "warmup_runs": 10, "timing_runs": 50},
    "default": {"num_schemes_per_region": 50, "warmup_runs": 50, "timing_runs": 100},
    "extensive": {"num_schemes_per_region": 200, "warmup_runs": 50, "timing_runs": 200},
}


def parse_gpu_selector(value: str):
    if value.lower() == "auto-fp8":
        return "auto-fp8"
    try:
        index = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--gpu must be a non-negative index or 'auto-fp8'.") from exc
    if index < 0:
        raise argparse.ArgumentTypeError("--gpu index must be non-negative.")
    return index


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export an explicitly selected EdgeCrafter checkpoint, calibrate it on real validation images, "
            "build calibrated FP16 dataflow, let NVIDIA ModelOpt 0.46 Autotune select profitable "
            "FP8 QuantizeLinear/DequantizeLinear regions on the target GPU, validate the ONNX ABI, "
            "and build a strongly typed TensorRT engine. Autotune may select no FP8 on RTX 4090."
        )
    )
    required = parser.add_argument_group("required inputs")
    required.add_argument("--data", required=True, help="YOLO dataset directory or data.yaml path.")
    required.add_argument("--config", required=True, help="Model YAML matching the checkpoint.")
    required.add_argument("--checkpoint", required=True, help="Checkpoint loaded with strict validation.")

    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--input-size", nargs="+", type=int, default=None)
    parser.add_argument("--opset", type=int, default=20)
    parser.add_argument("--calibration-samples", type=int, default=128)
    parser.add_argument("--calibration-batch-size", type=int, default=1)
    parser.add_argument("--calibration-method", choices=["entropy", "max"], default="entropy")
    parser.add_argument(
        "--calibration-eps",
        nargs="+",
        default=["auto"],
        help="ModelOpt calibration EP priority: auto, cpu, cuda[:id], or trt.",
    )
    parser.add_argument("--mha-accumulation-dtype", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument(
        "--exclude-node-regex",
        action="append",
        default=[],
        help="Additional ModelOpt node-name regex to keep outside FP8; may be repeated.",
    )
    parser.add_argument(
        "--autotune-mode",
        choices=sorted(MODEL_OPT_AUTOTUNE_PRESETS),
        default="quick",
        help="ModelOpt 0.46 Autotune search budget preset.",
    )
    parser.add_argument(
        "--autotune-num-schemes-per-region",
        type=int,
        default=None,
        help="Override the preset's number of Q/DQ schemes tested per region.",
    )
    parser.add_argument(
        "--autotune-warmup-runs",
        type=int,
        default=None,
        help="Override the preset's TensorRT warmup runs.",
    )
    parser.add_argument(
        "--autotune-timing-runs",
        type=int,
        default=None,
        help="Override the preset's TensorRT timing runs.",
    )
    parser.add_argument(
        "--autotune-pattern-cache",
        default=None,
        help="Existing ModelOpt Autotune pattern-cache YAML used as a warm start.",
    )
    parser.add_argument(
        "--autotune-state-file",
        default=None,
        help="Optional ModelOpt Autotune state YAML to resume and update.",
    )
    parser.add_argument(
        "--autotune-qdq-baseline",
        default=None,
        help="Optional pre-quantized ONNX whose Q/DQ patterns seed Autotune.",
    )
    parser.add_argument(
        "--autotune-node-filter",
        action="append",
        default=[],
        help=(
            "Wildcard matched against ONNX node names; regions without a match are skipped. "
            "May be repeated."
        ),
    )
    parser.add_argument("--autotune-use-trtexec", action="store_true")
    parser.add_argument(
        "--autotune-trtexec-args",
        default=None,
        help="Additional quoted trtexec arguments used only with --autotune-use-trtexec.",
    )
    parser.add_argument(
        "--autotune-timing-cache",
        default=None,
        help="Persistent TensorRT timing-cache path for Autotune engine builds.",
    )
    parser.add_argument(
        "--exclude-op-type",
        action="append",
        choices=sorted(MODEL_OPT_FP8_SUPPORTED_OP_TYPES),
        default=[],
        help="Additional ONNX op type to keep outside FP8; may be repeated.",
    )
    parser.add_argument(
        "--disable-mha-qdq",
        action="store_true",
        help="Keep ModelOpt-detected MHA BMM1/BMM2 computations outside FP8.",
    )
    parser.add_argument(
        "--require-fp8-qdq",
        action="store_true",
        help=(
            "Fail when Autotune selects no FP8 regions. Off by default because FP16 may be "
            "faster on RTX 4090."
        ),
    )
    parser.add_argument("--keep-modelopt-intermediates", action="store_true")
    parser.add_argument("--min-batch", type=int, default=1)
    parser.add_argument("--opt-batch", type=int, default=4)
    parser.add_argument("--max-batch", type=int, default=8)
    parser.add_argument("--workspace-gb", type=float, default=8.0)
    parser.add_argument(
        "--gpu",
        type=parse_gpu_selector,
        default="auto-fp8",
        help=(
            "CUDA device index or 'auto-fp8' to select the first local GPU whose compute "
            "capability and TensorRT API support FP8."
        ),
    )
    parser.add_argument(
        "--expected-gpu-regex",
        default=None,
        help=(
            "Optional additional device-name filter, for example 'NVIDIA L4'. FP8 support "
            "is checked independently."
        ),
    )
    parser.add_argument("--onnx-only", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument(
        "--quality-reference",
        default=None,
        help="JSON with reference COCO metrics (coco_bbox or baseline.coco_bbox).",
    )
    parser.add_argument("--max-map-drop", type=float, default=0.005)
    parser.add_argument("--max-ar100-drop", type=float, default=0.005)
    parser.add_argument("--verbose-trt", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def resolve_calibration_eps(values: Sequence[str], gpu: int) -> List[str]:
    if "auto" in values and len(values) != 1:
        raise ValueError("Use --calibration-eps auto alone, or provide an explicit EP priority list.")
    if list(values) != ["auto"]:
        supported = all(
            value in {"cpu", "trt"} or value == "cuda" or re.fullmatch(r"cuda:\d+", value)
            for value in values
        )
        if not supported:
            raise ValueError(f"Unsupported ModelOpt calibration EP list: {list(values)}")
        return [f"cuda:{gpu}" if value == "cuda" else value for value in values]

    import onnxruntime as ort

    available = set(ort.get_available_providers())
    result = []
    if "TensorrtExecutionProvider" in available:
        result.append("trt")
    if "CUDAExecutionProvider" in available:
        result.append(f"cuda:{gpu}")
    result.append("cpu")
    return result


def _resolved_file(value: str, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} is not a file: {path}")
    return path


def _safe_stem(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "model"


def _output_dir(args, config: Path, checkpoint: Path) -> Path:
    if args.output_dir:
        path = Path(args.output_dir).expanduser().resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{stamp}_{_safe_stem(config.stem)}_{_safe_stem(checkpoint.stem)}"
        path = SCRIPT_ROOT / "outputs" / "modelopt_fp8_exports" / name
    path.mkdir(parents=True, exist_ok=False)
    return path


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def resolve_autotune_settings(args: argparse.Namespace) -> Dict[str, int]:
    settings = dict(MODEL_OPT_AUTOTUNE_PRESETS[args.autotune_mode])
    overrides = {
        "num_schemes_per_region": args.autotune_num_schemes_per_region,
        "warmup_runs": args.autotune_warmup_runs,
        "timing_runs": args.autotune_timing_runs,
    }
    settings.update({name: int(value) for name, value in overrides.items() if value is not None})
    if settings["num_schemes_per_region"] <= 0:
        raise ValueError("--autotune-num-schemes-per-region must be positive.")
    if settings["warmup_runs"] < 0:
        raise ValueError("--autotune-warmup-runs must be non-negative.")
    if settings["timing_runs"] <= 0:
        raise ValueError("--autotune-timing-runs must be positive.")
    return settings


def _build_cfg(
    config: Path,
    dataset_root: Path,
    data_file: Path,
    batch_size: int,
    input_size: Optional[List[int]],
) -> YAMLConfig:
    update: Dict[str, object] = {
        "yolo_root": str(dataset_root),
        "yolo_data_file": str(data_file),
        "val_dataloader": {"total_batch_size": int(batch_size)},
    }
    if input_size is not None:
        update = merge_dict(update, {"eval_spatial_size": input_size})
    return YAMLConfig(str(config), **update)


def _dataset_classes(data_cfg: dict) -> Optional[int]:
    if data_cfg.get("nc") is not None:
        return int(data_cfg["nc"])
    names = data_cfg.get("names")
    return len(names) if isinstance(names, (list, tuple, dict)) else None


def _validate_model_contract(cfg: YAMLConfig, data_cfg: dict) -> Tuple[int, Tuple[int, int], int]:
    if cfg.yaml_cfg.get("task") != "detection":
        raise ValueError(f"FP8 exporter currently supports detection only, got {cfg.yaml_cfg.get('task')!r}.")
    model_classes = int(cfg.yaml_cfg["num_classes"])
    dataset_classes = _dataset_classes(data_cfg)
    if dataset_classes != model_classes:
        raise ValueError(
            f"Dataset/model class mismatch: dataset={dataset_classes}, config={model_classes}."
        )
    image_size = _parse_input_size(list(cfg.yaml_cfg["eval_spatial_size"]))
    if image_size is None:
        raise ValueError("Model config has no eval_spatial_size.")
    return model_classes, (image_size[0], image_size[1]), _configured_num_top_queries(cfg)


def _cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _gpu_fingerprint(gpu: int) -> Optional[Dict[str, object]]:
    if not torch.cuda.is_available() or gpu < 0 or gpu >= torch.cuda.device_count():
        return None
    import tensorrt as trt

    properties = torch.cuda.get_device_properties(gpu)
    compute_capability = (properties.major, properties.minor)
    has_fp8_builder_flag = hasattr(trt.BuilderFlag, "FP8")
    has_fp8_hardware = compute_capability >= (8, 9)
    return {
        "index": gpu,
        "name": properties.name,
        "compute_capability": list(compute_capability),
        "total_memory_bytes": properties.total_memory,
        "tensorrt_version": trt.__version__,
        "torch_cuda_version": torch.version.cuda,
        "fp8_builder_flag": has_fp8_builder_flag,
        "fp8_hardware_capability": has_fp8_hardware,
        "fp8_supported": has_fp8_builder_flag and has_fp8_hardware,
    }


def validate_expected_gpu(
    fingerprint: Optional[Dict[str, object]],
    expected_gpu_regex: Optional[str],
) -> None:
    if expected_gpu_regex is None:
        return
    try:
        pattern = re.compile(expected_gpu_regex, re.IGNORECASE)
    except re.error as exc:
        raise ValueError(f"Invalid --expected-gpu-regex {expected_gpu_regex!r}: {exc}") from exc
    if fingerprint is None:
        raise RuntimeError(
            f"CUDA GPU matching {expected_gpu_regex!r} is required for ModelOpt Autotune."
        )
    if pattern.search(str(fingerprint["name"])) is None:
        raise RuntimeError(
            "Refusing to create a target-specific Autotune profile on the wrong GPU: "
            f"expected /{expected_gpu_regex}/, found {fingerprint['name']!r}."
        )


def resolve_autotune_gpu(gpu_selector, expected_gpu_regex: Optional[str]) -> Tuple[int, Dict[str, object]]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for ModelOpt FP8 Autotune.")
    if gpu_selector == "auto-fp8":
        indices = range(torch.cuda.device_count())
    else:
        if gpu_selector < 0 or gpu_selector >= torch.cuda.device_count():
            raise ValueError(
                f"--gpu {gpu_selector} is outside {torch.cuda.device_count()} visible GPU(s)."
            )
        indices = (gpu_selector,)

    inspected = []
    for gpu in indices:
        fingerprint = _gpu_fingerprint(gpu)
        assert fingerprint is not None
        inspected.append(fingerprint)
        try:
            validate_expected_gpu(fingerprint, expected_gpu_regex)
        except RuntimeError:
            if gpu_selector == "auto-fp8":
                continue
            raise
        if fingerprint["fp8_supported"]:
            return gpu, fingerprint
        if gpu_selector != "auto-fp8":
            raise RuntimeError(
                "Selected CUDA device does not expose the required FP8 hardware/TensorRT "
                f"contract: {fingerprint}"
            )
    raise RuntimeError(
        "No local CUDA device satisfies the FP8 hardware/TensorRT requirement"
        f" and optional name filter {expected_gpu_regex!r}: {inspected}"
    )


def _autotune_profile_sidecar(path: Path) -> Path:
    return path.with_name(path.name + ".edgecrafter-hardware.json")


def bind_autotune_cache_to_gpu(path: Path, fingerprint: Dict[str, object]) -> Path:
    """Bind resumable latency/timing state to the GPU and TensorRT profile."""
    sidecar = _autotune_profile_sidecar(path)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not sidecar.is_file():
        raise ValueError(
            f"Autotune cache has no EdgeCrafter hardware profile and cannot be safely reused: {path}"
        )
    if sidecar.is_file():
        saved = json.loads(sidecar.read_text(encoding="utf-8"))
        comparable_keys = (
            "name",
            "compute_capability",
            "total_memory_bytes",
            "tensorrt_version",
        )
        mismatches = {
            key: {"cache": saved.get(key), "current": fingerprint.get(key)}
            for key in comparable_keys
            if saved.get(key) != fingerprint.get(key)
        }
        if mismatches:
            raise ValueError(
                f"Autotune cache belongs to a different hardware profile: {mismatches}"
            )
    _write_json(sidecar, fingerprint)
    return sidecar


def _coco_stats(stats: Sequence[float]) -> Dict[str, float]:
    if len(stats) < len(COCO_BBOX_NAMES):
        raise ValueError("TensorRT evaluation did not return COCO bbox metrics.")
    return {name: float(stats[index]) for index, name in enumerate(COCO_BBOX_NAMES)}


def _reference_coco_metrics(path: Path) -> Dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidates = (
        payload.get("coco_bbox"),
        (payload.get("baseline") or {}).get("coco_bbox"),
        (payload.get("explicit_fp16_dataflow") or {}).get("coco_bbox"),
    )
    metrics = next((value for value in candidates if isinstance(value, dict)), None)
    if metrics is None:
        raise ValueError(
            f"Quality reference has no coco_bbox, baseline.coco_bbox, or "
            f"explicit_fp16_dataflow.coco_bbox: {path}"
        )
    missing = [name for name in ("map_50_95", "ar_100") if name not in metrics]
    if missing:
        raise ValueError(f"Quality reference is missing {missing}: {path}")
    return {name: float(value) for name, value in metrics.items() if name in COCO_BBOX_NAMES}


def validate_quality_reference_contract(
    path: Path,
    *,
    checkpoint: Path,
    image_hw: Tuple[int, int],
) -> Dict[str, object]:
    """Reject metrics captured with a different checkpoint or input shape."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    inputs = payload.get("inputs") if isinstance(payload.get("inputs"), dict) else {}
    reference_checkpoint_sha256 = payload.get("checkpoint_sha256") or inputs.get(
        "checkpoint_sha256"
    )
    actual_checkpoint_sha256 = _sha256_file(checkpoint)
    if (
        reference_checkpoint_sha256 is not None
        and reference_checkpoint_sha256 != actual_checkpoint_sha256
    ):
        raise ValueError(
            "Quality reference checkpoint differs from --checkpoint: "
            f"reference={reference_checkpoint_sha256}, actual={actual_checkpoint_sha256}."
        )

    reference_image_hw = None
    if isinstance(inputs.get("image_hw"), list) and len(inputs["image_hw"]) == 2:
        reference_image_hw = tuple(int(value) for value in inputs["image_hw"])
    settings = payload.get("settings") if isinstance(payload.get("settings"), dict) else {}
    input_contract = settings.get("input")
    if reference_image_hw is None and isinstance(input_contract, str):
        match = re.search(r"\[N,3,(\d+),(\d+)\]", input_contract)
        if match:
            reference_image_hw = (int(match.group(1)), int(match.group(2)))
    if reference_image_hw is not None and reference_image_hw != image_hw:
        raise ValueError(
            "Quality reference input size differs from this export: "
            f"reference={list(reference_image_hw)}, actual={list(image_hw)}."
        )

    return {
        "checkpoint_sha256": reference_checkpoint_sha256,
        "image_hw": list(reference_image_hw) if reference_image_hw is not None else None,
        "coco_bbox": _reference_coco_metrics(path),
    }


def assess_quality(
    reference: Dict[str, float],
    candidate: Dict[str, float],
    *,
    max_map_drop: float,
    max_ar100_drop: float,
) -> Dict[str, object]:
    if max_map_drop < 0 or max_ar100_drop < 0:
        raise ValueError("Quality drop tolerances must be non-negative.")
    checks = {
        "map_50_95_drop": {
            "reference": reference["map_50_95"],
            "candidate": candidate["map_50_95"],
            "drop": reference["map_50_95"] - candidate["map_50_95"],
            "maximum": max_map_drop,
        },
        "ar_100_drop": {
            "reference": reference["ar_100"],
            "candidate": candidate["ar_100"],
            "drop": reference["ar_100"] - candidate["ar_100"],
            "maximum": max_ar100_drop,
        },
    }
    for check in checks.values():
        check["passed"] = check["drop"] <= check["maximum"]
    passed = all(check["passed"] for check in checks.values())
    return {"status": "PASS" if passed else "FAIL", "passed": passed, "checks": checks}


def _runtime_smoke(engine_path: Path, calibration_path: Path, verbose: bool) -> Dict[str, object]:
    import numpy as np

    runner = TRTInference(engine_path, verbose=verbose)
    with np.load(calibration_path, allow_pickle=False) as archive:
        images = torch.from_numpy(np.ascontiguousarray(archive["images"][:1]))
    images = images.to(dtype=runner.input_torch_dtype("images"), device="cuda")
    outputs = runner({"images": images})
    torch.cuda.synchronize()
    tensors = {}
    passed = True
    for name, tensor in outputs.items():
        nonfinite = int((~torch.isfinite(tensor)).sum().item()) if tensor.is_floating_point() else 0
        passed = passed and nonfinite == 0
        tensors[name] = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "nonfinite": nonfinite,
        }
    if not passed:
        raise ValueError(f"TensorRT autotuned runtime smoke produced non-finite outputs: {tensors}")
    return {"passed": True, "batch_size": 1, "source": "first calibration image", "outputs": tensors}


def run(args: argparse.Namespace) -> Tuple[Path, Optional[Path]]:
    config = _resolved_file(args.config, "Model config")
    checkpoint = _resolved_file(args.checkpoint, "Checkpoint")
    dataset_root, data_file, val_path, data_cfg = resolve_yolo_data(args.data)
    assert dataset_root is not None and data_file is not None and val_path is not None and data_cfg is not None
    if not val_path.exists():
        raise FileNotFoundError(f"Validation image path does not exist: {val_path}")
    if args.opset < 19:
        raise ValueError("FP8 Q/DQ requires --opset 19 or newer.")
    if args.calibration_samples <= 0 or args.calibration_batch_size <= 0:
        raise ValueError("Calibration samples and batch size must be positive.")
    if args.calibration_samples % args.calibration_batch_size != 0:
        raise ValueError("--calibration-samples must be divisible by --calibration-batch-size.")
    if args.evaluate and args.onnx_only:
        raise ValueError("--evaluate requires TensorRT build; remove --onnx-only.")
    _validate_batch_request(False, args.min_batch, args.opt_batch, args.max_batch, args.eval_batch_size)
    selected_gpu, gpu_fingerprint = resolve_autotune_gpu(args.gpu, args.expected_gpu_regex)
    calibration_eps = resolve_calibration_eps(args.calibration_eps, selected_gpu)
    input_size = _parse_input_size(args.input_size)
    quality_reference_path = (
        _resolved_file(args.quality_reference, "Quality reference")
        if args.quality_reference
        else None
    )
    if quality_reference_path is not None and not args.evaluate:
        raise ValueError("--quality-reference requires --evaluate.")
    if args.autotune_trtexec_args and not args.autotune_use_trtexec:
        raise ValueError("--autotune-trtexec-args requires --autotune-use-trtexec.")
    autotune_settings = resolve_autotune_settings(args)
    modelopt_version = require_modelopt_autotune_version()
    autotune_pattern_cache = (
        _resolved_file(args.autotune_pattern_cache, "Autotune pattern cache")
        if args.autotune_pattern_cache
        else None
    )
    autotune_qdq_baseline = (
        _resolved_file(args.autotune_qdq_baseline, "Autotune Q/DQ baseline")
        if args.autotune_qdq_baseline
        else None
    )
    autotune_state_file = (
        Path(args.autotune_state_file).expanduser().resolve()
        if args.autotune_state_file
        else None
    )
    autotune_timing_cache = (
        Path(args.autotune_timing_cache).expanduser().resolve()
        if args.autotune_timing_cache
        else None
    )
    for pattern in args.exclude_node_regex:
        try:
            re.compile(pattern)
        except re.error as exc:
            raise ValueError(f"Invalid ModelOpt node regex {pattern!r}: {exc}") from exc

    probe_cfg = _build_cfg(config, dataset_root, data_file, args.calibration_batch_size, input_size)
    model_classes, image_hw, num_top_queries = _validate_model_contract(probe_cfg, data_cfg)
    del probe_cfg
    _cleanup()
    quality_reference = (
        validate_quality_reference_contract(
            quality_reference_path,
            checkpoint=checkpoint,
            image_hw=image_hw,
        )
        if quality_reference_path is not None
        else None
    )

    dry_summary = {
        "dataset": str(data_file.resolve()),
        "validation_images": str(val_path.resolve()),
        "config": str(config),
        "checkpoint": str(checkpoint),
        "num_classes": model_classes,
        "image_hw": list(image_hw),
        "num_top_queries": num_top_queries,
        "calibration_samples": args.calibration_samples,
        "calibration_batch_size": args.calibration_batch_size,
        "calibration_eps": calibration_eps,
        "modelopt_version": modelopt_version,
        "gpu_selector": args.gpu,
        "selected_gpu": selected_gpu,
        "autotune_hardware": gpu_fingerprint,
        "expected_gpu_regex": args.expected_gpu_regex,
        "source_precision_policy": DATAFLOW_FP16_POLICY,
        "high_precision_dtype": "fp16",
        "exclude_node_regex": list(args.exclude_node_regex),
        "exclude_op_types": list(args.exclude_op_type),
        "disable_mha_qdq": args.disable_mha_qdq,
        "require_fp8_qdq": args.require_fp8_qdq,
        "autotune": {
            "mode": args.autotune_mode,
            **autotune_settings,
            "pattern_cache": str(autotune_pattern_cache) if autotune_pattern_cache else None,
            "state_file": str(autotune_state_file) if autotune_state_file else None,
            "qdq_baseline": str(autotune_qdq_baseline) if autotune_qdq_baseline else None,
            "node_filter": list(args.autotune_node_filter),
            "use_trtexec": args.autotune_use_trtexec,
            "trtexec_args": args.autotune_trtexec_args,
            "timing_cache": str(autotune_timing_cache) if autotune_timing_cache else None,
        },
        "quality_reference": str(quality_reference_path) if quality_reference_path else None,
        "quality_reference_contract": quality_reference,
        "build_tensorrt": not args.onnx_only,
        "evaluate": args.evaluate,
    }
    if args.dry_run:
        print(json.dumps(dry_summary, indent=2, ensure_ascii=False))
        return Path(), None

    torch.cuda.set_device(selected_gpu)

    output_dir = _output_dir(args, config, checkpoint)
    status_path = output_dir / "status.json"
    status: Dict[str, object] = {
        "state": "running",
        "phase": "initialization",
        "started_at": datetime.now().astimezone().isoformat(),
    }

    def update(phase: str) -> None:
        status["phase"] = phase
        _write_json(status_path, status)

    source_onnx = output_dir / "source.fp16-dataflow.onnx"
    fp16_report_path = output_dir / "source.fp16-dataflow.report.json"
    calibration_path = output_dir / "calibration.npz"
    qdq_onnx = output_dir / "model.fp8.autotuned.qdq.onnx"
    qdq_report_path = output_dir / "model.fp8.autotuned.qdq.report.json"
    autotune_output_dir = output_dir / "autotune"
    effective_autotune_state = autotune_state_file or autotune_output_dir / "autotuner_state.yaml"
    engine_path = output_dir / "model.autotuned.engine"

    try:
        profile_sidecars = {
            "state": str(bind_autotune_cache_to_gpu(effective_autotune_state, gpu_fingerprint)),
            "timing_cache": (
                str(bind_autotune_cache_to_gpu(autotune_timing_cache, gpu_fingerprint))
                if autotune_timing_cache is not None
                else None
            ),
        }
        update("export_source_onnx")
        cfg = _build_cfg(config, dataset_root, data_file, args.calibration_batch_size, input_size)
        export_started = time.perf_counter()
        export_onnx(
            cfg=cfg,
            checkpoint=str(checkpoint),
            output_file=source_onnx,
            batch_size=args.opt_batch,
            opset=args.opset,
            static_batch=False,
            check=True,
            simplify=False,
            strict_load=True,
            export_mode="normalized",
            input_dtype="float16",
            onnx_exporter="legacy",
            onnx_precision_policy="baseline",
        )
        export_seconds = time.perf_counter() - export_started

        update("collect_real_calibration")
        write_modelopt_calibration_data(
            cfg,
            calibration_path,
            sample_count=args.calibration_samples,
            export_mode="normalized",
            input_dtype="float16",
        )
        del cfg
        _cleanup()

        update("apply_calibrated_fp16_dataflow")
        fp16_started = time.perf_counter()
        fp16_report = apply_dataflow_fp16_precision(
            source_onnx,
            calibration_path=calibration_path,
            report_path=fp16_report_path,
            calibration_batch_size=args.calibration_batch_size,
            fp32_node_patterns=DEFAULT_FP32_NODE_PATTERNS,
            providers=modelopt_gpu_first_providers(selected_gpu),
        )
        fp16_seconds = time.perf_counter() - fp16_started

        source_contract = _read_onnx_contract(source_onnx)
        _validate_onnx_contract(
            source_contract,
            "detection",
            "normalized",
            "float16",
            image_hw,
            False,
            args.opt_batch,
        )

        update("modelopt_fp8_autotune_qdq")
        quantize_started = time.perf_counter()
        fp16_exclusions = sorted(SENSITIVE_FP32_OP_TYPES)
        node_exclusions = [*DEFAULT_FP32_NODE_PATTERNS, *args.exclude_node_regex]
        qdq_report = apply_modelopt_fp8_qdq(
            source_onnx,
            qdq_onnx,
            calibration_path,
            calibration_shapes=(
                f"images:{args.calibration_batch_size}x3x{image_hw[0]}x{image_hw[1]}"
            ),
            calibration_method=args.calibration_method,
            calibration_eps=calibration_eps,
            high_precision_dtype="fp16",
            mha_accumulation_dtype=args.mha_accumulation_dtype,
            nodes_to_exclude=node_exclusions,
            # ModelOpt 0.46 Autotune returns a generic ORT op-type set. Limit it
            # to ModelOpt's own FP8-supported types while leaving the discovered
            # node set entirely under Autotune control.
            op_types_to_quantize=sorted(MODEL_OPT_FP8_SUPPORTED_OP_TYPES),
            op_types_to_exclude=args.exclude_op_type,
            op_types_to_exclude_fp16=fp16_exclusions,
            disable_mha_qdq=args.disable_mha_qdq,
            keep_intermediate_files=args.keep_modelopt_intermediates,
            autotune=True,
            autotune_output_dir=autotune_output_dir,
            autotune_num_schemes_per_region=autotune_settings["num_schemes_per_region"],
            autotune_pattern_cache_file=autotune_pattern_cache,
            autotune_state_file=effective_autotune_state,
            autotune_qdq_baseline=autotune_qdq_baseline,
            autotune_node_filter_list=args.autotune_node_filter,
            autotune_verbose=args.verbose_trt,
            autotune_use_trtexec=args.autotune_use_trtexec,
            autotune_timing_cache=autotune_timing_cache,
            autotune_warmup_runs=autotune_settings["warmup_runs"],
            autotune_timing_runs=autotune_settings["timing_runs"],
            autotune_trtexec_args=args.autotune_trtexec_args,
            require_fp8_qdq=args.require_fp8_qdq,
            log_level="DEBUG" if args.verbose_trt else "INFO",
            report_path=qdq_report_path,
        )
        quantize_seconds = time.perf_counter() - quantize_started

        qdq_contract = _read_onnx_contract(qdq_onnx)
        _validate_onnx_contract(
            qdq_contract,
            "detection",
            "normalized",
            "float16",
            image_hw,
            False,
            args.opt_batch,
        )
        if qdq_contract != source_contract:
            raise ValueError("Q/DQ ONNX external ABI differs from the source ONNX ABI.")

        engine_report = None
        metrics = None
        selected_precision = "fp8" if qdq_report["selected_fp8"] else "fp16"
        if not args.onnx_only:
            update("build_strongly_typed_tensorrt")
            build_started = time.perf_counter()
            built = build_engine(
                qdq_onnx,
                engine_path,
                precision=selected_precision,
                min_batch=args.min_batch,
                opt_batch=args.opt_batch,
                max_batch=args.max_batch,
                image_hw=image_hw,
                workspace_gb=args.workspace_gb,
                verbose=args.verbose_trt,
                profiling_verbosity="detailed",
                strongly_typed=True,
            )
            if built is None:
                raise RuntimeError(
                    f"TensorRT reports that selected precision {selected_precision!r} is unsupported."
                )
            build_seconds = time.perf_counter() - build_started
            manifest = _write_engine_manifest(engine_path, qdq_onnx, selected_precision)
            engine_contract = _read_engine_contract(engine_path, args.verbose_trt)
            _validate_engine_contract(
                engine_contract,
                qdq_contract,
                image_hw,
                False,
                args.min_batch,
                args.opt_batch,
                args.max_batch,
            )
            _validate_engine_top_queries(engine_contract, "normalized", num_top_queries)
            runtime_smoke = _runtime_smoke(engine_path, calibration_path, args.verbose_trt)
            engine_report = {
                "path": str(engine_path.resolve()),
                "bytes": engine_path.stat().st_size,
                "sha256": _sha256_file(engine_path),
                "manifest": str(manifest.resolve()),
                "selected_precision": selected_precision,
                "build_seconds": build_seconds,
                "runtime_smoke": runtime_smoke,
            }

            if args.evaluate:
                update("evaluate_tensorrt")
                eval_cfg = _build_cfg(config, dataset_root, data_file, args.eval_batch_size, input_size)
                stats = evaluate_engine(
                    engine_path,
                    eval_cfg,
                    args.score_threshold,
                    args.verbose_trt,
                    args.eval_limit,
                    "normalized",
                )
                metrics = _coco_stats(stats)
                del eval_cfg
                _cleanup()

        if metrics is None:
            quality_gate = {"status": "NOT_EVALUATED", "passed": None}
            result = "COMPLETED"
        elif quality_reference_path is None:
            quality_gate = {
                "status": "UNASSESSED",
                "passed": None,
                "reason": "No --quality-reference was supplied.",
            }
            result = "COMPLETED_UNASSESSED"
        else:
            quality_gate = assess_quality(
                quality_reference["coco_bbox"],
                metrics,
                max_map_drop=args.max_map_drop,
                max_ar100_drop=args.max_ar100_drop,
            )
            quality_gate["reference"] = str(quality_reference_path)
            result = "QUALITY_PASS" if quality_gate["passed"] else "QUALITY_FAIL"

        if not qdq_report["selected_fp8"] and result != "QUALITY_FAIL":
            result = "NO_FP8_BENEFIT"

        report = {
            "schema_version": 1,
            "result": result,
            "inputs": {
                **dry_summary,
                "config_sha256": _sha256_file(config),
                "checkpoint_sha256": _sha256_file(checkpoint),
            },
            "source_onnx": {
                "path": str(source_onnx.resolve()),
                "bytes": source_onnx.stat().st_size,
                "sha256": _sha256_file(source_onnx),
                "export_seconds": export_seconds,
                "fp16_dataflow_seconds": fp16_seconds,
                "fp16_dataflow_report": fp16_report,
            },
            "modelopt_autotune": {
                **qdq_report,
                "quantize_seconds": quantize_seconds,
                "hardware": gpu_fingerprint,
                "hardware_profile_sidecars": profile_sidecars,
            },
            "tensorrt": engine_report,
            "coco_bbox": metrics,
            "quality_gate": quality_gate,
        }
        report_path = output_dir / "export_report.json"
        _write_json(report_path, report)
        status.update(
            {
                "state": "completed",
                "phase": "done",
                "finished_at": datetime.now().astimezone().isoformat(),
                "report": str(report_path.resolve()),
            }
        )
        _write_json(status_path, status)
        return report_path, engine_path if engine_report is not None else None
    except BaseException as exc:
        status.update(
            {
                "state": "error",
                "finished_at": datetime.now().astimezone().isoformat(),
                "error": {"type": type(exc).__name__, "message": str(exc)},
                "traceback": traceback.format_exc(),
            }
        )
        _write_json(status_path, status)
        raise


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = get_parser().parse_args(argv)
    report, engine = run(args)
    if args.dry_run:
        print("Dry run passed; no artifacts were created.")
        return 0
    print(f"Export report: {report}")
    print(f"TensorRT engine: {engine if engine is not None else 'not requested (--onnx-only)'}")
    payload = json.loads(report.read_text(encoding="utf-8"))
    return 2 if payload.get("result") == "QUALITY_FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
