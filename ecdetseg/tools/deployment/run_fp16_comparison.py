"""Run a reproducible baseline-vs-explicit-FP16 ONNX/TensorRT comparison.

The dataset, model configuration, and checkpoint are all explicit inputs.  The
script intentionally does not infer a configuration from a checkpoint name:
that would make a successful but semantically wrong export too easy.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
import re
import statistics
import sys
import time
import traceback
from collections import Counter
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
    dump_engine_inspector,
    evaluate_engine,
    export_onnx,
    resolve_yolo_data,
)
from tools.deployment.onnx_dataflow_fp16 import (
    DATAFLOW_FP16_POLICY,
    DEFAULT_DATA_MAX,
    DEFAULT_INIT_MAX,
)


COCO_BBOX_NAMES = (
    "map_50_95",
    "map_50",
    "map_75",
    "map_small",
    "map_medium",
    "map_large",
    "ar_1",
    "ar_10",
    "ar_100",
    "ar_small",
    "ar_medium",
    "ar_large",
)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export one explicitly selected EdgeCrafter model twice, build TensorRT engines, "
            "evaluate both on a YOLO validation dataset, and apply parity/performance gates."
        )
    )
    required = parser.add_argument_group("required model and data inputs")
    required.add_argument("--data", required=True, help="YOLO dataset directory or data.yaml path.")
    required.add_argument("--config", required=True, help="Model YAML configuration used by the checkpoint.")
    required.add_argument("--checkpoint", required=True, help="Checkpoint to export; loaded with strict validation.")

    parser.add_argument(
        "--output-dir",
        default=None,
        help="New result directory. By default a timestamped directory is created under outputs/trt_fp16_auto.",
    )
    parser.add_argument(
        "--input-size",
        nargs="+",
        type=int,
        default=None,
        metavar="SIZE",
        help="Optional square size override. Without it, eval_spatial_size is read from --config.",
    )
    parser.add_argument("--min-batch", type=int, default=1)
    parser.add_argument("--opt-batch", type=int, default=4)
    parser.add_argument("--max-batch", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--benchmark-batch-size", type=int, default=None)
    parser.add_argument("--workspace-gb", type=float, default=8.0)
    parser.add_argument("--opset", type=int, default=20)
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--eval-limit", type=int, default=None, help="Debug only: limit validation images.")
    parser.add_argument("--calibration-samples", type=int, default=1)
    parser.add_argument("--fp16-data-max", type=float, default=DEFAULT_DATA_MAX)
    parser.add_argument("--fp16-init-max", type=float, default=DEFAULT_INIT_MAX)
    parser.add_argument("--benchmark-warmup", type=int, default=10)
    parser.add_argument("--benchmark-iterations", type=int, default=40)
    parser.add_argument("--benchmark-rounds", type=int, default=5)
    parser.add_argument("--max-map-drop", type=float, default=0.001)
    parser.add_argument("--max-ar100-drop", type=float, default=0.001)
    parser.add_argument(
        "--max-latency-regression",
        type=float,
        default=0.05,
        help="Allowed fractional regression, e.g. 0.05 means 5%%.",
    )
    parser.add_argument(
        "--max-engine-size-regression",
        type=float,
        default=0.10,
        help="Allowed fractional regression, e.g. 0.10 means 10%%.",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--verbose-trt", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths, dataset/config class count, resolution, and batch settings without exporting.",
    )
    return parser


def _resolved_file(value: str, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} is not a file: {path}")
    return path


def _safe_stem(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned or "model"


def _default_output_dir(config: Path, checkpoint: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{stamp}_{_safe_stem(config.stem)}_{_safe_stem(checkpoint.stem)}"
    return SCRIPT_ROOT / "outputs" / "trt_fp16_auto" / name


def _create_output_dir(requested: Optional[str], config: Path, checkpoint: Path) -> Path:
    output_dir = Path(requested).expanduser().resolve() if requested else _default_output_dir(config, checkpoint)
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _dataset_class_count(data_cfg: dict) -> Optional[int]:
    if data_cfg.get("nc") is not None:
        return int(data_cfg["nc"])
    names = data_cfg.get("names")
    if isinstance(names, (list, tuple, dict)):
        return len(names)
    return None


def _build_cfg(
    config: Path,
    dataset_root: Path,
    data_file: Path,
    eval_batch_size: int,
    input_size: Optional[List[int]],
) -> YAMLConfig:
    update: Dict[str, object] = {
        "yolo_root": str(dataset_root),
        "yolo_data_file": str(data_file),
        "val_dataloader": {"total_batch_size": int(eval_batch_size)},
    }
    if input_size is not None:
        update = merge_dict(update, {"eval_spatial_size": input_size})
    return YAMLConfig(str(config), **update)


def _validate_model_data_contract(cfg: YAMLConfig, data_cfg: dict) -> Tuple[int, Tuple[int, int], int]:
    task = cfg.yaml_cfg.get("task")
    if task != "detection":
        raise ValueError(
            f"The automatic comparison currently supports detection configs only, got task={task!r}."
        )

    model_classes = int(cfg.yaml_cfg["num_classes"])
    dataset_classes = _dataset_class_count(data_cfg)
    if dataset_classes is None:
        raise ValueError("Dataset YAML must define either 'nc' or 'names' so class compatibility can be checked.")
    if dataset_classes != model_classes:
        raise ValueError(
            f"Dataset/model class mismatch: dataset has {dataset_classes}, config has {model_classes}. "
            "Select the config that was used to train this checkpoint."
        )

    input_size = _parse_input_size(list(cfg.yaml_cfg["eval_spatial_size"]))
    if input_size is None:
        raise ValueError("Model config has no eval_spatial_size.")
    image_hw = (int(input_size[0]), int(input_size[1]))
    return model_classes, image_hw, _configured_num_top_queries(cfg)


def coco_bbox_dict(stats: Sequence[float]) -> Dict[str, float]:
    if len(stats) < len(COCO_BBOX_NAMES):
        raise ValueError(f"Expected {len(COCO_BBOX_NAMES)} COCO bbox values, got {len(stats)}.")
    return {name: float(stats[index]) for index, name in enumerate(COCO_BBOX_NAMES)}


def summarize_inspector(path: Path) -> Dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    layers = payload.get("Layers", []) if isinstance(payload, dict) else []
    formats: Counter[str] = Counter()

    def visit(value: object) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "Format/Datatype" and isinstance(child, str):
                    normalized = child.lower()
                    if "half" in normalized or "fp16" in normalized:
                        formats["float16"] += 1
                    elif "float" in normalized or "fp32" in normalized:
                        formats["float32"] += 1
                    elif "int64" in normalized:
                        formats["int64"] += 1
                    elif "int32" in normalized:
                        formats["int32"] += 1
                    else:
                        formats[child] += 1
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(payload)
    return {"layers": len(layers), "tensor_edges": dict(sorted(formats.items()))}


def assess_gates(
    baseline_stats: Sequence[float],
    candidate_stats: Sequence[float],
    baseline_latency_ms: float,
    candidate_latency_ms: float,
    baseline_engine_bytes: int,
    candidate_engine_bytes: int,
    *,
    max_map_drop: float,
    max_ar100_drop: float,
    max_latency_regression: float,
    max_engine_size_regression: float,
) -> Dict[str, object]:
    baseline = coco_bbox_dict(baseline_stats)
    candidate = coco_bbox_dict(candidate_stats)
    if baseline_latency_ms <= 0 or baseline_engine_bytes <= 0:
        raise ValueError("Baseline latency and engine size must be positive.")

    values = {
        "map_drop": baseline["map_50_95"] - candidate["map_50_95"],
        "ar100_drop": baseline["ar_100"] - candidate["ar_100"],
        "latency_regression": candidate_latency_ms / baseline_latency_ms - 1.0,
        "engine_size_regression": candidate_engine_bytes / baseline_engine_bytes - 1.0,
    }
    limits = {
        "map_drop": max_map_drop,
        "ar100_drop": max_ar100_drop,
        "latency_regression": max_latency_regression,
        "engine_size_regression": max_engine_size_regression,
    }
    checks = {
        name: {"value": float(values[name]), "maximum": float(limit), "passed": values[name] <= limit}
        for name, limit in limits.items()
    }
    return {"passed": all(check["passed"] for check in checks.values()), "checks": checks}


def _benchmark_engine(runner: TRTInference, images: torch.Tensor, iterations: int) -> float:
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(iterations):
        runner({"images": images})
    torch.cuda.synchronize()
    return (time.perf_counter() - started) * 1000.0 / iterations


def benchmark_pair(
    baseline_engine: Path,
    candidate_engine: Path,
    image_hw: Tuple[int, int],
    batch_size: int,
    warmup: int,
    iterations: int,
    rounds: int,
    verbose: bool,
) -> Dict[str, object]:
    if warmup < 0 or iterations <= 0 or rounds <= 0:
        raise ValueError("Benchmark requires warmup >= 0, iterations > 0, and rounds > 0.")
    baseline = TRTInference(baseline_engine, verbose=verbose)
    candidate = TRTInference(candidate_engine, verbose=verbose)
    baseline_images = torch.rand(
        batch_size,
        3,
        image_hw[0],
        image_hw[1],
        dtype=baseline.input_torch_dtype("images"),
        device="cuda",
    )
    candidate_images = baseline_images.to(dtype=candidate.input_torch_dtype("images"))

    for _ in range(warmup):
        baseline({"images": baseline_images})
        candidate({"images": candidate_images})
    torch.cuda.synchronize()

    timings: Dict[str, List[float]] = {"baseline": [], "explicit_fp16_dataflow": []}
    for round_index in range(rounds):
        order = (
            (("baseline", baseline, baseline_images), ("explicit_fp16_dataflow", candidate, candidate_images))
            if round_index % 2 == 0
            else (("explicit_fp16_dataflow", candidate, candidate_images), ("baseline", baseline, baseline_images))
        )
        for name, runner, images in order:
            timings[name].append(_benchmark_engine(runner, images, iterations))

    result: Dict[str, object] = {
        "method": "same process, alternating order, CUDA synchronize",
        "batch_size": batch_size,
        "warmup": warmup,
        "iterations_per_round": iterations,
        "rounds": rounds,
    }
    for name, values in timings.items():
        median_ms = statistics.median(values)
        result[name] = {
            "round_ms": values,
            "median_ms": median_ms,
            "samples_per_second": batch_size * 1000.0 / median_ms,
        }
    return result


def _package_version(distribution: str) -> Optional[str]:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _export_variant(
    cfg: YAMLConfig,
    checkpoint: Path,
    onnx_path: Path,
    policy: str,
    args: argparse.Namespace,
    report_path: Optional[Path] = None,
) -> float:
    started = time.perf_counter()
    export_onnx(
        cfg=cfg,
        checkpoint=str(checkpoint),
        output_file=onnx_path,
        batch_size=args.opt_batch,
        opset=args.opset,
        static_batch=False,
        check=True,
        simplify=False,
        strict_load=True,
        export_mode="normalized",
        input_dtype="float16",
        onnx_exporter="legacy",
        onnx_precision_policy=policy,
        fp16_report=report_path,
        fp16_calibration_samples=args.calibration_samples,
        fp16_data_max=args.fp16_data_max,
        fp16_init_max=args.fp16_init_max,
    )
    return time.perf_counter() - started


def _build_variant(
    onnx_path: Path,
    engine_path: Path,
    image_hw: Tuple[int, int],
    args: argparse.Namespace,
    *,
    strongly_typed: bool,
) -> float:
    started = time.perf_counter()
    built = build_engine(
        onnx_file=onnx_path,
        output_file=engine_path,
        precision="fp16",
        min_batch=args.min_batch,
        opt_batch=args.opt_batch,
        max_batch=args.max_batch,
        image_hw=image_hw,
        workspace_gb=args.workspace_gb,
        verbose=args.verbose_trt,
        profiling_verbosity="detailed",
        strongly_typed=strongly_typed,
    )
    if built is None:
        raise RuntimeError("TensorRT reports that FP16 is unsupported on the selected platform.")
    return time.perf_counter() - started


def run(args: argparse.Namespace) -> Tuple[Path, bool]:
    config = _resolved_file(args.config, "Model config")
    checkpoint = _resolved_file(args.checkpoint, "Checkpoint")
    dataset_root, data_file, val_path, data_cfg = resolve_yolo_data(args.data)
    assert dataset_root is not None and data_file is not None and val_path is not None and data_cfg is not None
    if not val_path.exists():
        raise FileNotFoundError(f"Validation image path does not exist: {val_path}")

    input_size_override = _parse_input_size(args.input_size)
    benchmark_batch = args.benchmark_batch_size or args.opt_batch
    _validate_batch_request(
        False,
        args.min_batch,
        args.opt_batch,
        args.max_batch,
        args.eval_batch_size,
    )
    if not args.min_batch <= benchmark_batch <= args.max_batch:
        raise ValueError(
            f"Benchmark batch {benchmark_batch} is outside TensorRT profile "
            f"{args.min_batch}/{args.opt_batch}/{args.max_batch}."
        )

    probe_cfg = _build_cfg(config, dataset_root, data_file, args.eval_batch_size, input_size_override)
    model_classes, image_hw, num_top_queries = _validate_model_data_contract(probe_cfg, data_cfg)
    del probe_cfg
    _cleanup_cuda()

    if args.dry_run:
        summary = {
            "data": str(data_file),
            "validation_images": str(val_path),
            "config": str(config),
            "checkpoint": str(checkpoint),
            "model_classes": model_classes,
            "image_hw": list(image_hw),
            "num_top_queries": num_top_queries,
            "profile_batches": [args.min_batch, args.opt_batch, args.max_batch],
            "eval_batch_size": args.eval_batch_size,
            "benchmark_batch_size": benchmark_batch,
        }
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return Path(), True

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for TensorRT build and evaluation.")
    if args.gpu < 0 or args.gpu >= torch.cuda.device_count():
        raise ValueError(f"--gpu {args.gpu} is out of range for {torch.cuda.device_count()} visible GPU(s).")
    torch.cuda.set_device(args.gpu)

    output_dir = _create_output_dir(args.output_dir, config, checkpoint)
    status_path = output_dir / "status.json"
    status: Dict[str, object] = {
        "state": "running",
        "phase": "initialization",
        "started_at": datetime.now().astimezone().isoformat(),
        "output_dir": str(output_dir),
    }

    def update_status(phase: str, **extra: object) -> None:
        status["phase"] = phase
        status.update(extra)
        _write_json(status_path, status)

    baseline_dir = output_dir / "baseline"
    candidate_dir = output_dir / "explicit_fp16_dataflow"
    baseline_onnx = baseline_dir / "model.onnx"
    candidate_onnx = candidate_dir / "model.onnx"
    baseline_engine = baseline_dir / "model.fp16.engine"
    candidate_engine = candidate_dir / "model.fp16.engine"
    fp16_report_path = candidate_dir / "precision_report.json"

    try:
        update_status("export_baseline")
        cfg = _build_cfg(config, dataset_root, data_file, args.eval_batch_size, input_size_override)
        baseline_export_seconds = _export_variant(cfg, checkpoint, baseline_onnx, "baseline", args)
        del cfg
        _cleanup_cuda()

        update_status("export_explicit_fp16_dataflow")
        cfg = _build_cfg(config, dataset_root, data_file, args.eval_batch_size, input_size_override)
        candidate_export_seconds = _export_variant(
            cfg,
            checkpoint,
            candidate_onnx,
            "explicit-fp16-dataflow",
            args,
            fp16_report_path,
        )
        del cfg
        _cleanup_cuda()

        update_status("validate_onnx")
        onnx_contracts = {}
        for name, path in (("baseline", baseline_onnx), ("explicit_fp16_dataflow", candidate_onnx)):
            contract = _read_onnx_contract(path)
            _validate_onnx_contract(
                contract,
                task="detection",
                export_mode="normalized",
                input_dtype="float16",
                image_hw=image_hw,
                static_batch=False,
                opt_batch=args.opt_batch,
            )
            onnx_contracts[name] = contract

        update_status("build_baseline_engine")
        baseline_build_seconds = _build_variant(
            baseline_onnx, baseline_engine, image_hw, args, strongly_typed=False
        )
        _write_engine_manifest(baseline_engine, baseline_onnx, "fp16")
        _cleanup_cuda()

        update_status("build_explicit_fp16_dataflow_engine")
        candidate_build_seconds = _build_variant(
            candidate_onnx, candidate_engine, image_hw, args, strongly_typed=True
        )
        _write_engine_manifest(candidate_engine, candidate_onnx, "fp16")
        _cleanup_cuda()

        update_status("validate_engines")
        engine_contracts = {}
        for name, engine_path in (("baseline", baseline_engine), ("explicit_fp16_dataflow", candidate_engine)):
            engine_contract = _read_engine_contract(engine_path, args.verbose_trt)
            _validate_engine_contract(
                engine_contract,
                onnx_contracts[name],
                image_hw,
                False,
                args.min_batch,
                args.opt_batch,
                args.max_batch,
            )
            _validate_engine_top_queries(engine_contract, "normalized", num_top_queries)
            engine_contracts[name] = engine_contract

        baseline_inspector_path = dump_engine_inspector(
            baseline_engine, baseline_dir / "inspector.json", args.verbose_trt
        )
        candidate_inspector_path = dump_engine_inspector(
            candidate_engine, candidate_dir / "inspector.json", args.verbose_trt
        )

        update_status("benchmark")
        raw_benchmark = benchmark_pair(
            baseline_engine,
            candidate_engine,
            image_hw,
            benchmark_batch,
            args.benchmark_warmup,
            args.benchmark_iterations,
            args.benchmark_rounds,
            args.verbose_trt,
        )
        _cleanup_cuda()

        update_status("evaluate_baseline")
        baseline_eval_cfg = _build_cfg(config, dataset_root, data_file, args.eval_batch_size, input_size_override)
        validation_images = len(baseline_eval_cfg.val_dataloader.dataset)
        baseline_stats = evaluate_engine(
            baseline_engine,
            baseline_eval_cfg,
            args.score_threshold,
            args.verbose_trt,
            args.eval_limit,
            "normalized",
        )
        del baseline_eval_cfg
        _cleanup_cuda()

        update_status("evaluate_explicit_fp16_dataflow")
        candidate_eval_cfg = _build_cfg(config, dataset_root, data_file, args.eval_batch_size, input_size_override)
        candidate_stats = evaluate_engine(
            candidate_engine,
            candidate_eval_cfg,
            args.score_threshold,
            args.verbose_trt,
            args.eval_limit,
            "normalized",
        )
        del candidate_eval_cfg
        _cleanup_cuda()

        baseline_latency = float(raw_benchmark["baseline"]["median_ms"])
        candidate_latency = float(raw_benchmark["explicit_fp16_dataflow"]["median_ms"])
        gates = assess_gates(
            baseline_stats,
            candidate_stats,
            baseline_latency,
            candidate_latency,
            baseline_engine.stat().st_size,
            candidate_engine.stat().st_size,
            max_map_drop=args.max_map_drop,
            max_ar100_drop=args.max_ar100_drop,
            max_latency_regression=args.max_latency_regression,
            max_engine_size_regression=args.max_engine_size_regression,
        )

        fp16_policy_report = json.loads(fp16_report_path.read_text(encoding="utf-8"))
        report = {
            "schema_version": 2,
            "verdict": "PASS" if gates["passed"] else "FAIL",
            "inputs": {
                "dataset_root": str(dataset_root.resolve()),
                "data_file": str(data_file.resolve()),
                "validation_images": validation_images,
                "config": str(config),
                "config_sha256": _sha256_file(config),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": _sha256_file(checkpoint),
            },
            "model_contract": {
                "task": "detection",
                "num_classes": model_classes,
                "input_dtype": "float16",
                "image_hw": list(image_hw),
                "num_top_queries": num_top_queries,
                "export_mode": "normalized",
                "checkpoint_load": "strict",
            },
            "settings": {
                "profile_batches": [args.min_batch, args.opt_batch, args.max_batch],
                "eval_batch_size": args.eval_batch_size,
                "score_threshold": args.score_threshold,
                "eval_limit": args.eval_limit,
                "workspace_gb": args.workspace_gb,
                "opset": args.opset,
                "calibration_samples": args.calibration_samples,
                "fp16_data_max": args.fp16_data_max,
                "fp16_init_max": args.fp16_init_max,
            },
            "environment": {
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "onnx": _package_version("onnx"),
                "tensorrt": _package_version("tensorrt"),
                "nvidia_modelopt": _package_version("nvidia-modelopt"),
                "gpu": torch.cuda.get_device_name(args.gpu),
            },
            "baseline": {
                "policy": "weakly_typed_trt_fp16",
                "onnx": {
                    "path": str(baseline_onnx),
                    "bytes": baseline_onnx.stat().st_size,
                    "sha256": _sha256_file(baseline_onnx),
                    "export_seconds": baseline_export_seconds,
                },
                "engine": {
                    "path": str(baseline_engine),
                    "bytes": baseline_engine.stat().st_size,
                    "sha256": _sha256_file(baseline_engine),
                    "build_seconds": baseline_build_seconds,
                    **summarize_inspector(baseline_inspector_path),
                },
                "coco_bbox": coco_bbox_dict(baseline_stats),
                "raw_benchmark": raw_benchmark["baseline"],
            },
            "explicit_fp16_dataflow": {
                "policy": DATAFLOW_FP16_POLICY,
                "network_typing": "strongly_typed",
                "onnx": {
                    "path": str(candidate_onnx),
                    "bytes": candidate_onnx.stat().st_size,
                    "sha256": _sha256_file(candidate_onnx),
                    "export_seconds": candidate_export_seconds,
                },
                "engine": {
                    "path": str(candidate_engine),
                    "bytes": candidate_engine.stat().st_size,
                    "sha256": _sha256_file(candidate_engine),
                    "build_seconds": candidate_build_seconds,
                    **summarize_inspector(candidate_inspector_path),
                },
                "precision_report": fp16_policy_report,
                "coco_bbox": coco_bbox_dict(candidate_stats),
                "raw_benchmark": raw_benchmark["explicit_fp16_dataflow"],
            },
            "raw_benchmark_method": {
                key: value
                for key, value in raw_benchmark.items()
                if key not in ("baseline", "explicit_fp16_dataflow")
            },
            "gates": gates,
        }
        report_path = output_dir / "comparison.json"
        _write_json(report_path, report)
        status.update(
            {
                "state": "completed" if gates["passed"] else "gate_failed",
                "phase": "done",
                "finished_at": datetime.now().astimezone().isoformat(),
                "verdict": report["verdict"],
                "report": str(report_path),
            }
        )
        _write_json(status_path, status)
        return report_path, bool(gates["passed"])
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
    report_path, passed = run(args)
    if args.dry_run:
        print("Dry run passed; no ONNX or TensorRT artifacts were created.")
        return 0
    print(f"Comparison report: {report_path}")
    print(f"Verdict: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
