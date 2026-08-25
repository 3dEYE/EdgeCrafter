"""One-command EdgeCrafter checkpoint -> calibrated FP16 dataflow ONNX -> TensorRT."""

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
)
from tools.deployment.onnx_dataflow_fp16 import (
    DATAFLOW_FP16_POLICY,
    DEFAULT_DATA_MAX,
    DEFAULT_INIT_MAX,
)
from tools.deployment.onnx_precision import read_precision_policy


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
            "Export an explicitly selected EdgeCrafter checkpoint, calibrate NVIDIA ModelOpt "
            "AutoCast on real validation images, write explicit FP16 Cast dataflow, validate "
            "the ONNX ABI, and build a strongly typed TensorRT FP16 engine."
        )
    )
    required = parser.add_argument_group("required inputs")
    required.add_argument("--data", required=True, help="YOLO dataset directory or data.yaml path.")
    required.add_argument("--config", required=True, help="Model YAML matching the checkpoint.")
    required.add_argument("--checkpoint", required=True, help="Checkpoint loaded with strict validation.")

    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--input-size", nargs="+", type=int, default=None)
    parser.add_argument("--opset", type=int, default=20)
    parser.add_argument("--calibration-samples", type=int, default=1)
    parser.add_argument(
        "--calibration-batch-size",
        type=int,
        default=1,
        help="Maximum ModelOpt/ORT calibration batch held in host RAM (default: 1).",
    )
    parser.add_argument("--fp16-data-max", type=float, default=DEFAULT_DATA_MAX)
    parser.add_argument("--fp16-init-max", type=float, default=DEFAULT_INIT_MAX)
    parser.add_argument("--min-batch", type=int, default=1)
    parser.add_argument("--opt-batch", type=int, default=4)
    parser.add_argument("--max-batch", type=int, default=8)
    parser.add_argument("--workspace-gb", type=float, default=8.0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--onnx-only", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--verbose-trt", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _resolved_file(value: str, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} is not a file: {path}")
    return path


def _safe_stem(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "model"


def _create_output_dir(requested: Optional[str], config: Path, checkpoint: Path) -> Path:
    if requested:
        output_dir = Path(requested).expanduser().resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{stamp}_{_safe_stem(config.stem)}_{_safe_stem(checkpoint.stem)}"
        output_dir = SCRIPT_ROOT / "outputs" / "fp16_dataflow_exports" / name
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


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
    task = cfg.yaml_cfg.get("task")
    if task != "detection":
        raise ValueError(f"FP16 dataflow exporter currently supports detection only, got {task!r}.")
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


def _coco_stats(stats: Sequence[float]) -> Dict[str, float]:
    if len(stats) < len(COCO_BBOX_NAMES):
        raise ValueError("TensorRT evaluation did not return COCO bbox metrics.")
    return {name: float(stats[index]) for index, name in enumerate(COCO_BBOX_NAMES)}


def _runtime_smoke(engine_path: Path, calibration_path: Path, verbose: bool) -> Dict[str, object]:
    import numpy as np

    runner = TRTInference(engine_path, verbose=verbose)
    with np.load(calibration_path, allow_pickle=False) as archive:
        images = torch.from_numpy(np.ascontiguousarray(archive["images"][:1]))
    images = images.to(dtype=runner.input_torch_dtype("images"), device="cuda")
    outputs = runner({"images": images})
    torch.cuda.synchronize()
    tensors: Dict[str, object] = {}
    for name, tensor in outputs.items():
        nonfinite = int((~torch.isfinite(tensor)).sum().item()) if tensor.is_floating_point() else 0
        if nonfinite:
            raise ValueError(
                f"TensorRT FP16 dataflow smoke produced {nonfinite} non-finite values in {name}."
            )
        tensors[name] = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "nonfinite": nonfinite,
        }
    return {"passed": True, "batch_size": 1, "source": "first calibration image", "outputs": tensors}


def run(args: argparse.Namespace) -> Tuple[Path, Optional[Path]]:
    config = _resolved_file(args.config, "Model config")
    checkpoint = _resolved_file(args.checkpoint, "Checkpoint")
    dataset_root, data_file, val_path, data_cfg = resolve_yolo_data(args.data)
    assert dataset_root is not None and data_file is not None and val_path is not None and data_cfg is not None
    if not val_path.exists():
        raise FileNotFoundError(f"Validation image path does not exist: {val_path}")
    if args.calibration_samples <= 0 or args.calibration_batch_size <= 0:
        raise ValueError("Calibration samples and batch size must be positive.")
    if args.calibration_samples % args.calibration_batch_size != 0:
        raise ValueError("--calibration-samples must be divisible by --calibration-batch-size.")
    if args.fp16_data_max <= 0.0 or args.fp16_init_max <= 0.0:
        raise ValueError("FP16 data and initializer limits must be positive.")
    if args.evaluate and args.onnx_only:
        raise ValueError("--evaluate requires TensorRT build; remove --onnx-only.")
    _validate_batch_request(
        False,
        args.min_batch,
        args.opt_batch,
        args.max_batch,
        args.eval_batch_size,
    )
    input_size = _parse_input_size(args.input_size)

    probe_cfg = _build_cfg(config, dataset_root, data_file, args.eval_batch_size, input_size)
    model_classes, image_hw, num_top_queries = _validate_model_contract(probe_cfg, data_cfg)
    del probe_cfg
    _cleanup()

    summary = {
        "dataset": str(data_file.resolve()),
        "validation_images": str(val_path.resolve()),
        "config": str(config),
        "checkpoint": str(checkpoint),
        "precision_policy": DATAFLOW_FP16_POLICY,
        "num_classes": model_classes,
        "image_hw": list(image_hw),
        "num_top_queries": num_top_queries,
        "calibration_samples": args.calibration_samples,
        "calibration_batch_size": args.calibration_batch_size,
        "profile_batches": [args.min_batch, args.opt_batch, args.max_batch],
        "build_tensorrt": not args.onnx_only,
        "evaluate": args.evaluate,
    }
    if args.dry_run:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return Path(), None

    if not args.onnx_only:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required to build TensorRT; use --onnx-only to stop at ONNX.")
        if args.gpu < 0 or args.gpu >= torch.cuda.device_count():
            raise ValueError(f"--gpu {args.gpu} is outside {torch.cuda.device_count()} visible GPU(s).")
        torch.cuda.set_device(args.gpu)

    output_dir = _create_output_dir(args.output_dir, config, checkpoint)
    status_path = output_dir / "status.json"
    status: Dict[str, object] = {
        "state": "running",
        "phase": "initialization",
        "started_at": datetime.now().astimezone().isoformat(),
        "output_dir": str(output_dir),
    }

    def update(phase: str) -> None:
        status["phase"] = phase
        _write_json(status_path, status)

    onnx_path = output_dir / "model.fp16.dataflow.onnx"
    calibration_path = onnx_path.with_suffix(".fp16-calibration.npz")
    precision_report_path = output_dir / "precision_report.json"
    engine_path = output_dir / "model.fp16.dataflow.engine"

    try:
        update("export_modelopt_fp16_dataflow_onnx")
        cfg = _build_cfg(config, dataset_root, data_file, args.eval_batch_size, input_size)
        export_started = time.perf_counter()
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
            onnx_precision_policy="explicit-fp16-dataflow",
            fp16_report=precision_report_path,
            fp16_calibration_samples=args.calibration_samples,
            fp16_calibration_batch_size=args.calibration_batch_size,
            fp16_data_max=args.fp16_data_max,
            fp16_init_max=args.fp16_init_max,
        )
        export_seconds = time.perf_counter() - export_started
        del cfg
        _cleanup()

        if read_precision_policy(onnx_path) != DATAFLOW_FP16_POLICY:
            raise ValueError("Exported ONNX does not contain the required FP16 dataflow policy metadata.")
        onnx_contract = _read_onnx_contract(onnx_path)
        _validate_onnx_contract(
            onnx_contract,
            "detection",
            "normalized",
            "float16",
            image_hw,
            False,
            args.opt_batch,
        )

        engine_report = None
        metrics = None
        if not args.onnx_only:
            update("build_strongly_typed_tensorrt_fp16")
            build_started = time.perf_counter()
            built = build_engine(
                onnx_path,
                engine_path,
                precision="fp16",
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
                raise RuntimeError("TensorRT reports that FP16 is unsupported on this platform.")
            build_seconds = time.perf_counter() - build_started
            manifest = _write_engine_manifest(engine_path, onnx_path, "fp16")
            engine_contract = _read_engine_contract(engine_path, args.verbose_trt)
            _validate_engine_contract(
                engine_contract,
                onnx_contract,
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
                "build_seconds": build_seconds,
                "network_typing": "strongly_typed",
                "runtime_smoke": runtime_smoke,
            }

            if args.evaluate:
                update("evaluate_tensorrt")
                eval_cfg = _build_cfg(config, dataset_root, data_file, args.eval_batch_size, input_size)
                metrics = _coco_stats(
                    evaluate_engine(
                        engine_path,
                        eval_cfg,
                        args.score_threshold,
                        args.verbose_trt,
                        args.eval_limit,
                        "normalized",
                    )
                )
                del eval_cfg
                _cleanup()

        report = {
            "schema_version": 1,
            "result": "PASS",
            "inputs": {
                **summary,
                "config_sha256": _sha256_file(config),
                "checkpoint_sha256": _sha256_file(checkpoint),
            },
            "onnx": {
                "path": str(onnx_path.resolve()),
                "bytes": onnx_path.stat().st_size,
                "sha256": _sha256_file(onnx_path),
                "export_seconds": export_seconds,
                "calibration": str(calibration_path.resolve()),
                "precision_report": json.loads(precision_report_path.read_text(encoding="utf-8")),
            },
            "tensorrt": engine_report,
            "coco_bbox": metrics,
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
