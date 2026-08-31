"""Select an exact, diverse YOLO validation subset for FP16 calibration.

The selector is intentionally separate from ``balance_yolo_dataset.py``:
training balancing unions independent per-class budgets and therefore does not
guarantee an exact global sample count.  Calibration needs an exact-size,
representative coreset and must preserve the original class numbering.

Selection combines:

* DINOv3 ConvNeXt embeddings for visual farthest-first (k-Center) coverage;
* SigLIP2 embeddings for independent scene/view clusters;
* class, brightness, object-scale, and CCTV-score coverage constraints;
* a DINO cosine near-duplicate guard.

The command is a read-only dry run unless ``--apply`` is passed.  Apply mode
copies image/label pairs into a fresh portable YOLO dataset and can write the
FP16 calibration NPZ through EdgeCrafter's real validation dataloader.  The
default output is ``<dataset_root>/calibration``; an explicit ``--output`` may
still point to a fresh directory outside the source dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm


SCRIPT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_SIGLIP_MODEL = "google/siglip2-base-patch16-384"
DEFAULT_DINO_REPO = Path(r"C:\Users\Kirill\Desktop\dinov3")
DEFAULT_DINO_WEIGHTS = Path(
    r"C:\Develop\3deye-analytics\tools\dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth"
)

BRIGHTNESS_NAMES = ("very_dark", "dark", "normal", "bright", "very_bright")
SCALE_NAMES = ("small", "medium", "large")


@dataclass(frozen=True)
class Sample:
    image: Path
    label: Path
    relative_image: Path
    classes: Tuple[int, ...]
    object_classes: Tuple[int, ...]
    median_box_area: float


@dataclass(frozen=True)
class ImageStats:
    brightness: float
    contrast: float
    saturation: float
    sharpness: float
    width: int
    height: int
    grayscale: bool


@dataclass
class SelectionResult:
    selected: List[int]
    reasons: Dict[int, List[str]]
    relaxed_duplicate_picks: int
    nearest_distance: np.ndarray
    cctv_high_cutoff: float
    cctv_low_cutoff: float


class SelectionError(RuntimeError):
    pass


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="YOLO dataset root or data.yaml.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Fresh output subset directory (default: <dataset_root>/calibration).",
    )
    parser.add_argument("--samples", type=int, default=100, help="Exact subset size (default: 100).")
    parser.add_argument("--apply", action="store_true", help="Materialize the subset. Omit for dry run.")
    parser.add_argument(
        "--copy-mode",
        choices=("copy", "hardlink"),
        default="copy",
        help="How to materialize selected pairs (default: independent copies).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional embedding cache. Dry run reads an existing cache but never writes it.",
    )

    parser.add_argument("--dinov3-repo", type=Path, default=DEFAULT_DINO_REPO)
    parser.add_argument("--dinov3-weights", type=Path, default=DEFAULT_DINO_WEIGHTS)
    parser.add_argument(
        "--dinov3-arch",
        default="auto",
        help="DINOv3 hub entrypoint or auto (inferred from checkpoint name).",
    )
    parser.add_argument("--embedder-image-size", type=int, default=384)
    parser.add_argument("--dino-batch-size", type=int, default=64)
    parser.add_argument("--siglip-model", default=DEFAULT_SIGLIP_MODEL)
    parser.add_argument("--siglip-batch-size", type=int, default=32)
    parser.add_argument("--siglip-clusters", type=int, default=32)
    parser.add_argument("--siglip-cluster-seed", type=int, default=42)
    parser.add_argument(
        "--allow-siglip-download",
        action="store_true",
        help="Allow Hugging Face network access. Default is local cache only.",
    )

    parser.add_argument("--min-per-class", type=int, default=8)
    parser.add_argument("--min-per-brightness", type=int, default=4)
    parser.add_argument("--min-per-scale", type=int, default=4)
    parser.add_argument("--min-cctv-high", type=int, default=15)
    parser.add_argument("--min-cctv-low", type=int, default=10)
    parser.add_argument(
        "--dedup-cosine-threshold",
        type=float,
        default=0.985,
        help="Reject a pick when its DINO cosine similarity to a selected image reaches this value.",
    )

    parser.add_argument(
        "--write-npz",
        action="store_true",
        help="After --apply, write calibration.fp16.npz through the real val dataloader.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="EdgeCrafter model config required by --write-npz.",
    )
    parser.add_argument("--input-size", nargs="+", type=int, default=[512])
    parser.add_argument("--calibration-batch-size", type=int, default=1)
    parser.add_argument("--input-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--workers", type=int, default=8, help="Image-stat decode workers.")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.samples <= 0:
        raise SelectionError("--samples must be positive.")
    for name in (
        "min_per_class",
        "min_per_brightness",
        "min_per_scale",
        "min_cctv_high",
        "min_cctv_low",
    ):
        if int(getattr(args, name)) < 0:
            raise SelectionError(f"--{name.replace('_', '-')} must be non-negative.")
    if not 0.0 < args.dedup_cosine_threshold <= 1.0:
        raise SelectionError("--dedup-cosine-threshold must be in (0, 1].")
    if args.siglip_clusters <= 0:
        raise SelectionError("--siglip-clusters must be positive.")
    if args.dino_batch_size <= 0 or args.siglip_batch_size <= 0:
        raise SelectionError("Embedding batch sizes must be positive.")
    if args.calibration_batch_size <= 0:
        raise SelectionError("--calibration-batch-size must be positive.")
    if args.samples % args.calibration_batch_size != 0:
        raise SelectionError("--samples must be divisible by --calibration-batch-size.")
    if args.write_npz and not args.apply:
        raise SelectionError("--write-npz requires --apply.")
    if args.write_npz and args.config is None:
        raise SelectionError("--write-npz requires --config.")
    if not args.dinov3_repo.is_dir():
        raise SelectionError(f"DINOv3 repository not found: {args.dinov3_repo}")
    if not args.dinov3_weights.is_file():
        raise SelectionError(f"DINOv3 checkpoint not found: {args.dinov3_weights}")


def _find_data_file(path: Path) -> Path:
    if path.is_file():
        return path.resolve()
    for name in ("data.yaml", "data.yml", "dataset.yaml", "dataset.yml"):
        candidate = path / name
        if candidate.is_file():
            return candidate.resolve()
    raise SelectionError(f"Cannot find YOLO data yaml in {path}")


def resolve_yolo_validation(data: Path) -> Tuple[Path, Path, Path, Path, Dict[str, Any]]:
    data_arg = data.expanduser().resolve()
    data_file = _find_data_file(data_arg)
    dataset_root = data_arg if data_arg.is_dir() else data_file.parent
    cfg = yaml.safe_load(data_file.read_text(encoding="utf-8")) or {}
    val_entry = cfg.get("val")
    if not isinstance(val_entry, str):
        raise SelectionError(f"YOLO data yaml must contain one string val entry: {data_file}")

    # Match export_trt_eval.py: an explicit --data directory is authoritative,
    # even when a copied data.yaml retains a stale `path:` value.
    val_path = Path(val_entry).expanduser()
    if not val_path.is_absolute():
        val_path = dataset_root / val_path
    val_path = val_path.resolve()
    if not val_path.is_dir():
        raise SelectionError(f"Validation image directory not found: {val_path}")

    try:
        rel = val_path.relative_to(dataset_root)
    except ValueError as exc:
        raise SelectionError(f"Validation directory is outside dataset root: {val_path}") from exc
    rel_parts = list(rel.parts)
    try:
        images_index = next(index for index, part in enumerate(rel_parts) if part.lower() == "images")
    except StopIteration as exc:
        raise SelectionError(
            "The validation path must contain an images directory so the matching labels "
            "directory can be inferred."
        ) from exc
    rel_parts[images_index] = "labels"
    label_path = dataset_root.joinpath(*rel_parts).resolve()
    if not label_path.is_dir():
        raise SelectionError(f"Validation label directory not found: {label_path}")
    return dataset_root, data_file, val_path, label_path, cfg


def _parse_label(path: Path, nc: int) -> Tuple[Tuple[int, ...], Tuple[int, ...], float]:
    classes: List[int] = []
    areas: List[float] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise SelectionError(f"{path}:{line_no}: expected 5 YOLO fields, got {len(parts)}")
        try:
            class_value = float(parts[0])
            x, y, width, height = (float(value) for value in parts[1:])
        except ValueError as exc:
            raise SelectionError(f"{path}:{line_no}: non-numeric YOLO value") from exc
        if not math.isfinite(class_value) or not class_value.is_integer():
            raise SelectionError(f"{path}:{line_no}: class id must be a finite integer")
        class_id = int(class_value)
        if not 0 <= class_id < nc:
            raise SelectionError(f"{path}:{line_no}: class {class_id} is outside [0, {nc - 1}]")
        if not all(math.isfinite(value) for value in (x, y, width, height)):
            raise SelectionError(f"{path}:{line_no}: non-finite YOLO geometry")
        if any(value < 0.0 or value > 1.0 for value in (x, y, width, height)):
            raise SelectionError(f"{path}:{line_no}: normalized coordinates must be within [0, 1]")
        if width <= 0.0 or height <= 0.0:
            raise SelectionError(f"{path}:{line_no}: non-positive box size")
        classes.append(class_id)
        areas.append(width * height)
    median_area = float(np.median(areas)) if areas else 0.0
    return tuple(sorted(set(classes))), tuple(classes), median_area


def index_validation(
    image_root: Path,
    label_root: Path,
    data_cfg: Dict[str, Any],
) -> List[Sample]:
    names = data_cfg.get("names")
    nc_value = data_cfg.get("nc")
    nc = int(nc_value) if nc_value is not None else len(names or [])
    if nc <= 0:
        raise SelectionError("Dataset nc/names does not define any classes.")

    image_files = sorted(
        path
        for path in image_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_files:
        raise SelectionError(f"No validation images found in {image_root}")

    seen_stems: Dict[Path, Path] = {}
    samples: List[Sample] = []
    invalid_images: List[str] = []
    for image in image_files:
        relative = image.relative_to(image_root)
        stem_key = relative.with_suffix("")
        if stem_key in seen_stems:
            raise SelectionError(
                f"Ambiguous duplicate image stem {stem_key}: {seen_stems[stem_key]} and {image}"
            )
        seen_stems[stem_key] = image
        label = label_root / relative.with_suffix(".txt")
        if not label.is_file():
            raise SelectionError(f"Missing label for {image}: {label}")
        try:
            classes, object_classes, median_area = _parse_label(label, nc)
        except SelectionError as exc:
            invalid_images.append(str(exc))
            continue
        samples.append(
            Sample(
                image=image,
                label=label,
                relative_image=relative,
                classes=classes,
                object_classes=object_classes,
                median_box_area=median_area,
            )
        )
    if invalid_images:
        shown = "\n".join(invalid_images[:20])
        hidden = len(invalid_images) - 20
        if hidden > 0:
            shown += f"\n... and {hidden} more"
        warnings.warn(
            f"Excluded {len(invalid_images)} validation image(s) with invalid YOLO labels:\n{shown}",
            UserWarning,
        )
    if not samples:
        raise SelectionError("No validation images with valid YOLO labels remain.")
    return samples


def dataset_fingerprint(samples: Sequence[Sample]) -> str:
    digest = hashlib.sha256()
    for sample in samples:
        image_stat = sample.image.stat()
        label_stat = sample.label.stat()
        row = (
            f"{sample.relative_image.as_posix()}\t"
            f"{image_stat.st_size}\t{image_stat.st_mtime_ns}\t"
            f"{label_stat.st_size}\t{label_stat.st_mtime_ns}\n"
        )
        digest.update(row.encode("utf-8"))
    return digest.hexdigest()


def _image_stats(path: Path) -> ImageStats:
    with Image.open(path) as image:
        width, height = image.size
        image = image.convert("RGB")
        image.thumbnail((160, 160), Image.Resampling.BILINEAR)
        rgb = np.asarray(image, dtype=np.float32) / np.float32(255.0)
    luma = (
        np.float32(0.2126) * rgb[..., 0]
        + np.float32(0.7152) * rgb[..., 1]
        + np.float32(0.0722) * rgb[..., 2]
    )
    channel_spread = rgb.max(axis=2) - rgb.min(axis=2)
    if luma.shape[0] >= 3 and luma.shape[1] >= 3:
        laplacian = (
            -4.0 * luma[1:-1, 1:-1]
            + luma[:-2, 1:-1]
            + luma[2:, 1:-1]
            + luma[1:-1, :-2]
            + luma[1:-1, 2:]
        )
        sharpness = float(np.var(laplacian))
    else:
        sharpness = 0.0
    return ImageStats(
        brightness=float(luma.mean()),
        contrast=float(luma.std()),
        saturation=float(channel_spread.mean()),
        sharpness=sharpness,
        width=int(width),
        height=int(height),
        grayscale=bool(float(channel_spread.mean()) < 0.015),
    )


def compute_stats(samples: Sequence[Sample], workers: int, quiet: bool) -> List[ImageStats]:
    paths = [sample.image for sample in samples]
    workers = max(1, min(int(workers), len(paths)))
    if workers == 1:
        iterator: Iterable[ImageStats] = (_image_stats(path) for path in paths)
        return list(tqdm(iterator, total=len(paths), desc="Image stats", disable=quiet))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        iterator = executor.map(_image_stats, paths)
        return list(tqdm(iterator, total=len(paths), desc="Image stats", disable=quiet))


def _normalise_rows(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32, copy=False)
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norm, np.float32(1e-12))


def _infer_dino_arch(weights: Path, requested: str) -> str:
    requested = requested.strip().lower()
    aliases = {name: f"dinov3_convnext_{name}" for name in ("tiny", "small", "base", "large")}
    if requested and requested != "auto":
        return aliases.get(requested, requested)
    name = weights.name.lower()
    for size, arch in aliases.items():
        if f"convnext_{size}" in name:
            return arch
    raise SelectionError(f"Cannot infer DINOv3 architecture from {weights.name}; pass --dinov3-arch.")


def _pool_dino(value: Any):
    import torch

    if not isinstance(value, torch.Tensor):
        raise RuntimeError(f"Expected tensor, got {type(value).__name__}")
    if value.ndim == 2:
        return value
    if value.ndim == 3:
        return value.mean(dim=1)
    if value.ndim == 4:
        return value.flatten(2).mean(dim=2)
    raise RuntimeError(f"Unsupported DINO tensor shape: {tuple(value.shape)}")


def _extract_dino(value: Any):
    import torch

    if isinstance(value, torch.Tensor):
        return _pool_dino(value)
    if isinstance(value, dict):
        for key in (
            "x_norm_clstoken",
            "x_norm_global_pool",
            "x_norm_patchtokens",
            "x_prenorm",
            "last_hidden_state",
        ):
            candidate = value.get(key)
            if isinstance(candidate, torch.Tensor):
                return _pool_dino(candidate)
        for candidate in value.values():
            try:
                return _extract_dino(candidate)
            except RuntimeError:
                pass
    if isinstance(value, (list, tuple)):
        for candidate in value:
            try:
                return _extract_dino(candidate)
            except RuntimeError:
                pass
    raise RuntimeError(f"Cannot extract DINO embedding from {type(value).__name__}")


def _forward_dino(model: Any, images: Any):
    forward_features = getattr(model, "forward_features", None)
    if callable(forward_features):
        try:
            return _extract_dino(forward_features(images))
        except (RuntimeError, TypeError):
            pass
    return _extract_dino(model(images))


def compute_dino_embeddings(samples: Sequence[Sample], args: argparse.Namespace) -> np.ndarray:
    import torch
    from torchvision.transforms import v2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and not args.quiet:
        print("warning: CUDA is unavailable; DINOv3 will run on CPU.", file=sys.stderr)
    arch = _infer_dino_arch(args.dinov3_weights, args.dinov3_arch)
    model = torch.hub.load(
        str(args.dinov3_repo),
        arch,
        source="local",
        weights=str(args.dinov3_weights),
    ).eval().to(device)
    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize((args.embedder_image_size, args.embedder_image_size), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    rows: List[np.ndarray] = []
    paths = [sample.image for sample in samples]
    progress = tqdm(total=len(paths), desc=f"DINOv3 {arch}", disable=args.quiet)
    try:
        with torch.inference_mode():
            for start in range(0, len(paths), args.dino_batch_size):
                tensors = []
                for path in paths[start : start + args.dino_batch_size]:
                    with Image.open(path) as image:
                        tensors.append(transform(image.convert("RGB")))
                batch = torch.stack(tensors).to(device, non_blocking=True)
                with torch.autocast(
                    device_type=device,
                    dtype=torch.float16 if device == "cuda" else torch.float32,
                    enabled=device == "cuda",
                ):
                    features = _forward_dino(model, batch)
                features = torch.nn.functional.normalize(features.float(), dim=1)
                rows.append(features.cpu().numpy().astype(np.float16))
                progress.update(len(tensors))
    finally:
        progress.close()
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    return np.concatenate(rows, axis=0)


def _feature_tensor(output: Any):
    if hasattr(output, "detach"):
        return output
    for name in ("pooler_output", "image_embeds", "text_embeds"):
        value = getattr(output, name, None)
        if value is not None:
            return value
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state[:, 0]
    raise RuntimeError(f"Cannot find feature tensor in {type(output).__name__}")


def compute_siglip_embeddings(
    samples: Sequence[Sample], args: argparse.Namespace
) -> Tuple[np.ndarray, np.ndarray]:
    import torch
    from transformers import AutoModel, AutoProcessor

    local_only = not args.allow_siglip_download
    processor = AutoProcessor.from_pretrained(
        args.siglip_model,
        local_files_only=local_only,
        use_fast=False,
    )
    model = AutoModel.from_pretrained(
        args.siglip_model,
        local_files_only=local_only,
    ).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    rows: List[np.ndarray] = []
    paths = [sample.image for sample in samples]
    progress = tqdm(total=len(paths), desc=f"SigLIP2 {args.siglip_model}", disable=args.quiet)
    try:
        with torch.inference_mode():
            for start in range(0, len(paths), args.siglip_batch_size):
                images = []
                for path in paths[start : start + args.siglip_batch_size]:
                    with Image.open(path) as image:
                        images.append(image.convert("RGB"))
                inputs = processor(images=images, return_tensors="pt")
                inputs = {name: value.to(device) for name, value in inputs.items()}
                with torch.autocast(
                    device_type=device,
                    dtype=torch.float16 if device == "cuda" else torch.float32,
                    enabled=device == "cuda",
                ):
                    if hasattr(model, "get_image_features"):
                        features = _feature_tensor(model.get_image_features(**inputs))
                    else:
                        features = _feature_tensor(model(**inputs))
                features = torch.nn.functional.normalize(features.float(), dim=1)
                rows.append(features.cpu().numpy().astype(np.float16))
                progress.update(len(images))

            positive = [
                "a frame from a fixed security camera",
                "CCTV surveillance footage",
                "an outdoor security camera view",
            ]
            negative = [
                "a close-up object photograph",
                "a professionally composed stock photo",
                "an illustration or screenshot",
            ]
            # SigLIP2 processor configs do not necessarily declare a tokenizer
            # model_max_length.  Dynamic padding is sufficient for this small
            # prompt batch and avoids a ragged input_ids tensor.
            text_inputs = processor(
                text=positive + negative,
                padding=True,
                return_tensors="pt",
            )
            text_inputs = {name: value.to(device) for name, value in text_inputs.items()}
            if hasattr(model, "get_text_features"):
                text_features = _feature_tensor(model.get_text_features(**text_inputs))
            else:
                text_features = _feature_tensor(model(**text_inputs))
            text_features = torch.nn.functional.normalize(text_features.float(), dim=1)
            text_np = text_features.cpu().numpy()
    finally:
        progress.close()

    embeddings = np.concatenate(rows, axis=0)
    logits = embeddings.astype(np.float32) @ text_np.T
    logit_scale = float(model.logit_scale.exp().item()) if hasattr(model, "logit_scale") else 1.0
    logit_bias = float(model.logit_bias.item()) if hasattr(model, "logit_bias") else 0.0
    logits = logits * np.float32(logit_scale) + np.float32(logit_bias)
    positive_score = logits[:, : len(positive)].max(axis=1)
    negative_score = logits[:, len(positive) :].max(axis=1)
    delta = np.clip(positive_score - negative_score, -60.0, 60.0)
    cctv_scores = (1.0 / (1.0 + np.exp(-delta))).astype(np.float32)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return embeddings, cctv_scores


def compute_siglip_clusters(embeddings: np.ndarray, count: int, seed: int) -> np.ndarray:
    from sklearn.cluster import MiniBatchKMeans

    count = max(1, min(int(count), embeddings.shape[0]))
    model = MiniBatchKMeans(
        n_clusters=count,
        random_state=int(seed),
        batch_size=max(1024, count * 8),
        n_init=3,
    )
    return model.fit_predict(embeddings.astype(np.float32)).astype(np.int32)


def _cache_identity(samples: Sequence[Sample], args: argparse.Namespace) -> Dict[str, Any]:
    weights_stat = args.dinov3_weights.stat()
    return {
        "schema_version": 1,
        "dataset_fingerprint": dataset_fingerprint(samples),
        "dino_arch": _infer_dino_arch(args.dinov3_weights, args.dinov3_arch),
        "dino_weights": args.dinov3_weights.name,
        "dino_weights_bytes": weights_stat.st_size,
        "dino_weights_mtime_ns": weights_stat.st_mtime_ns,
        "embedder_image_size": args.embedder_image_size,
        "siglip_model": args.siglip_model,
        "siglip_clusters": args.siglip_clusters,
        "siglip_cluster_seed": args.siglip_cluster_seed,
        "sample_count": len(samples),
    }


def load_cache(
    cache_dir: Optional[Path], identity: Dict[str, Any]
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    if cache_dir is None:
        return None
    meta_path = cache_dir / "meta.json"
    arrays_path = cache_dir / "embeddings.npz"
    if not meta_path.is_file() or not arrays_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta != identity:
            return None
        with np.load(arrays_path, allow_pickle=False) as archive:
            dino = archive["dino"]
            siglip = archive["siglip"]
            clusters = archive["clusters"]
            cctv = archive["cctv"]
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None
    if any(array.shape[0] != identity["sample_count"] for array in (dino, siglip, clusters, cctv)):
        return None
    return dino, siglip, clusters, cctv


def save_cache(
    cache_dir: Optional[Path],
    identity: Dict[str, Any],
    dino: np.ndarray,
    siglip: np.ndarray,
    clusters: np.ndarray,
    cctv: np.ndarray,
) -> None:
    if cache_dir is None:
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(cache_dir / "embeddings.npz", dino=dino, siglip=siglip, clusters=clusters, cctv=cctv)
    (cache_dir / "meta.json").write_text(
        json.dumps(identity, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def brightness_bucket(value: float) -> int:
    return int(np.searchsorted(np.asarray([0.15, 0.30, 0.55, 0.75]), value, side="right"))


def scale_bucket(value: float) -> int:
    if value < 0.01:
        return 0
    if value < 0.10:
        return 1
    return 2


def combine_embeddings(dino: np.ndarray, siglip: np.ndarray) -> np.ndarray:
    dino_norm = _normalise_rows(dino)
    siglip_norm = _normalise_rows(siglip)
    combined = np.concatenate([dino_norm, siglip_norm], axis=1)
    return _normalise_rows(combined)


def select_subset(
    samples: Sequence[Sample],
    stats: Sequence[ImageStats],
    dino: np.ndarray,
    siglip: np.ndarray,
    clusters: np.ndarray,
    cctv_scores: np.ndarray,
    args: argparse.Namespace,
) -> SelectionResult:
    import torch

    count = len(samples)
    if args.samples > count:
        raise SelectionError(f"Requested {args.samples} samples but val contains only {count}.")
    combined = combine_embeddings(dino, siglip)
    if not np.isfinite(combined).all():
        raise SelectionError("Embedding matrix contains NaN/Inf.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    combined_t = torch.as_tensor(combined, dtype=torch.float32, device=device)
    dino_t = torch.as_tensor(_normalise_rows(dino), dtype=torch.float32, device=device)
    selected_mask = torch.zeros(count, dtype=torch.bool, device=device)
    centroid = torch.nn.functional.normalize(combined_t.mean(dim=0, keepdim=True), dim=1)
    nearest_distance = 1.0 - (combined_t @ centroid.T).squeeze(1)
    max_dino_similarity = torch.full((count,), -1.0, dtype=torch.float32, device=device)
    selected: List[int] = []
    reasons: Dict[int, List[str]] = {}
    relaxed = 0

    class_sets = [set(sample.classes) for sample in samples]
    brightness = np.asarray([brightness_bucket(item.brightness) for item in stats], dtype=np.int8)
    scales = np.asarray([scale_bucket(sample.median_box_area) for sample in samples], dtype=np.int8)
    cctv_high_cutoff = float(np.quantile(cctv_scores, 0.75))
    cctv_low_cutoff = float(np.quantile(cctv_scores, 0.25))

    def add_reason(index: int, reason: str) -> None:
        values = reasons.setdefault(index, [])
        if reason not in values:
            values.append(reason)

    def selected_count(predicate) -> int:
        return sum(1 for index in selected if predicate(index))

    def pick(candidates: Iterable[int], reason: str) -> bool:
        nonlocal relaxed
        if len(selected) >= args.samples:
            return False
        candidate_list = sorted(set(int(index) for index in candidates))
        if not candidate_list:
            return False
        candidate_t = torch.as_tensor(candidate_list, dtype=torch.long, device=device)
        available = ~selected_mask[candidate_t]
        deduplicated = max_dino_similarity[candidate_t] < float(args.dedup_cosine_threshold)
        eligible = available & deduplicated
        if not bool(eligible.any().item()):
            eligible = available
            if not bool(eligible.any().item()):
                return False
            relaxed += 1
        scores = nearest_distance[candidate_t].clone()
        scores[~eligible] = float("-inf")
        local = int(torch.argmax(scores).item())
        chosen = candidate_list[local]
        if not math.isfinite(float(scores[local].item())):
            return False
        selected.append(chosen)
        selected_mask[chosen] = True
        add_reason(chosen, reason)
        new_distance = 1.0 - (combined_t @ combined_t[chosen])
        nearest_distance.copy_(torch.minimum(nearest_distance, new_distance))
        new_dino_similarity = dino_t @ dino_t[chosen]
        max_dino_similarity.copy_(torch.maximum(max_dino_similarity, new_dino_similarity))
        nearest_distance[selected_mask] = -1.0
        return True

    class_population = Counter(class_id for sample in samples for class_id in sample.classes)
    for class_id, _population in sorted(class_population.items(), key=lambda item: (item[1], item[0])):
        candidates = [index for index, classes in enumerate(class_sets) if class_id in classes]
        target = min(int(args.min_per_class), len(candidates))
        while selected_count(lambda index, class_id=class_id: class_id in class_sets[index]) < target:
            if not pick(candidates, f"class:{class_id}"):
                break

    for cluster_id in sorted(set(int(value) for value in clusters)):
        candidates = np.flatnonzero(clusters == cluster_id).tolist()
        if selected_count(lambda index, cluster_id=cluster_id: int(clusters[index]) == cluster_id) == 0:
            pick(candidates, f"siglip_cluster:{cluster_id}")

    for bucket, name in enumerate(BRIGHTNESS_NAMES):
        candidates = np.flatnonzero(brightness == bucket).tolist()
        target = min(int(args.min_per_brightness), len(candidates))
        while selected_count(lambda index, bucket=bucket: int(brightness[index]) == bucket) < target:
            if not pick(candidates, f"brightness:{name}"):
                break

    for bucket, name in enumerate(SCALE_NAMES):
        candidates = np.flatnonzero(scales == bucket).tolist()
        target = min(int(args.min_per_scale), len(candidates))
        while selected_count(lambda index, bucket=bucket: int(scales[index]) == bucket) < target:
            if not pick(candidates, f"scale:{name}"):
                break

    cctv_high = np.flatnonzero(cctv_scores >= cctv_high_cutoff).tolist()
    high_target = min(int(args.min_cctv_high), len(cctv_high))
    while selected_count(lambda index: float(cctv_scores[index]) >= cctv_high_cutoff) < high_target:
        if not pick(cctv_high, "cctv_score:top_quartile"):
            break

    cctv_low = np.flatnonzero(cctv_scores <= cctv_low_cutoff).tolist()
    low_target = min(int(args.min_cctv_low), len(cctv_low))
    while selected_count(lambda index: float(cctv_scores[index]) <= cctv_low_cutoff) < low_target:
        if not pick(cctv_low, "cctv_score:bottom_quartile"):
            break

    extremes = {
        "appearance:darkest": int(np.argmin([item.brightness for item in stats])),
        "appearance:brightest": int(np.argmax([item.brightness for item in stats])),
        "appearance:highest_contrast": int(np.argmax([item.contrast for item in stats])),
        "appearance:lowest_sharpness": int(np.argmin([item.sharpness for item in stats])),
        "appearance:highest_sharpness": int(np.argmax([item.sharpness for item in stats])),
    }
    grayscale = [index for index, item in enumerate(stats) if item.grayscale]
    if grayscale:
        extremes["appearance:grayscale"] = grayscale[0]
    for reason, index in extremes.items():
        if selected_mask[index]:
            add_reason(index, reason)
        else:
            pick([index], reason)

    all_candidates = range(count)
    while len(selected) < args.samples:
        if not pick(all_candidates, "global_kcenter"):
            raise SelectionError(f"Could select only {len(selected)} / {args.samples} samples.")

    selected_class_counts = Counter(class_id for index in selected for class_id in samples[index].classes)
    unmet = {
        class_id: min(args.min_per_class, population) - selected_class_counts[class_id]
        for class_id, population in class_population.items()
        if selected_class_counts[class_id] < min(args.min_per_class, population)
    }
    if unmet:
        raise SelectionError(f"Class coverage constraints do not fit in {args.samples} samples: {unmet}")
    unmet_groups: Dict[str, int] = {}
    for cluster_id in sorted(set(int(value) for value in clusters)):
        actual = selected_count(
            lambda index, cluster_id=cluster_id: int(clusters[index]) == cluster_id
        )
        if actual < 1:
            unmet_groups[f"siglip_cluster:{cluster_id}"] = 1 - actual
    for bucket, name in enumerate(BRIGHTNESS_NAMES):
        available = int((brightness == bucket).sum())
        target = min(int(args.min_per_brightness), available)
        actual = selected_count(lambda index, bucket=bucket: int(brightness[index]) == bucket)
        if actual < target:
            unmet_groups[f"brightness:{name}"] = target - actual
    for bucket, name in enumerate(SCALE_NAMES):
        available = int((scales == bucket).sum())
        target = min(int(args.min_per_scale), available)
        actual = selected_count(lambda index, bucket=bucket: int(scales[index]) == bucket)
        if actual < target:
            unmet_groups[f"scale:{name}"] = target - actual
    high_actual = selected_count(lambda index: float(cctv_scores[index]) >= cctv_high_cutoff)
    high_target = min(int(args.min_cctv_high), len(cctv_high))
    if high_actual < high_target:
        unmet_groups["cctv_score:top_quartile"] = high_target - high_actual
    low_actual = selected_count(lambda index: float(cctv_scores[index]) <= cctv_low_cutoff)
    low_target = min(int(args.min_cctv_low), len(cctv_low))
    if low_actual < low_target:
        unmet_groups["cctv_score:bottom_quartile"] = low_target - low_actual
    if unmet_groups:
        raise SelectionError(
            f"Coverage constraints do not fit in {args.samples} samples: {unmet_groups}"
        )
    if len(set(selected)) != args.samples:
        raise SelectionError("Internal error: selection is not exact and unique.")
    return SelectionResult(
        selected=selected,
        reasons=reasons,
        relaxed_duplicate_picks=relaxed,
        nearest_distance=nearest_distance.detach().cpu().numpy(),
        cctv_high_cutoff=cctv_high_cutoff,
        cctv_low_cutoff=cctv_low_cutoff,
    )


def _class_names(data_cfg: Dict[str, Any]) -> Dict[int, str]:
    names = data_cfg.get("names") or {}
    if isinstance(names, list):
        return {index: str(name) for index, name in enumerate(names)}
    return {int(key): str(value) for key, value in names.items()}


def selection_report(
    samples: Sequence[Sample],
    stats: Sequence[ImageStats],
    clusters: np.ndarray,
    cctv_scores: np.ndarray,
    dino: np.ndarray,
    result: SelectionResult,
    data_cfg: Dict[str, Any],
    fingerprint: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    selected = result.selected
    class_names = _class_names(data_cfg)

    def image_class_counts(indices: Sequence[int]) -> Counter:
        return Counter(class_id for index in indices for class_id in samples[index].classes)

    def object_class_counts(indices: Sequence[int]) -> Counter:
        return Counter(class_id for index in indices for class_id in samples[index].object_classes)

    source_indices = list(range(len(samples)))
    selected_dino = _normalise_rows(dino[selected])
    similarity = selected_dino @ selected_dino.T
    np.fill_diagonal(similarity, -1.0)
    max_pair_similarity = float(similarity.max()) if len(selected) > 1 else -1.0
    brightness_source = Counter(BRIGHTNESS_NAMES[brightness_bucket(item.brightness)] for item in stats)
    brightness_selected = Counter(
        BRIGHTNESS_NAMES[brightness_bucket(stats[index].brightness)] for index in selected
    )
    scale_source = Counter(SCALE_NAMES[scale_bucket(sample.median_box_area)] for sample in samples)
    scale_selected = Counter(SCALE_NAMES[scale_bucket(samples[index].median_box_area)] for index in selected)
    source_class_images = image_class_counts(source_indices)
    selected_class_images = image_class_counts(selected)
    source_class_objects = object_class_counts(source_indices)
    selected_class_objects = object_class_counts(selected)
    remaining = result.nearest_distance[result.nearest_distance >= 0.0]

    entries = []
    for rank, index in enumerate(selected, start=1):
        sample = samples[index]
        item = stats[index]
        entries.append(
            {
                "rank": rank,
                "relative_image": sample.relative_image.as_posix(),
                "classes": list(sample.classes),
                "class_names": [class_names.get(value, str(value)) for value in sample.classes],
                "objects": len(sample.object_classes),
                "median_box_area": sample.median_box_area,
                "brightness": item.brightness,
                "contrast": item.contrast,
                "saturation": item.saturation,
                "sharpness": item.sharpness,
                "grayscale": item.grayscale,
                "siglip_cluster": int(clusters[index]),
                "cctv_score": float(cctv_scores[index]),
                "reasons": result.reasons.get(index, []),
            }
        )

    return {
        "schema_version": 1,
        "mode": "apply" if args.apply else "dry_run",
        "source": {
            "data": str(args.data.resolve()),
            "validation_images": len(samples),
            "fingerprint_sha256": fingerprint,
        },
        "selection": {
            "requested": args.samples,
            "selected": len(selected),
            "unique": len(set(selected)),
            "dino_arch": _infer_dino_arch(args.dinov3_weights, args.dinov3_arch),
            "dino_weights": str(args.dinov3_weights.resolve()),
            "siglip_model": args.siglip_model,
            "siglip_clusters_source": len(set(int(value) for value in clusters)),
            "siglip_clusters_selected": len(set(int(clusters[index]) for index in selected)),
            "dedup_cosine_threshold": args.dedup_cosine_threshold,
            "dedup_relaxed_picks": result.relaxed_duplicate_picks,
            "selected_max_pairwise_dino_cosine": max_pair_similarity,
            "combined_coverage_nearest_distance": {
                "mean": float(remaining.mean()) if remaining.size else 0.0,
                "p95": float(np.quantile(remaining, 0.95)) if remaining.size else 0.0,
                "max": float(remaining.max()) if remaining.size else 0.0,
            },
            "cctv_score": {
                "source_q25": result.cctv_low_cutoff,
                "source_q75": result.cctv_high_cutoff,
                "selected_min": float(cctv_scores[selected].min()),
                "selected_median": float(np.median(cctv_scores[selected])),
                "selected_max": float(cctv_scores[selected].max()),
            },
        },
        "coverage": {
            "class_image_counts_source": {str(k): v for k, v in sorted(source_class_images.items())},
            "class_image_counts_selected": {str(k): v for k, v in sorted(selected_class_images.items())},
            "class_object_counts_source": {str(k): v for k, v in sorted(source_class_objects.items())},
            "class_object_counts_selected": {str(k): v for k, v in sorted(selected_class_objects.items())},
            "brightness_source": dict(sorted(brightness_source.items())),
            "brightness_selected": dict(sorted(brightness_selected.items())),
            "scale_source": dict(sorted(scale_source.items())),
            "scale_selected": dict(sorted(scale_selected.items())),
        },
        "samples": entries,
    }


def print_summary(report: Dict[str, Any], output: Path) -> None:
    selection = report["selection"]
    coverage = report["coverage"]
    print(
        f"Mode: {report['mode']}\n"
        f"Validation images: {report['source']['validation_images']}\n"
        f"Selected: {selection['selected']} unique images\n"
        f"SigLIP clusters: {selection['siglip_clusters_selected']} / "
        f"{selection['siglip_clusters_source']}\n"
        f"Max selected DINO cosine: {selection['selected_max_pairwise_dino_cosine']:.6f}\n"
        f"Dedup-relaxed picks: {selection['dedup_relaxed_picks']}\n"
        f"Class image coverage: {coverage['class_image_counts_selected']}\n"
        f"Brightness coverage: {coverage['brightness_selected']}\n"
        f"Scale coverage: {coverage['scale_selected']}\n"
        f"Output: {output}"
    )
    if report["mode"] == "dry_run":
        print("Dry run: no subset, report, NPZ, or cache files were created.")


def _portable_data_yaml(data_cfg: Dict[str, Any]) -> str:
    class_names = _class_names(data_cfg)
    nc = int(data_cfg.get("nc", len(class_names)))
    lines = ["train: images/val", "val: images/val", f"nc: {nc}", "names:"]
    for class_id in range(nc):
        lines.append(f"  {class_id}: {class_names.get(class_id, str(class_id))}")
    return "\n".join(lines) + "\n"


def materialize_subset(
    output: Path,
    samples: Sequence[Sample],
    report: Dict[str, Any],
    data_cfg: Dict[str, Any],
    copy_mode: str,
) -> None:
    output = output.resolve()
    if output.exists():
        raise SelectionError(f"Refusing to overwrite existing output: {output}")
    selected_by_rel = {
        entry["relative_image"]: entry for entry in report["samples"]
    }
    sample_by_rel = {sample.relative_image.as_posix(): sample for sample in samples}
    try:
        for relative_text in selected_by_rel:
            sample = sample_by_rel[relative_text]
            relative = Path(relative_text)
            destination_image = output / "images" / "val" / relative
            destination_label = output / "labels" / "val" / relative.with_suffix(".txt")
            destination_image.parent.mkdir(parents=True, exist_ok=True)
            destination_label.parent.mkdir(parents=True, exist_ok=True)
            if copy_mode == "hardlink":
                os.link(sample.image, destination_image)
                os.link(sample.label, destination_label)
            else:
                shutil.copy2(sample.image, destination_image)
                shutil.copy2(sample.label, destination_label)
        (output / "data.yaml").write_text(_portable_data_yaml(data_cfg), encoding="utf-8")
        (output / "selection.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except BaseException:
        if output.exists():
            shutil.rmtree(output)
        raise


def validate_materialized(output: Path, expected: int) -> Dict[str, int]:
    images_root = output / "images" / "val"
    labels_root = output / "labels" / "val"
    images = sorted(
        path for path in images_root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    labels = sorted(labels_root.rglob("*.txt"))
    image_stems = {path.relative_to(images_root).with_suffix("") for path in images}
    label_stems = {path.relative_to(labels_root).with_suffix("") for path in labels}
    missing = image_stems - label_stems
    orphan = label_stems - image_stems
    if len(images) != expected or len(labels) != expected or missing or orphan:
        raise SelectionError(
            f"Materialized subset failed validation: images={len(images)}, labels={len(labels)}, "
            f"missing={len(missing)}, orphan={len(orphan)}"
        )
    return {"images": len(images), "labels": len(labels), "missing": 0, "orphan": 0}


def _parse_input_size(values: Sequence[int]) -> List[int]:
    if len(values) == 1:
        height = width = int(values[0])
    elif len(values) == 2:
        height, width = (int(value) for value in values)
    else:
        raise SelectionError("--input-size expects S or H W.")
    if height <= 0 or width <= 0 or height % 32 or width % 32:
        raise SelectionError("--input-size values must be positive and divisible by 32.")
    return [height, width]


def write_calibration_npz(output: Path, args: argparse.Namespace) -> Path:
    sys.path.insert(0, str(SCRIPT_ROOT))
    from engine.core import YAMLConfig
    from engine.core.yaml_utils import merge_dict
    from tools.deployment.export_trt_eval import write_modelopt_calibration_data

    config = args.config.expanduser()
    if not config.is_absolute():
        config = SCRIPT_ROOT / config
    config = config.resolve()
    if not config.is_file():
        raise SelectionError(f"EdgeCrafter config not found: {config}")
    input_size = _parse_input_size(args.input_size)
    update: Dict[str, Any] = {
        "yolo_root": str(output.resolve()),
        "yolo_data_file": str((output / "data.yaml").resolve()),
        "val_dataloader": {"total_batch_size": int(args.calibration_batch_size)},
    }
    update = merge_dict(update, {"eval_spatial_size": input_size})
    cfg = YAMLConfig(str(config), **update)
    npz_path = output / "calibration.fp16.npz"
    return write_modelopt_calibration_data(
        cfg,
        npz_path,
        sample_count=int(args.samples),
        export_mode="normalized",
        input_dtype=args.input_dtype,
    )


def resolve_output_path(dataset_root: Path, requested: Optional[Path]) -> Path:
    dataset_root = dataset_root.resolve()
    default_output = (dataset_root / "calibration").resolve()
    output = default_output if requested is None else requested.expanduser().resolve()
    if output == dataset_root:
        raise SelectionError("--output must not be the source dataset root.")
    if dataset_root in output.parents and output != default_output:
        raise SelectionError(
            f"Only {default_output} is allowed inside the source dataset; "
            "choose a directory outside the dataset or omit --output."
        )
    return output


def run(args: argparse.Namespace) -> int:
    validate_args(args)
    dataset_root, data_file, image_root, label_root, data_cfg = resolve_yolo_validation(args.data)
    args.output = resolve_output_path(dataset_root, args.output)
    samples = index_validation(image_root, label_root, data_cfg)
    if args.apply and args.output.exists():
        raise SelectionError(f"Refusing to overwrite existing output: {args.output.resolve()}")

    if not args.quiet:
        print(f"YOLO data: {data_file}")
        print(f"Validation images: {image_root} ({len(samples)})")
        print(f"Validation labels: {label_root}")
    identity = _cache_identity(samples, args)
    cached = load_cache(args.cache_dir, identity)
    if cached is None:
        stats = compute_stats(samples, args.workers, args.quiet)
        dino = compute_dino_embeddings(samples, args)
        siglip, cctv = compute_siglip_embeddings(samples, args)
        clusters = compute_siglip_clusters(siglip, args.siglip_clusters, args.siglip_cluster_seed)
        if args.apply:
            save_cache(args.cache_dir, identity, dino, siglip, clusters, cctv)
    else:
        if not args.quiet:
            print(f"Embedding cache hit: {args.cache_dir}")
        dino, siglip, clusters, cctv = cached
        stats = compute_stats(samples, args.workers, args.quiet)

    result = select_subset(samples, stats, dino, siglip, clusters, cctv, args)
    fingerprint = identity["dataset_fingerprint"]
    report = selection_report(
        samples, stats, clusters, cctv, dino, result, data_cfg, fingerprint, args
    )
    print_summary(report, args.output.resolve())
    if not args.apply:
        return 0

    materialize_subset(args.output, samples, report, data_cfg, args.copy_mode)
    validation = validate_materialized(args.output, args.samples)
    print(f"Subset validation: {validation}")
    if args.write_npz:
        npz_path = write_calibration_npz(args.output, args)
        with np.load(npz_path, allow_pickle=False) as archive:
            images = archive["images"]
            if images.shape[0] != args.samples:
                raise SelectionError(
                    f"Calibration NPZ has {images.shape[0]} samples, expected {args.samples}."
                )
            print(
                f"Calibration NPZ: {npz_path.resolve()} shape={images.shape} "
                f"dtype={images.dtype} range=[{images.min():.6g}, {images.max():.6g}]"
            )
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        return run(parse_args(argv))
    except (SelectionError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
