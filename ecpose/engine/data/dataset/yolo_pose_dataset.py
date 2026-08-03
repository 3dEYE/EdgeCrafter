"""YOLO four-corner keypoint dataset support for ECPose."""

import math
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from PIL import Image
from faster_coco_eval import COCO
import yaml

from ...core import register
from .._misc import convert_to_tv_tensor
from ._dataset import DetDataset

Image.MAX_IMAGE_PIXELS = None

__all__ = ["YOLOPoseDetection"]


def _as_path(path: Optional[str]) -> Optional[Path]:
    return None if path is None else Path(path).expanduser()


def _resolve_path(path: str, root: Optional[Path]) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() or root is None else root / candidate


def _find_data_file(root: Optional[Path]) -> Optional[Path]:
    if root is None:
        return None
    for name in ("data.yaml", "data.yml", "data_win.yaml", "dataset.yaml", "dataset.yml"):
        candidate = root / name
        if candidate.exists():
            return candidate
    return None


def _normalize_names(names: Any, num_classes: Optional[int]) -> List[str]:
    if names is None:
        return [str(index) for index in range(int(num_classes or 0))]
    if isinstance(names, dict):
        return [str(names[key]) for key in sorted(names, key=lambda value: int(value))]
    if isinstance(names, (list, tuple)):
        return [str(name) for name in names]
    raise TypeError(f"Unsupported YOLO names format: {type(names)!r}")


@register()
class YOLOPoseDetection(DetDataset):
    """Read four-corner YOLO labels as named ECPose keypoints.

    Each non-empty label row must be:
        class x_lt y_lt x_rt y_rt x_rb y_rb x_lb y_lb
    Coordinates may be normalized (default) or absolute. All four supplied
    keypoints are marked visible using the COCO visibility value 2.
    """

    __inject__ = ["transforms"]
    __share__ = ["num_classes", "yolo_root", "yolo_data_file"]

    keypoint_names = ["left_top", "right_top", "right_bottom", "left_bottom"]
    skeleton = [[1, 2], [2, 3], [3, 4], [4, 1]]

    #: How many individual label problems are spelled out before summarising.
    max_reported_problems = 20

    def __init__(
        self,
        img_folder: Optional[str] = None,
        label_folder: Optional[str] = None,
        root: Optional[str] = None,
        split: Optional[str] = None,
        data_file: Optional[str] = None,
        yolo_root: Optional[str] = None,
        yolo_data_file: Optional[str] = None,
        transforms=None,
        names: Any = None,
        num_classes: Optional[int] = None,
        num_keypoints: int = 4,
        image_extensions: Optional[Sequence[str]] = None,
        recursive: bool = True,
        normalized: bool = True,
        strict: bool = True,
        validate_on_init: bool = True,
        **kwargs,
    ):
        if num_keypoints != 4:
            raise ValueError("YOLOPoseDetection currently requires num_keypoints=4")
        if root is None:
            root = yolo_root
        if data_file is None:
            data_file = yolo_data_file

        root_path = _as_path(root)
        if data_file is None:
            data_file = _find_data_file(root_path)
        self.data_file = _as_path(data_file)
        data_cfg = self._load_data_cfg(self.data_file)
        data_dir = self.data_file.parent if self.data_file else None
        self.root = root_path or data_dir

        if names is None and data_cfg:
            names = data_cfg.get("names")
        if num_classes is None and data_cfg:
            num_classes = data_cfg.get("nc")
        self.names = _normalize_names(names, num_classes)
        self.num_classes = int(num_classes if num_classes is not None else len(self.names))
        self.num_keypoints = num_keypoints

        source: Optional[Path] = None
        if split is not None:
            if not data_cfg or split not in data_cfg:
                raise ValueError(f"YOLO data file does not define split {split!r}")
            source = _resolve_path(str(data_cfg[split]), self.root)
        elif img_folder is not None:
            source = _resolve_path(img_folder, self.root)
        if source is None:
            raise ValueError("YOLOPoseDetection requires img_folder or a configured split")

        self.img_folder = source if source.is_dir() else None
        self.label_folder = _as_path(label_folder)
        self._transforms = transforms
        self.normalized = normalized
        self.strict = strict
        self.validate_on_init = validate_on_init
        # When set, invalid labels are collected instead of raised/warned so a
        # single validation pass can report everything that is wrong.
        self._problem_sink: Optional[List[str]] = None
        self._skipped_labels = 0

        extensions = image_extensions or [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
        self.image_extensions = {
            ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions
        }
        self.image_files = self._collect_images(source, recursive)
        if not self.image_files:
            raise FileNotFoundError(f"No images found from {source}")
        if self.validate_on_init:
            self._validate_dataset()

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx):
        image, target = self.load_item(idx)
        if self._transforms is not None:
            image, target = self._transforms(image, target, self)
        return image, target

    def load_item(self, idx):
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        boxes, labels, keypoints, areas = self._parse_label_file(
            self._label_path(image_path), width, height
        )
        target = {
            "boxes": convert_to_tv_tensor(boxes, key="boxes", spatial_size=image.size[::-1]),
            "labels": labels,
            "keypoints": keypoints,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": areas,
            "iscrowd": torch.zeros((labels.shape[0],), dtype=torch.int64),
            # DETRPosePostProcessor expects target sizes in (width, height) order.
            "orig_size": torch.as_tensor([width, height]),
            "size": torch.as_tensor([height, width]),
            "idx": torch.tensor([idx], dtype=torch.int64),
        }
        return image, target

    @property
    def categories(self) -> List[Dict[str, Any]]:
        names = self.names or [str(index) for index in range(self.num_classes)]
        return [
            {
                "id": index,
                "name": name,
                "keypoints": self.keypoint_names,
                "skeleton": self.skeleton,
            }
            for index, name in enumerate(names)
        ]

    @property
    def category2name(self) -> Dict[int, str]:
        return {category["id"]: category["name"] for category in self.categories}

    @property
    def category2label(self) -> Dict[int, int]:
        return {category["id"]: index for index, category in enumerate(self.categories)}

    @property
    def label2category(self) -> Dict[int, int]:
        return {index: category["id"] for index, category in enumerate(self.categories)}

    def get_coco_api(self) -> COCO:
        coco_ds = COCO()
        dataset = {"images": [], "categories": self.categories, "annotations": []}
        annotation_id = 1
        for image_id, image_path in enumerate(self.image_files):
            with Image.open(image_path) as image:
                width, height = image.size
            dataset["images"].append(
                {
                    "id": image_id,
                    "file_name": image_path.name,
                    "width": width,
                    "height": height,
                }
            )
            boxes, labels, keypoints, areas = self._parse_label_file(
                self._label_path(image_path), width, height
            )
            for object_index in range(labels.shape[0]):
                x1, y1, x2, y2 = boxes[object_index].tolist()
                dataset["annotations"].append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": int(labels[object_index]),
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "keypoints": keypoints[object_index].reshape(-1).tolist(),
                        "num_keypoints": self.num_keypoints,
                        "area": float(areas[object_index]),
                        "iscrowd": 0,
                    }
                )
                annotation_id += 1
        coco_ds.dataset = dataset
        coco_ds.createIndex()
        return coco_ds

    def extra_repr(self) -> str:
        return (
            f" images: {len(self.image_files)}\n"
            f" num_classes: {self.num_classes}\n"
            f" num_keypoints: {self.num_keypoints}\n"
        )

    @staticmethod
    def _load_data_cfg(data_file: Optional[Path]) -> Dict[str, Any]:
        if data_file is None:
            return {}
        with data_file.open("r", encoding="utf-8") as stream:
            return yaml.safe_load(stream) or {}

    def _collect_images(self, source: Path, recursive: bool) -> List[Path]:
        if source.is_file():
            base = self.root or source.parent
            image_files = []
            for raw_line in source.read_text(encoding="utf-8-sig").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                image_path = _resolve_path(line, base)
                if image_path.suffix.lower() not in self.image_extensions:
                    raise ValueError(f"Unsupported image extension in {source}: {line}")
                if not image_path.is_file():
                    raise FileNotFoundError(f"Image listed in {source} does not exist: {image_path}")
                image_files.append(image_path)
            return sorted(dict.fromkeys(image_files))
        if not source.is_dir():
            raise FileNotFoundError(f"YOLO image source does not exist: {source}")
        iterator = source.rglob("*") if recursive else source.glob("*")
        return sorted(
            path for path in iterator if path.is_file() and path.suffix.lower() in self.image_extensions
        )

    def _label_path(self, image_path: Path) -> Path:
        if self.label_folder is not None:
            if self.img_folder is None:
                return self.label_folder / image_path.with_suffix(".txt").name
            return self.label_folder / image_path.relative_to(self.img_folder).with_suffix(".txt")
        if self.root is not None:
            images_root = self.root / "images"
            try:
                relative = image_path.relative_to(images_root)
                return (self.root / "labels" / relative).with_suffix(".txt")
            except ValueError:
                pass
        parts = list(image_path.parts)
        if "images" in parts:
            parts[parts.index("images")] = "labels"
            return Path(*parts).with_suffix(".txt")
        raise ValueError(f"Cannot infer label path for {image_path}; provide label_folder")

    def _validate_dataset(self) -> None:
        """Collect every label problem in one pass instead of failing on the first."""
        problems: List[str] = []
        self._problem_sink = problems
        try:
            for image_path in self.image_files:
                if self.normalized:
                    width, height = 1, 1
                else:
                    with Image.open(image_path) as image:
                        width, height = image.size
                self._parse_label_file(self._label_path(image_path), width, height)
        finally:
            self._problem_sink = None

        if not problems:
            return

        listed = "\n".join(problems[: self.max_reported_problems])
        hidden = len(problems) - self.max_reported_problems
        if hidden > 0:
            listed += f"\n... and {hidden} more"
        location = self.img_folder or self.root
        summary = f"Found {len(problems)} invalid label(s) under {location}:\n{listed}"
        if self.strict:
            raise ValueError(summary)
        warnings.warn(f"{summary}\nThese objects are skipped (strict=False).", UserWarning)

    def _parse_label_file(self, label_path: Path, width: int, height: int):
        if not label_path.exists():
            raise FileNotFoundError(f"Missing YOLO label file: {label_path}")
        boxes: List[List[float]] = []
        labels: List[int] = []
        keypoints: List[List[List[float]]] = []
        areas: List[float] = []
        for line_number, raw_line in enumerate(
            label_path.read_text(encoding="utf-8-sig").splitlines(), start=1
        ):
            line = raw_line.strip()
            if not line:
                continue
            parsed = self._parse_line(line, width, height, label_path, line_number)
            if parsed is None:
                continue
            class_id, points, area = parsed
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            boxes.append([min(xs), min(ys), max(xs), max(ys)])
            labels.append(class_id)
            keypoints.append([[x, y, 2.0] for x, y in points])
            areas.append(area)
        return (
            torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            torch.as_tensor(labels, dtype=torch.int64),
            torch.as_tensor(keypoints, dtype=torch.float32).reshape(-1, self.num_keypoints, 3),
            torch.as_tensor(areas, dtype=torch.float32),
        )

    def _parse_line(self, line, width, height, label_path, line_number):
        parts = line.split()
        if len(parts) != 9:
            return self._invalid_label(
                label_path,
                line_number,
                "expected exactly 9 fields: class plus four x/y keypoints",
            )
        try:
            class_value = float(parts[0])
            coordinates = [float(value) for value in parts[1:]]
        except ValueError:
            return self._invalid_label(label_path, line_number, "contains non-numeric values")
        if not math.isfinite(class_value) or not class_value.is_integer():
            return self._invalid_label(label_path, line_number, "class id must be a finite integer")
        if not all(math.isfinite(value) for value in coordinates):
            return self._invalid_label(label_path, line_number, "contains non-finite coordinates")
        class_id = int(class_value)
        if class_id < 0 or (self.num_classes > 0 and class_id >= self.num_classes):
            return self._invalid_label(label_path, line_number, f"class id {class_id} is out of range")
        if self.normalized and any(value < 0.0 or value > 1.0 for value in coordinates):
            return self._invalid_label(label_path, line_number, "normalized coordinates must be in [0, 1]")

        points = []
        for x, y in zip(coordinates[0::2], coordinates[1::2]):
            px = x * width if self.normalized else x
            py = y * height if self.normalized else y
            points.append((min(max(px, 0.0), float(width)), min(max(py, 0.0), float(height))))
        if len(set(points)) != self.num_keypoints:
            return self._invalid_label(label_path, line_number, "keypoints must be distinct")
        area = self._polygon_area(points)
        if area <= 0.0:
            return self._invalid_label(label_path, line_number, "four-corner polygon has no area")
        return class_id, points, area

    def _invalid_label(self, label_path: Path, line_number: int, reason: str):
        message = f"Invalid label in {label_path}:{line_number}: {reason}"
        if self._problem_sink is not None:
            self._problem_sink.append(message)
            return None
        if self.strict:
            raise ValueError(message)
        # Silently dropping objects looks like a clean run on broken data, so say
        # so at least once per worker.
        self._skipped_labels += 1
        if self._skipped_labels <= self.max_reported_problems:
            warnings.warn(f"{message}; skipping this object", UserWarning, stacklevel=2)
        elif self._skipped_labels == self.max_reported_problems + 1:
            warnings.warn(
                f"More invalid labels under {self.img_folder or self.root}; further reports suppressed",
                UserWarning,
                stacklevel=2,
            )
        return None

    @staticmethod
    def _polygon_area(points) -> float:
        area = 0.0
        for index, (x1, y1) in enumerate(points):
            x2, y2 = points[(index + 1) % len(points)]
            area += x1 * y2 - x2 * y1
        return abs(area) / 2.0
