"""
YOLO-format detection dataset support for EdgeCrafter.
"""

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image, ImageDraw
from pycocotools.coco import COCO
import yaml

from ...core import register
from .._misc import convert_to_tv_tensor
from ._dataset import DetDataset

Image.MAX_IMAGE_PIXELS = None

__all__ = ["YOLODetection"]


def _as_path(path: Optional[str]) -> Optional[Path]:
    if path is None:
        return None
    return Path(path).expanduser()


def _resolve_path(path: str, root: Optional[Path]) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute() or root is None:
        return candidate
    return root / candidate


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
        return [str(i) for i in range(int(num_classes or 0))]

    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names, key=lambda x: int(x))]

    if isinstance(names, (list, tuple)):
        return [str(name) for name in names]

    raise TypeError(f"Unsupported YOLO names format: {type(names)!r}")


@register()
class YOLODetection(DetDataset):
    __inject__ = ["transforms"]
    __share__ = ["num_classes", "yolo_root", "yolo_data_file"]

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
        return_masks: bool = False,
        names: Any = None,
        num_classes: Optional[int] = None,
        image_extensions: Optional[Sequence[str]] = None,
        recursive: bool = True,
        normalized: bool = True,
        bbox_to_mask: bool = False,
        strict: bool = True,
        validate_on_init: bool = True,
        **kwargs,
    ):
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
        # Treat the explicitly selected dataset directory as authoritative. YOLO
        # data files are often copied together with a dataset and retain a stale
        # `path` entry pointing to the source location.
        self.root = root_path or data_dir

        if names is None and data_cfg:
            names = data_cfg.get("names")
        if num_classes is None and data_cfg:
            num_classes = data_cfg.get("nc")

        if split is not None:
            split_entry = data_cfg.get(split, f"images/{split}") if data_cfg else f"images/{split}"
            img_folder = str(_resolve_path(split_entry, self.root))
        elif img_folder is None:
            raise ValueError("YOLODetection requires img_folder or split.")

        self.img_folder = _as_path(img_folder)
        if self.img_folder is None:
            raise ValueError("YOLODetection img_folder was not resolved.")

        if label_folder is None:
            label_folder = str(self._infer_label_folder(self.img_folder, self.root))
        self.label_folder = _as_path(label_folder)

        self._transforms = transforms
        self.return_masks = return_masks
        self.normalized = normalized
        self.bbox_to_mask = bbox_to_mask
        self.strict = strict
        self.validate_on_init = validate_on_init

        self.names = _normalize_names(names, num_classes)
        self.num_classes = int(num_classes if num_classes is not None else len(self.names))

        extensions = image_extensions or [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
        self.image_extensions = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions}
        iterator = self.img_folder.rglob("*") if recursive else self.img_folder.glob("*")
        self.image_files = sorted(path for path in iterator if path.is_file() and path.suffix.lower() in self.image_extensions)

        if not self.image_files:
            raise FileNotFoundError(f"No images found in {self.img_folder}")

        if self.validate_on_init:
            self._validate_dataset()

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx):
        img, target = self.load_item(idx)
        if self._transforms is not None:
            self._transforms.set_epoch(self.epoch)
            img, target = self._transforms(img, target)
        return img, target

    def load_item(self, idx):
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert("RGB")
        w, h = image.size

        label_path = self._label_path(image_path)
        boxes, labels, masks, _ = self._parse_label_file(label_path, w, h, need_masks=self.return_masks)

        target = {
            "boxes": convert_to_tv_tensor(boxes, key="boxes", spatial_size=image.size[::-1]),
            "labels": labels,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": self._box_area(boxes),
            "iscrowd": torch.zeros((labels.shape[0],), dtype=torch.int64),
            "orig_size": torch.as_tensor([int(w), int(h)]),
            "idx": torch.tensor([idx], dtype=torch.int64),
        }

        if self.return_masks:
            target["masks"] = convert_to_tv_tensor(masks.bool(), key="masks")

        return image, target

    def extra_repr(self) -> str:
        return (
            f" img_folder: {self.img_folder}\n"
            f" label_folder: {self.label_folder}\n"
            f" images: {len(self.image_files)}\n"
            f" return_masks: {self.return_masks}\n"
            f" num_classes: {self.num_classes}\n"
        )

    @property
    def categories(self) -> List[Dict[str, Any]]:
        if self.names:
            return [{"id": i, "name": name} for i, name in enumerate(self.names)]
        return [{"id": i, "name": str(i)} for i in range(self.num_classes)]

    @property
    def category2name(self) -> Dict[int, str]:
        return {cat["id"]: cat["name"] for cat in self.categories}

    @property
    def category2label(self) -> Dict[int, int]:
        return {cat["id"]: i for i, cat in enumerate(self.categories)}

    @property
    def label2category(self) -> Dict[int, int]:
        return {i: cat["id"] for i, cat in enumerate(self.categories)}

    def get_coco_api(self) -> COCO:
        coco_ds = COCO()
        dataset = {"images": [], "categories": self.categories, "annotations": []}
        ann_id = 1

        for idx, image_path in enumerate(self.image_files):
            with Image.open(image_path) as image:
                w, h = image.size

            dataset["images"].append({
                "id": idx,
                "file_name": image_path.name,
                "width": w,
                "height": h,
            })

            label_path = self._label_path(image_path)
            boxes, labels, _, segments = self._parse_label_file(
                label_path,
                w,
                h,
                need_masks=False,
                need_segments=self.return_masks,
            )

            xywh = boxes.clone()
            xywh[:, 2:] -= xywh[:, :2]
            areas = self._box_area(boxes)

            for obj_idx in range(labels.shape[0]):
                ann = {
                    "id": ann_id,
                    "image_id": idx,
                    "category_id": int(labels[obj_idx].item()),
                    "bbox": xywh[obj_idx].tolist(),
                    "area": float(areas[obj_idx].item()),
                    "iscrowd": 0,
                }
                if self.return_masks:
                    ann["segmentation"] = segments[obj_idx]
                dataset["annotations"].append(ann)
                ann_id += 1

        coco_ds.dataset = dataset
        coco_ds.createIndex()
        return coco_ds

    @staticmethod
    def _load_data_cfg(data_file: Optional[Path]) -> Dict[str, Any]:
        if data_file is None:
            return {}
        with data_file.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    @staticmethod
    def _infer_label_folder(img_folder: Path, root: Optional[Path]) -> Path:
        if root is not None:
            try:
                rel = img_folder.relative_to(root)
                if rel.parts and rel.parts[0] == "images":
                    return root.joinpath("labels", *rel.parts[1:])
            except ValueError:
                pass

        parts = list(img_folder.parts)
        if "images" in parts:
            idx = parts.index("images")
            parts[idx] = "labels"
            return Path(*parts)

        return img_folder.parent / "labels"

    def _label_path(self, image_path: Path) -> Path:
        rel_path = image_path.relative_to(self.img_folder)
        return self.label_folder / rel_path.with_suffix(".txt")

    def _parse_label_file(
        self,
        label_path: Path,
        width: int,
        height: int,
        need_masks: bool = False,
        need_segments: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[List[List[float]]]]:
        boxes: List[List[float]] = []
        labels: List[int] = []
        masks: List[torch.Tensor] = []
        segments: List[List[List[float]]] = []

        if not label_path.exists():
            raise FileNotFoundError(f"Missing YOLO label file: {label_path}")

        with label_path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                parsed = self._parse_line(line, width, height, label_path, line_number)
                if parsed is None:
                    continue

                class_id, box, polygon = parsed
                boxes.append(box)
                labels.append(class_id)

                if need_masks or need_segments:
                    segment = polygon or self._box_to_polygon(box)
                    if polygon is None and not self.bbox_to_mask and need_masks:
                        raise ValueError(
                            "YOLODetection return_masks=True requires YOLO-seg polygons. "
                            "Set bbox_to_mask=True only if rectangular masks are acceptable."
                        )
                    if need_segments:
                        segments.append([segment])
                    if need_masks:
                        masks.append(self._polygon_to_mask(segment, width, height))

        if not boxes:
            return self._empty_targets(height, width)

        boxes_t = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels_t = torch.as_tensor(labels, dtype=torch.int64)
        keep = (boxes_t[:, 2] > boxes_t[:, 0]) & (boxes_t[:, 3] > boxes_t[:, 1])

        if self.num_classes > 0:
            keep = keep & (labels_t >= 0) & (labels_t < self.num_classes)

        boxes_t = boxes_t[keep]
        labels_t = labels_t[keep]

        if need_masks:
            keep_list = keep.tolist()
            kept_masks = [mask for mask, is_kept in zip(masks, keep_list) if is_kept]
            masks_t = torch.stack(kept_masks, dim=0) if kept_masks else torch.zeros((0, height, width), dtype=torch.uint8)
        else:
            masks_t = torch.zeros((0, height, width), dtype=torch.uint8)

        if need_segments:
            keep_list = keep.tolist()
            segments = [segment for segment, is_kept in zip(segments, keep_list) if is_kept]

        return boxes_t, labels_t, masks_t, segments

    def _validate_dataset(self) -> None:
        """Validate every image/label pair before a dataloader starts yielding batches."""
        for image_path in self.image_files:
            if self.normalized:
                width, height = 1, 1
            else:
                with Image.open(image_path) as image:
                    width, height = image.size

            self._validate_label_file(self._label_path(image_path), width, height)

    def _validate_label_file(self, label_path: Path, width: int, height: int) -> None:
        if not label_path.exists():
            raise FileNotFoundError(f"Missing YOLO label file: {label_path}")

        with label_path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                parsed = self._parse_line(line, width, height, label_path, line_number)
                if parsed is None:
                    continue

                _, _, polygon = parsed
                if self.return_masks and polygon is None and not self.bbox_to_mask:
                    raise ValueError(
                        f"Invalid label in {label_path}:{line_number}: detection bbox cannot be used as a mask"
                    )

    def _parse_line(
        self,
        line: str,
        width: int,
        height: int,
        label_path: Path,
        line_number: int,
    ) -> Optional[Tuple[int, List[float], Optional[List[float]]]]:
        parts = line.split()
        if len(parts) < 5:
            return self._invalid_label(label_path, line_number, "expected at least 5 fields")

        try:
            class_value = float(parts[0])
            coords = [float(v) for v in parts[1:]]
        except ValueError:
            return self._invalid_label(label_path, line_number, "contains non-numeric values")

        if not math.isfinite(class_value) or not class_value.is_integer():
            return self._invalid_label(label_path, line_number, "class id must be a finite integer")
        if not all(math.isfinite(value) for value in coords):
            return self._invalid_label(label_path, line_number, "contains non-finite coordinates")

        class_id = int(class_value)
        if class_id < 0:
            return self._invalid_label(label_path, line_number, "class id must be non-negative")
        if self.num_classes > 0 and class_id >= self.num_classes:
            return self._invalid_label(
                label_path,
                line_number,
                f"class id {class_id} is outside [0, {self.num_classes - 1}]",
            )

        if self.normalized and any(value < 0.0 or value > 1.0 for value in coords):
            return self._invalid_label(label_path, line_number, "normalized coordinates must be within [0, 1]")

        if len(coords) == 4:
            if coords[2] <= 0.0 or coords[3] <= 0.0:
                return self._invalid_label(label_path, line_number, "bbox width and height must be positive")
            box = self._yolo_box_to_xyxy(coords, width, height)
            if box[2] <= box[0] or box[3] <= box[1]:
                return self._invalid_label(label_path, line_number, "bbox has no area inside the image")
            return class_id, box, None

        if len(coords) >= 6 and len(coords) % 2 == 0:
            points = set(zip(coords[0::2], coords[1::2]))
            if len(points) < 3:
                return self._invalid_label(label_path, line_number, "segmentation polygon needs 3 unique points")
            if self._polygon_area(coords) <= 0.0:
                return self._invalid_label(label_path, line_number, "segmentation polygon has no area")

            polygon = self._polygon_to_pixels(coords, width, height)
            xs = polygon[0::2]
            ys = polygon[1::2]
            box = [min(xs), min(ys), max(xs), max(ys)]
            box = self._clip_box(box, width, height)
            if box[2] <= box[0] or box[3] <= box[1]:
                return self._invalid_label(label_path, line_number, "segmentation polygon has no area")
            return class_id, box, polygon

        return self._invalid_label(label_path, line_number, "invalid YOLO segmentation polygon")

    def _invalid_label(self, label_path: Path, line_number: int, reason: str):
        message = f"Invalid label in {label_path}:{line_number}: {reason}"
        if self.strict:
            raise ValueError(message)
        return None

    def _scale_xy(self, x: float, y: float, width: int, height: int) -> Tuple[float, float]:
        if self.normalized:
            return x * width, y * height
        return x, y

    def _yolo_box_to_xyxy(self, coords: Sequence[float], width: int, height: int) -> List[float]:
        cx, cy = self._scale_xy(coords[0], coords[1], width, height)
        bw = coords[2] * width if self.normalized else coords[2]
        bh = coords[3] * height if self.normalized else coords[3]
        return self._clip_box([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], width, height)

    def _polygon_to_pixels(self, coords: Sequence[float], width: int, height: int) -> List[float]:
        polygon: List[float] = []
        for x, y in zip(coords[0::2], coords[1::2]):
            px, py = self._scale_xy(x, y, width, height)
            polygon.extend([min(max(px, 0.0), float(width)), min(max(py, 0.0), float(height))])
        return polygon

    @staticmethod
    def _polygon_area(coords: Sequence[float]) -> float:
        points = list(zip(coords[0::2], coords[1::2]))
        area = 0.0
        for index, (x1, y1) in enumerate(points):
            x2, y2 = points[(index + 1) % len(points)]
            area += x1 * y2 - x2 * y1
        return abs(area) / 2.0

    @staticmethod
    def _clip_box(box: Sequence[float], width: int, height: int) -> List[float]:
        x1, y1, x2, y2 = box
        return [
            min(max(x1, 0.0), float(width)),
            min(max(y1, 0.0), float(height)),
            min(max(x2, 0.0), float(width)),
            min(max(y2, 0.0), float(height)),
        ]

    @staticmethod
    def _box_to_polygon(box: Sequence[float]) -> List[float]:
        x1, y1, x2, y2 = box
        return [x1, y1, x2, y1, x2, y2, x1, y2]

    @staticmethod
    def _polygon_to_mask(polygon: Sequence[float], width: int, height: int) -> torch.Tensor:
        mask = Image.new("L", (width, height), 0)
        points = list(zip(polygon[0::2], polygon[1::2]))
        ImageDraw.Draw(mask).polygon(points, outline=1, fill=1)
        return torch.as_tensor(torch.ByteTensor(torch.ByteStorage.from_buffer(mask.tobytes())).reshape(height, width))

    @staticmethod
    def _box_area(boxes: torch.Tensor) -> torch.Tensor:
        if boxes.numel() == 0:
            return torch.zeros((0,), dtype=torch.float32)
        return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    @staticmethod
    def _empty_targets(height: int, width: int):
        return (
            torch.zeros((0, 4), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.int64),
            torch.zeros((0, height, width), dtype=torch.uint8),
            [],
        )
