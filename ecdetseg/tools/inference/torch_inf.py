"""
EdgeCrafter: Compact ViTs for Edge Dense Prediction via Task-Specialized Distillation
Copyright (c) 2026 The EdgeCrafter Authors. All Rights Reserved.
"""

import concurrent.futures
import os
import queue
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import yaml
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from engine.core import YAMLConfig
from engine.core.yaml_utils import merge_dict, parse_cli
from engine.data.dataset.coco_dataset import mscoco_label2name_remap80


@dataclass
class Result:
    label: int
    score: float
    box: np.ndarray
    mask: np.ndarray = None


COCO_COLORS = [
    (0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (255, 0, 255),
    (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
    (128, 0, 128), (0, 128, 128), (255, 128, 0), (255, 0, 128), (0, 255, 128),
    (128, 255, 0), (255, 128, 128), (128, 255, 128), (128, 128, 255), (255, 255, 128),
    (255, 128, 255), (128, 255, 255), (192, 0, 0), (0, 192, 0), (0, 0, 192),
    (192, 192, 0), (192, 0, 192), (0, 192, 192), (255, 192, 0), (255, 0, 192),
    (0, 255, 192), (192, 255, 0), (255, 192, 128), (192, 255, 128), (128, 192, 255),
    (255, 128, 192), (128, 255, 192), (192, 128, 255), (255, 192, 192), (192, 255, 192),
    (192, 192, 255), (255, 255, 192), (255, 192, 255), (192, 255, 255), (64, 0, 0),
    (0, 64, 0), (0, 0, 64), (64, 64, 0), (64, 0, 64), (0, 64, 64), (128, 64, 0),
    (128, 0, 64), (0, 128, 64), (64, 128, 0), (128, 64, 128), (64, 128, 128), (128, 128, 64),
    (192, 64, 0), (192, 0, 64), (0, 192, 64), (64, 192, 0), (192, 64, 192), (64, 192, 192),
    (192, 192, 64), (255, 64, 0), (255, 0, 64), (0, 255, 64), (64, 255, 0),
    (255, 64, 128), (64, 255, 128), (128, 64, 255), (255, 128, 64), (128, 255, 64),
    (64, 128, 255), (192, 64, 128), (192, 128, 64), (64, 192, 128), (128, 192, 64),
    (64, 128, 192), (128, 64, 192), (192, 128, 192), (128, 192, 192), (192, 192, 128),
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_BOUNDARY_KERNEL_CACHE: dict[int, np.ndarray] = {}
_CLASS_NAMES = mscoco_label2name_remap80


_RESOLUTION_DEPENDENT_STATE_KEYS = {
    "decoder.anchors",
    "decoder.valid_mask",
}


def _parse_input_size(input_size):
    if input_size is None:
        return None
    if len(input_size) == 1:
        h = w = int(input_size[0])
    elif len(input_size) == 2:
        h, w = (int(v) for v in input_size)
    else:
        raise ValueError("--input-size expects one value S or two values H W.")
    if h <= 0 or w <= 0:
        raise ValueError(f"--input-size must be positive, got {h} {w}.")
    if h != w:
        raise ValueError("Only square inference resolutions are supported for now.")
    if h % 32 != 0 or w % 32 != 0:
        raise ValueError(f"--input-size must be divisible by 32, got {h} {w}.")
    return [h, w]


def _find_yolo_data_file(data: str) -> Path:
    path = Path(data).expanduser()
    if path.is_file():
        return path
    for name in ("data.yaml", "data.yml", "data_win.yaml", "dataset.yaml", "dataset.yml"):
        candidate = path / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Can not find YOLO data.yaml under: {path}")


def _resolve_yolo_data(data: str):
    data_arg = Path(data).expanduser()
    data_file = _find_yolo_data_file(data)
    with data_file.open("r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f) or {}
    root = data_arg if data_arg.is_dir() else data_file.parent
    return root, data_file, data_cfg


def _class_names_from_yolo(data_cfg: dict):
    names = data_cfg.get("names", None)
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, (list, tuple)):
        return {i: str(v) for i, v in enumerate(names)}
    return None


def _class_name(label: int) -> str:
    return _CLASS_NAMES.get(label, str(label))


def get_class_color(label: int):
    return COCO_COLORS[label % len(COCO_COLORS)]


def get_draw_params(image_shape: tuple[int, int, int]):
    height, width = image_shape[:2]
    min_side = max(1, min(height, width))
    base = min_side / 640.0
    font_scale = max(0.9, 0.95 * base)
    text_thickness = max(2, int(round(1.6 * base)))
    box_thickness = max(2, int(round(2.0 * base)))
    boundary_thickness = max(2, int(round(2.0 * base)))
    return font_scale, text_thickness, box_thickness, boundary_thickness


def draw_white_boundary_fast(image: np.ndarray, mask: np.ndarray, thickness: int = 2):
    mask_u8 = mask.astype(np.uint8)
    if not np.any(mask_u8):
        return

    k = max(1, int(thickness))
    kernel = _BOUNDARY_KERNEL_CACHE.get(k)
    if kernel is None:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
        _BOUNDARY_KERNEL_CACHE[k] = kernel
    edge = cv2.morphologyEx(mask_u8, cv2.MORPH_GRADIENT, kernel)
    image[edge > 0] = (255, 255, 255)


def draw_to_numpy(image: Image.Image, results: list[Result], alpha: float = 0.5):
    im_np = np.array(image, copy=True)
    font_scale, text_thickness, box_thickness, boundary_thickness = get_draw_params(im_np.shape)
    font = cv2.FONT_HERSHEY_SIMPLEX

    if results and alpha > 0:
        overlay = im_np.copy()
        any_mask = False
        merged_mask = np.zeros(im_np.shape[:2], dtype=np.uint8)
        count = 0
        for res in results:
            count += 1
            if res.mask is None:
                continue
            mask_bool = res.mask.astype(bool, copy=False)
            if not np.any(mask_bool):
                continue
            any_mask = True
            color_rgb = get_class_color(res.label)
            overlay[mask_bool] = color_rgb
            merged_mask[mask_bool] = 1

        if any_mask:
            im_np = cv2.addWeighted(im_np, 1.0 - alpha, overlay, alpha, 0)
            draw_white_boundary_fast(im_np, merged_mask, thickness=boundary_thickness)

    for res in results:
        color_rgb = get_class_color(res.label)
        x1, y1, x2, y2 = res.box.astype(int)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(im_np.shape[1] - 1, x2)
        y2 = min(im_np.shape[0] - 1, y2)

        cv2.rectangle(im_np, (x1, y1), (x2, y2), color_rgb, box_thickness)
        text = f"{_class_name(res.label)} {res.score:.2f}"
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, text_thickness)

        text_x = x1
        text_y = max(th + baseline + 2, y1)
        cv2.rectangle(
            im_np,
            (text_x, text_y - th - baseline - 4),
            (text_x + tw + 4, text_y + 2),
            color_rgb,
            -1,
        )
        cv2.putText(
            im_np,
            text,
            (text_x + 2, text_y - baseline - 1),
            font,
            font_scale,
            (255, 255, 255),
            text_thickness,
            cv2.LINE_AA,
        )

    return im_np


def draw(image: Image.Image, results: list[Result], alpha: float = 0.4):
    return Image.fromarray(draw_to_numpy(image, results, alpha=alpha).astype(np.uint8))


class ECInferencer:
    def __init__(self, model, task, device, size, thresh):
        self.model = model
        self.task = task
        self.device = device
        self.size = size
        self.thresh = thresh
        self.transforms = self._build_transforms()

    def _build_transforms(self):
        return T.Compose([
            T.Resize(self.size),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    @torch.no_grad()
    def infer(self, image: Image.Image):
        orig_sizes = torch.tensor([[image.size[0], image.size[1]]], device=self.device)
        tensor = self.transforms(image).unsqueeze(0).to(self.device)
        outputs = self.model(tensor, orig_sizes)

        if self.task == "segmentation":
            labels, boxes, scores, masks = outputs
        elif self.task == "detection":
            labels, boxes, scores = outputs
            masks = None
        else:
            raise ValueError(f"Unsupported task: {self.task}")

        keep = scores[0] > self.thresh
        lbls = labels[0][keep]
        bxs = boxes[0][keep]
        scs = scores[0][keep]

        results = []
        if masks is not None:
            img_h, img_w = image.size[1], image.size[0]
            msks = masks[0].unsqueeze(0)
            msks = torch.nn.functional.interpolate(
                msks,
                size=(img_h, img_w),
                mode="bilinear",
                align_corners=False,
            )[0]
            msks = msks > 0.0
            msks = msks[keep]

            for j in range(len(lbls)):
                results.append(
                    Result(
                        label=int(lbls[j].item()),
                        score=float(scs[j].item()),
                        box=bxs[j].cpu().numpy(),
                        mask=msks[j].cpu().numpy(),
                    )
                )
        else:
            for j in range(len(lbls)):
                results.append(
                    Result(
                        label=int(lbls[j].item()),
                        score=float(scs[j].item()),
                        box=bxs[j].cpu().numpy(),
                    )
                )
        return results


class VideoReader(threading.Thread):
    def __init__(self, cap, queue_size=32):
        super().__init__()
        self.cap = cap
        self.q = queue.Queue(maxsize=queue_size)
        self.stopped = False
        self.daemon = True

    def run(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                self.q.put(None)
                break
            self.q.put(frame)

    def read(self):
        return self.q.get()

    def stop(self):
        self.stopped = True
        while not self.q.empty():
            self.q.get()


def process_image(inferencer: ECInferencer, path: Path):
    image = Image.open(path).convert("RGB")
    results = inferencer.infer(image)
    image = draw(image, results)

    output_path = path.with_stem(f"{path.stem}_torch_inference")
    image.save(output_path, quality=95, subsampling=0)
    print(f"Saved result to {output_path}")
    print(f"Detected {len(results)} instances")


def process_image_dir(inferencer: ECInferencer, dir_path: Path):
    image_paths = sorted([p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
    if not image_paths:
        raise ValueError(f"No image files found in directory: {dir_path}")

    print(f"Found {len(image_paths)} images in {dir_path}")
    for idx, img_path in enumerate(image_paths, start=1):
        print(f"[{idx}/{len(image_paths)}] Processing {img_path.name}")
        process_image(inferencer, img_path)


def process_video(inferencer: ECInferencer, path: Path, num_workers: int = 4):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")

    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = path.with_stem(f"{path.stem}_torch_inference")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not out.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to create output video writer: {output_path}")

    reader = VideoReader(cap)
    reader.start()

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_workers)
    futures_queue = queue.Queue(maxsize=max(32, num_workers * 4))

    def process_and_draw(pil_img: Image.Image, results: list[Result]):
        res_np = draw_to_numpy(pil_img, results)
        return cv2.cvtColor(res_np, cv2.COLOR_RGB2BGR)

    frame_count = 0
    writer_error: dict[str, Exception | None] = {"exc": None}

    def writer_worker():
        nonlocal frame_count
        try:
            while True:
                future = futures_queue.get()
                if future is None:
                    break
                frame_out = future.result()
                out.write(frame_out)
                frame_count += 1
                if frame_count % 10 == 0:
                    print(f"Processed {frame_count}/{total_frame} frames")
        except Exception as exc:  # noqa: BLE001
            writer_error["exc"] = exc

    writer_thread = threading.Thread(target=writer_worker, daemon=True)
    writer_thread.start()

    try:
        while True:
            if writer_error["exc"] is not None:
                raise writer_error["exc"]

            frame = reader.read()
            if frame is None:
                break

            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            results = inferencer.infer(pil_img)
            future = executor.submit(process_and_draw, pil_img, results)
            futures_queue.put(future)
    finally:
        reader.stop()
        futures_queue.put(None)
        writer_thread.join()
        executor.shutdown()
        cap.release()
        out.release()

    if writer_error["exc"] is not None:
        raise writer_error["exc"]

    print(f"Saved video result to {output_path}")


def _load_model_state(model: nn.Module, state):
    model_state = model.state_dict()
    load_state = dict(state)
    skipped_resolution_keys = []
    for key in _RESOLUTION_DEPENDENT_STATE_KEYS:
        if key in load_state and key in model_state and tuple(load_state[key].shape) != tuple(model_state[key].shape):
            skipped_resolution_keys.append((key, tuple(load_state[key].shape), tuple(model_state[key].shape)))
            load_state.pop(key)

    incompatible = model.load_state_dict(load_state, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    missing = [key for key in incompatible.missing_keys if key not in _RESOLUTION_DEPENDENT_STATE_KEYS]
    if unexpected or missing:
        raise RuntimeError(
            "Checkpoint loading failed after skipping resolution-dependent buffers:\n"
            f"  missing: {missing}\n"
            f"  unexpected: {unexpected}"
        )
    for key, src_shape, dst_shape in skipped_resolution_keys:
        print(f"  skip resolution buffer {key}: checkpoint{src_shape} -> model{dst_shape}")


def build_model(config_path: str, resume_path: str, device: torch.device, update=None):
    cfg_kwargs = merge_dict({"resume": resume_path}, update or {}, inplace=False)
    cfg = YAMLConfig(config_path, **cfg_kwargs)
    if "ViTAdapter" in cfg.yaml_cfg:
        cfg.yaml_cfg["ViTAdapter"]["skip_load_backbone"] = True

    checkpoint = torch.load(resume_path, map_location="cpu", weights_only=True)
    state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
    _load_model_state(cfg.model, state)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            return self.postprocessor(outputs, orig_target_sizes)

    model = Model().to(device)
    model.eval()

    return model, cfg.yaml_cfg["eval_spatial_size"], cfg.yaml_cfg["task"]


def main(args):
    global _CLASS_NAMES

    device = torch.device(args.device)
    update = parse_cli(args.update)

    input_size = _parse_input_size(args.input_size)
    if input_size is not None:
        update = merge_dict(update, {"eval_spatial_size": input_size})

    if args.data is not None:
        yolo_root, yolo_data_file, data_cfg = _resolve_yolo_data(args.data)
        update = merge_dict(update, {
            "yolo_root": str(yolo_root),
            "yolo_data_file": str(yolo_data_file),
        })
        class_names = _class_names_from_yolo(data_cfg)
        if class_names is not None:
            _CLASS_NAMES = class_names
        print(f"YOLO data file: {yolo_data_file}")
        print(f"YOLO dataset root: {yolo_root}")

    model, img_size, task = build_model(args.config, args.resume, device, update)
    print(f"Inference spatial size: {img_size[0]}x{img_size[1]}")

    inferencer = ECInferencer(
        model=model,
        task=task,
        device=device,
        size=img_size,
        thresh=args.thresh,
    )

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if input_path.is_dir():
        process_image_dir(inferencer, input_path)
    elif input_path.suffix.lower() in IMAGE_EXTS:
        process_image(inferencer, input_path)
    else:
        process_video(inferencer, input_path, num_workers=args.num_workers)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="EdgeCrafter Inference")
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-r", "--resume", required=True)
    parser.add_argument("-i", "--input", required=True, help="Image path, image directory path, or video path")
    parser.add_argument("--data", default=None, help="YOLO dataset root or data.yaml for custom class names/classes.")
    parser.add_argument("-u", "--update", nargs="*", default=[], help="YAML overrides, e.g. yolo_root=/data/ds.")
    parser.add_argument(
        "--input-size",
        nargs="+",
        type=int,
        default=None,
        help="Inference resolution. Use S for SxS, or H W. Currently square and divisible by 32.",
    )
    parser.add_argument(
        "-d",
        "--device",
        default="cuda:0",
        help="Device to run inference on (e.g., 'cuda:0' or 'cpu')",
    )
    parser.add_argument("-t", "--thresh", type=float, default=0.4)
    parser.add_argument("--num-workers", type=int, default=4, help="Thread workers for video draw/write")

    main(parser.parse_args())
