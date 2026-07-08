"""
Export EdgeCrafter checkpoints to TensorRT and evaluate mAP on the validation set.
"""

import argparse
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

import torch
import torch.nn as nn
import yaml

from engine.core import YAMLConfig
from engine.core.yaml_utils import merge_dict, parse_cli
from engine.data import CocoEvaluator, get_coco_api_from_dataset
from engine.misc import MetricLogger


def _parse_input_size(input_size: Optional[List[int]]) -> Optional[List[int]]:
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
        raise ValueError("Only square export/eval resolutions are supported for now.")
    if h % 32 != 0 or w % 32 != 0:
        raise ValueError(f"--input-size must be divisible by 32, got {h} {w}.")
    return [h, w]


def _find_data_file(root: Path) -> Path:
    if root.is_file():
        return root
    for name in ("data.yaml", "data.yml", "data_win.yaml", "dataset.yaml", "dataset.yml"):
        candidate = root / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Can not find YOLO data yaml in: {root}")


def _resolve_path(path: str, root: Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return root / candidate


def resolve_yolo_data(data: Optional[str]) -> Tuple[Optional[Path], Optional[Path], Optional[Path], Optional[dict]]:
    if data is None:
        return None, None, None, None

    data_arg = Path(data).expanduser()
    data_file = _find_data_file(data_arg)
    dataset_root = data_arg if data_arg.is_dir() else data_file.parent

    with data_file.open("r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f) or {}

    split = data_cfg.get("val", None)
    if split is None:
        raise ValueError(f"YOLO data file has no 'val' entry: {data_file}")
    if isinstance(split, (list, tuple)):
        raise NotImplementedError("Validation split as a list is not supported by this EdgeCrafter evaluator yet.")

    # Prefer the explicit --data root over a possibly stale `path:` inside data.yaml.
    val_path = _resolve_path(str(split), dataset_root)
    return dataset_root, data_file, val_path, data_cfg


def _extract_state_dict(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "ema" in checkpoint:
        return checkpoint["ema"]["module"]
    elif "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint


_RESOLUTION_DEPENDENT_STATE_KEYS = {
    "decoder.anchors",
    "decoder.valid_mask",
}


def _load_checkpoint_model(cfg: YAMLConfig, checkpoint_path: str, strict: bool = False) -> None:
    state = _extract_state_dict(checkpoint_path)
    model_state = cfg.model.state_dict()

    if strict:
        strict_state = dict(state)
        skipped_resolution_keys = []
        for key in _RESOLUTION_DEPENDENT_STATE_KEYS:
            if key in strict_state and key in model_state and tuple(strict_state[key].shape) != tuple(model_state[key].shape):
                skipped_resolution_keys.append((key, tuple(strict_state[key].shape), tuple(model_state[key].shape)))
                strict_state.pop(key)
        incompatible = cfg.model.load_state_dict(strict_state, strict=False)
        unexpected = list(incompatible.unexpected_keys)
        missing = [key for key in incompatible.missing_keys if key not in _RESOLUTION_DEPENDENT_STATE_KEYS]
        if unexpected or missing:
            raise RuntimeError(
                "Strict checkpoint loading failed after skipping resolution-dependent buffers:\n"
                f"  missing: {missing}\n"
                f"  unexpected: {unexpected}"
            )
        print(f"Loaded checkpoint strictly: {checkpoint_path}")
        for key, src_shape, dst_shape in skipped_resolution_keys:
            print(f"  skip resolution buffer {key}: checkpoint{src_shape} -> model{dst_shape}")
        return

    matched = {}
    skipped = []
    missing = []
    for key, tensor in model_state.items():
        if key not in state:
            missing.append(key)
            continue
        if tuple(tensor.shape) == tuple(state[key].shape):
            matched[key] = state[key]
        else:
            skipped.append((key, tuple(state[key].shape), tuple(tensor.shape)))

    cfg.model.load_state_dict(matched, strict=False)
    print(f"Loaded checkpoint with shape filtering: {checkpoint_path}")
    print(f"  matched: {len(matched)}")
    print(f"  missing in checkpoint: {len(missing)}")
    print(f"  skipped due to shape mismatch: {len(skipped)}")
    for key, src_shape, dst_shape in skipped[:20]:
        print(f"    skip {key}: checkpoint{src_shape} -> model{dst_shape}")
    if len(skipped) > 20:
        print(f"    ... {len(skipped) - 20} more skipped tensors")


def _box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([
        cx - 0.5 * w,
        cy - 0.5 * h,
        cx + 0.5 * w,
        cy + 0.5 * h,
    ], dim=-1)


def _batch_index_select(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    batch_size, length = x.shape[:2]
    flat_index = index + torch.arange(batch_size, device=index.device).unsqueeze(1) * length
    flat_x = x.reshape(batch_size * length, *x.shape[2:])
    selected = flat_x.index_select(0, flat_index.reshape(-1))
    return selected.reshape(batch_size, index.shape[1], *x.shape[2:])


def _topk_detections(
    logits: torch.Tensor,
    boxes: torch.Tensor,
    num_classes: int,
    num_top_queries: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scores = torch.sigmoid(logits)
    scores, flat_index = torch.topk(scores.flatten(1), num_top_queries, dim=-1)
    labels = (flat_index - flat_index // num_classes * num_classes).to(torch.int32)
    box_index = flat_index // num_classes
    boxes = _batch_index_select(_box_cxcywh_to_xyxy(boxes), box_index)
    return labels, boxes, scores


class DeployModel(nn.Module):
    def __init__(self, cfg: YAMLConfig, export_mode: str) -> None:
        super().__init__()
        self.model = cfg.model.deploy()
        self.export_mode = export_mode
        self.num_classes = int(cfg.yaml_cfg["num_classes"])
        self.num_top_queries = int(cfg.postprocessor.num_top_queries)
        if export_mode == "pixel":
            self.postprocessor = cfg.postprocessor.deploy()

    def forward(self, images: torch.Tensor, orig_target_sizes: Optional[torch.Tensor] = None):
        outputs = self.model(images)
        if self.export_mode == "raw":
            return outputs["pred_logits"], outputs["pred_boxes"]

        if self.export_mode == "normalized":
            return _topk_detections(
                outputs["pred_logits"],
                outputs["pred_boxes"],
                self.num_classes,
                self.num_top_queries,
            )

        postprocessed = self.postprocessor(outputs, orig_target_sizes)
        if len(postprocessed) == 4:
            labels, boxes, scores, masks = postprocessed
            return labels.to(torch.int32), boxes, scores, masks
        labels, boxes, scores = postprocessed
        return labels.to(torch.int32), boxes, scores


def export_onnx(
    cfg: YAMLConfig,
    checkpoint: str,
    output_file: Path,
    batch_size: int,
    opset: int,
    static_batch: bool,
    check: bool,
    simplify: bool,
    strict_load: bool,
    export_mode: str,
) -> Path:
    if "ViTAdapter" in cfg.yaml_cfg:
        cfg.yaml_cfg["ViTAdapter"]["skip_load_backbone"] = True
    _load_checkpoint_model(cfg, checkpoint, strict=strict_load)

    model = DeployModel(cfg, export_mode).eval()
    img_h, img_w = cfg.yaml_cfg["eval_spatial_size"]
    images = torch.rand(batch_size, 3, img_h, img_w)
    orig_target_sizes = torch.tensor([[img_w, img_h]] * batch_size, dtype=torch.float32)
    task = cfg.yaml_cfg["task"]

    if task == "segmentation" and export_mode != "pixel":
        raise NotImplementedError(f"--export-mode {export_mode} is only implemented for detection.")

    if export_mode == "raw":
        input_args = (images,)
        input_names = ["images"]
        output_names = ["pred_logits", "pred_boxes"]
    else:
        output_names = ["labels", "boxes", "scores"] + (["masks"] if task == "segmentation" else [])
        if export_mode == "pixel":
            input_args = (images, orig_target_sizes)
            input_names = ["images", "orig_target_sizes"]
        else:
            input_args = (images,)
            input_names = ["images"]

    dynamic_axes = None
    if not static_batch:
        dynamic_axes = {"images": {0: "N"}}
        if "orig_target_sizes" in input_names:
            dynamic_axes["orig_target_sizes"] = {0: "N"}
        dynamic_axes.update({name: {0: "N"} for name in output_names})

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        _ = model(*input_args)
        torch.onnx.export(
            model,
            input_args,
            str(output_file),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            verbose=False,
            do_constant_folding=True,
            external_data=False,
            dynamo=False,
        )

    if check:
        import onnx

        onnx_model = onnx.load(str(output_file))
        onnx.checker.check_model(onnx_model)
        print(f"ONNX check passed: {output_file}")

    if simplify:
        import onnx
        import onnxsim

        input_shapes = {"images": images.shape}
        if "orig_target_sizes" in input_names:
            input_shapes["orig_target_sizes"] = orig_target_sizes.shape
        onnx_model_simplify, ok = onnxsim.simplify(str(output_file), test_input_shapes=input_shapes)
        onnx.save(onnx_model_simplify, str(output_file), save_as_external_data=False)
        print(f"ONNX simplify: {ok}")

    return output_file


def _trt_logger(trt, verbose: bool):
    return trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.INFO)


def _trt_major(trt) -> int:
    return int(str(trt.__version__).split(".", 1)[0])


def _set_workspace(config, trt, workspace_gb: Optional[float]) -> None:
    if not workspace_gb:
        return
    workspace_bytes = int(float(workspace_gb) * (1 << 30))
    if hasattr(trt, "MemoryPoolType"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    else:
        config.max_workspace_size = workspace_bytes


def _set_profiling_verbosity(config, trt, verbosity: str) -> None:
    if not hasattr(config, "profiling_verbosity") or not hasattr(trt, "ProfilingVerbosity"):
        return
    mapping = {
        "none": getattr(trt.ProfilingVerbosity, "NONE", None),
        "layer_names_only": getattr(trt.ProfilingVerbosity, "LAYER_NAMES_ONLY", None),
        "detailed": getattr(trt.ProfilingVerbosity, "DETAILED", None),
    }
    value = mapping.get(verbosity)
    if value is not None:
        config.profiling_verbosity = value


def _precision_supported(builder, trt, precision: str) -> bool:
    if precision == "fp32":
        return True
    if precision == "fp16":
        return bool(getattr(builder, "platform_has_fast_fp16", True))
    if precision == "fp8":
        if not hasattr(trt.BuilderFlag, "FP8"):
            return False
        fast_fp8 = getattr(builder, "platform_has_fast_fp8", None)
        return True if fast_fp8 is None else bool(fast_fp8)
    raise ValueError(f"Unsupported precision: {precision}")


def _set_profile(
    builder,
    config,
    network,
    min_batch: int,
    opt_batch: int,
    max_batch: int,
    image_hw: Tuple[int, int],
) -> None:
    profile = builder.create_optimization_profile()
    has_dynamic_input = False
    img_h, img_w = image_hw

    for i in range(network.num_inputs):
        inp = network.get_input(i)
        shape = tuple(inp.shape)
        if all(dim > 0 for dim in shape):
            continue
        has_dynamic_input = True

        if inp.name == "images":
            min_shape = (min_batch, 3, img_h, img_w)
            opt_shape = (opt_batch, 3, img_h, img_w)
            max_shape = (max_batch, 3, img_h, img_w)
        elif inp.name == "orig_target_sizes":
            min_shape = (min_batch, 2)
            opt_shape = (opt_batch, 2)
            max_shape = (max_batch, 2)
        else:
            raise ValueError(f"Dynamic TensorRT input is not handled: {inp.name} shape={shape}")

        profile.set_shape(inp.name, min=min_shape, opt=opt_shape, max=max_shape)

    if has_dynamic_input:
        config.add_optimization_profile(profile)


def _read_onnx_image_hw(onnx_file: Path) -> Optional[Tuple[int, int]]:
    try:
        import onnx
    except ImportError:
        return None
    model = onnx.load(str(onnx_file), load_external_data=False)
    for inp in model.graph.input:
        if inp.name != "images":
            continue
        dims = inp.type.tensor_type.shape.dim
        if len(dims) < 4:
            return None
        h = dims[2].dim_value
        w = dims[3].dim_value
        if h > 0 and w > 0:
            return int(h), int(w)
        return None
    return None


def build_engine(
    onnx_file: Path,
    output_file: Path,
    precision: str,
    min_batch: int,
    opt_batch: int,
    max_batch: int,
    image_hw: Tuple[int, int],
    workspace_gb: Optional[float],
    verbose: bool,
    profiling_verbosity: str,
) -> Optional[Path]:
    import tensorrt as trt

    logger = _trt_logger(trt, verbose)
    trt.init_libnvinfer_plugins(logger, "")
    builder = trt.Builder(logger)

    if not _precision_supported(builder, trt, precision):
        print(f"Skip {precision.upper()}: TensorRT/platform does not report support.")
        return None

    config = builder.create_builder_config()
    _set_workspace(config, trt, workspace_gb)
    _set_profiling_verbosity(config, trt, profiling_verbosity)

    explicit_batch = 0 if _trt_major(trt) >= 10 else (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    network = builder.create_network(explicit_batch)
    parser = trt.OnnxParser(network, logger)

    if not parser.parse_from_file(str(onnx_file)):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"Failed to parse ONNX file {onnx_file}:\n{errors}")

    print(f"TensorRT {trt.__version__}: building {precision.upper()} engine")
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        print(f'  input  {inp.name}: shape={tuple(inp.shape)} dtype={inp.dtype}')
    for i in range(network.num_outputs):
        out = network.get_output(i)
        print(f'  output {out.name}: shape={tuple(out.shape)} dtype={out.dtype}')

    if precision == "fp32":
        if hasattr(trt.BuilderFlag, "TF32") and hasattr(config, "clear_flag"):
            config.clear_flag(trt.BuilderFlag.TF32)
    elif precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "fp8":
        config.set_flag(trt.BuilderFlag.FP8)
        if hasattr(trt.BuilderFlag, "FP16"):
            config.set_flag(trt.BuilderFlag.FP16)

    _set_profile(builder, config, network, min_batch, opt_batch, max_batch, image_hw)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if _trt_major(trt) >= 10:
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError(f"TensorRT {precision.upper()} engine build failed.")
        output_file.write_bytes(serialized)
    else:
        engine = builder.build_engine(network, config)
        if engine is None:
            raise RuntimeError(f"TensorRT {precision.upper()} engine build failed.")
        output_file.write_bytes(engine.serialize())

    print(f"Saved {precision.upper()} engine: {output_file}")
    return output_file


def dump_engine_inspector(engine_path: Path, output_path: Optional[Path], verbose: bool) -> Path:
    import tensorrt as trt

    logger = _trt_logger(trt, verbose)
    trt.init_libnvinfer_plugins(logger, "")
    with engine_path.open("rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError(f"Failed to deserialize TensorRT engine for inspector: {engine_path}")
    if not hasattr(engine, "create_engine_inspector"):
        raise RuntimeError("TensorRT engine inspector is not available in this TensorRT build.")

    inspector = engine.create_engine_inspector()
    fmt = getattr(getattr(trt, "LayerInformationFormat", object), "JSON", None)
    if fmt is None:
        fmt = getattr(getattr(trt, "LayerInformationFormat", object), "ONELINE", None)
    if fmt is None:
        info = inspector.get_engine_information()
    else:
        info = inspector.get_engine_information(fmt)

    if output_path is None:
        output_path = engine_path.with_name(f"{engine_path.name}.inspector.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(info, encoding="utf-8")

    print(f"Saved TensorRT inspector dump: {output_path}")
    print(f"  FP8 mentions: {info.count('FP8')}")
    print(f"  FP16 mentions: {info.count('FP16')}")
    print(f"  FP32 mentions: {info.count('FP32')}")
    return output_path


def _torch_dtype_from_trt(trt, dtype):
    import numpy as np

    np_dtype = trt.nptype(dtype)
    mapping = {
        np.dtype("float32"): torch.float32,
        np.dtype("float16"): torch.float16,
        np.dtype("int32"): torch.int32,
        np.dtype("int64"): torch.int64,
        np.dtype("bool"): torch.bool,
    }
    if np.dtype(np_dtype) not in mapping:
        raise TypeError(f"Unsupported TensorRT dtype for torch allocation: {dtype} ({np_dtype})")
    return mapping[np.dtype(np_dtype)]


class TRTInference:
    def __init__(self, engine_path: Path, device: str = "cuda", verbose: bool = False) -> None:
        import tensorrt as trt

        if device != "cuda":
            raise ValueError("TensorRT inference requires CUDA device.")
        self.trt = trt
        self.device = torch.device(device)
        self.logger = _trt_logger(trt, verbose)
        trt.init_libnvinfer_plugins(self.logger, "")

        with engine_path.open("rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")

        self.context = self.engine.create_execution_context()
        if hasattr(self.engine, "num_io_tensors"):
            self.io_names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        else:
            self.io_names = [self.engine.get_binding_name(i) for i in range(self.engine.num_bindings)]
        if hasattr(self.engine, "get_tensor_mode"):
            self.input_names = [
                name for name in self.io_names if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            ]
            self.output_names = [
                name for name in self.io_names if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT
            ]
        else:
            self.input_names = [name for name in self.io_names if self.engine.binding_is_input(name)]
            self.output_names = [name for name in self.io_names if not self.engine.binding_is_input(name)]
        self.fixed_batch_size = self._get_fixed_batch_size()

    def _binding_index(self, name: str) -> int:
        return self.io_names.index(name)

    def _get_engine_shape(self, name: str) -> Tuple[int, ...]:
        if hasattr(self.engine, "get_tensor_shape"):
            return tuple(int(dim) for dim in self.engine.get_tensor_shape(name))
        return tuple(int(dim) for dim in self.engine.get_binding_shape(self._binding_index(name)))

    def _is_dynamic_input(self, name: str) -> bool:
        return any(dim < 0 for dim in self._get_engine_shape(name))

    def _get_fixed_batch_size(self) -> Optional[int]:
        if "images" not in self.input_names:
            return None
        shape = self._get_engine_shape("images")
        if shape and shape[0] > 0:
            return shape[0]
        return None

    def _set_input_shape(self, name: str, shape: Tuple[int, ...]) -> None:
        if hasattr(self.context, "set_input_shape"):
            self.context.set_input_shape(name, shape)
        else:
            self.context.set_binding_shape(self._binding_index(name), shape)

    def _get_output_shape(self, name: str) -> Tuple[int, ...]:
        if hasattr(self.context, "get_tensor_shape"):
            return tuple(int(dim) for dim in self.context.get_tensor_shape(name))
        return tuple(int(dim) for dim in self.context.get_binding_shape(self._binding_index(name)))

    def _get_tensor_dtype(self, name: str):
        if hasattr(self.engine, "get_tensor_dtype"):
            return self.engine.get_tensor_dtype(name)
        return self.engine.get_binding_dtype(self._binding_index(name))

    def __call__(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        stream = torch.cuda.current_stream().cuda_stream
        inputs = {name: tensor.contiguous().to(self.device) for name, tensor in inputs.items()}

        for name in self.input_names:
            if name not in inputs:
                raise KeyError(f"Missing TensorRT input: {name}")
            input_shape = tuple(inputs[name].shape)
            if self._is_dynamic_input(name):
                self._set_input_shape(name, input_shape)
            else:
                engine_shape = self._get_engine_shape(name)
                if input_shape != engine_shape:
                    raise ValueError(f"Static TensorRT input shape mismatch for {name}: got {input_shape}, expected {engine_shape}")

        outputs: Dict[str, torch.Tensor] = OrderedDict()
        for name in self.output_names:
            shape = self._get_output_shape(name)
            if any(dim < 0 for dim in shape):
                raise RuntimeError(f"TensorRT output shape is still dynamic for {name}: {shape}")
            dtype = _torch_dtype_from_trt(self.trt, self._get_tensor_dtype(name))
            outputs[name] = torch.empty(shape, dtype=dtype, device=self.device)

        if hasattr(self.context, "execute_async_v3"):
            for name, tensor in inputs.items():
                self.context.set_tensor_address(name, int(tensor.data_ptr()))
            for name, tensor in outputs.items():
                self.context.set_tensor_address(name, int(tensor.data_ptr()))
            ok = self.context.execute_async_v3(stream)
        else:
            bindings = [0] * self.engine.num_bindings
            for i in range(self.engine.num_bindings):
                name = self.engine.get_binding_name(i)
                tensor = inputs.get(name, outputs.get(name))
                bindings[i] = int(tensor.data_ptr())
            ok = self.context.execute_async_v2(bindings=bindings, stream_handle=stream)

        if not ok:
            raise RuntimeError("TensorRT execution failed.")
        return outputs


def _pad_static_batch(
    samples: torch.Tensor,
    orig_target_sizes: torch.Tensor,
    fixed_batch_size: Optional[int],
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    actual_batch_size = samples.shape[0]
    if fixed_batch_size is None or actual_batch_size == fixed_batch_size:
        return samples, orig_target_sizes, actual_batch_size
    if actual_batch_size > fixed_batch_size:
        raise ValueError(
            f"Batch size {actual_batch_size} is larger than static TensorRT batch size {fixed_batch_size}."
        )

    pad = fixed_batch_size - actual_batch_size
    samples = torch.cat([samples, samples[:1].expand(pad, -1, -1, -1)], dim=0)
    orig_target_sizes = torch.cat([orig_target_sizes, orig_target_sizes[:1].expand(pad, -1)], dim=0)
    return samples.contiguous(), orig_target_sizes.contiguous(), actual_batch_size


def _apply_eval_limit(dataset, eval_limit: Optional[int]) -> None:
    if eval_limit is None:
        return
    if eval_limit <= 0:
        raise ValueError(f"--eval-limit must be positive, got {eval_limit}.")
    if eval_limit >= len(dataset):
        print(f"Eval limit {eval_limit} >= dataset size {len(dataset)}; using full validation set.")
        return

    if hasattr(dataset, "image_files"):
        dataset.image_files = dataset.image_files[:eval_limit]
        print(f"Eval limited to first {eval_limit} images.")
        return

    raise TypeError(f"--eval-limit is not implemented for dataset type {type(dataset).__name__}.")


def _scale_normalized_boxes(boxes: torch.Tensor, orig_target_sizes: torch.Tensor) -> torch.Tensor:
    return boxes * orig_target_sizes.repeat(1, 2).unsqueeze(1)


def _parse_engine_outputs(
    outputs: Dict[str, torch.Tensor],
    orig_target_sizes: torch.Tensor,
    actual_batch_size: int,
    cfg: YAMLConfig,
    export_mode: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    orig_target_sizes = orig_target_sizes[:actual_batch_size].float()

    if export_mode == "raw":
        pred_logits = outputs["pred_logits"][:actual_batch_size].float()
        pred_boxes = outputs["pred_boxes"][:actual_batch_size].float()
        labels, boxes, scores = _topk_detections(
            pred_logits,
            pred_boxes,
            int(cfg.yaml_cfg["num_classes"]),
            int(cfg.postprocessor.num_top_queries),
        )
        boxes = _scale_normalized_boxes(boxes, orig_target_sizes)
        return labels.long(), boxes.float(), scores.float()

    labels = outputs["labels"][:actual_batch_size].long()
    boxes = outputs["boxes"][:actual_batch_size].float()
    scores = outputs["scores"][:actual_batch_size].float()

    if export_mode == "normalized":
        boxes = _scale_normalized_boxes(boxes, orig_target_sizes)

    return labels, boxes, scores


def _build_eval_cfg(args) -> YAMLConfig:
    update = parse_cli(args.update)
    input_size = _parse_input_size(args.input_size)
    if input_size is not None:
        update = merge_dict(update, {"eval_spatial_size": input_size})
    if args.data is not None:
        if not hasattr(args, "_resolved_yolo_data"):
            args._resolved_yolo_data = resolve_yolo_data(args.data)
            dataset_root, data_file, val_path, data_cfg = args._resolved_yolo_data
            print(f"YOLO data file: {data_file}")
            print(f"YOLO dataset root: {dataset_root}")
            print(f"YOLO val images: {val_path}")
            if data_cfg is not None:
                print(f"YOLO nc: {data_cfg.get('nc', 'unknown')}")
        else:
            dataset_root, data_file, _, _ = args._resolved_yolo_data
        update = merge_dict(update, {
            "yolo_root": str(dataset_root),
            "yolo_data_file": str(data_file),
        })
    if args.num_top_queries is not None:
        update = merge_dict(update, {"PostProcessor": {"num_top_queries": int(args.num_top_queries)}})
    if args.eval_batch_size is not None:
        update = merge_dict(update, {"val_dataloader": {"total_batch_size": int(args.eval_batch_size)}})
    elif args.opt_batch is not None:
        update = merge_dict(update, {"val_dataloader": {"total_batch_size": int(args.opt_batch)}})
    return YAMLConfig(args.config, **update)


@torch.no_grad()
def evaluate_engine(
    engine_path: Path,
    cfg: YAMLConfig,
    score_threshold: float,
    verbose: bool,
    eval_limit: Optional[int],
    export_mode: str,
) -> List[float]:
    device = "cuda"
    runner = TRTInference(engine_path, device=device, verbose=verbose)
    if runner.fixed_batch_size is not None:
        print(f"TensorRT engine uses static batch size {runner.fixed_batch_size}; final partial batch will be padded and trimmed.")

    val_loader = cfg.val_dataloader
    _apply_eval_limit(val_loader.dataset, eval_limit)
    coco_gt = get_coco_api_from_dataset(val_loader.dataset)
    iou_types = cfg.yaml_cfg["evaluator"].get("iou_types", ["bbox"])
    coco_evaluator = CocoEvaluator(coco_gt, iou_types, verbose=cfg.yaml_cfg["evaluator"].get("verbose", True))
    coco_evaluator.cleanup()

    metric_logger = MetricLogger(delimiter="  ")
    for samples, targets in metric_logger.log_every(val_loader, 10, f"TensorRT eval {engine_path.name}:"):
        samples = samples.to(device, non_blocking=True)
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0).to(device).float()
        samples, orig_target_sizes, actual_batch_size = _pad_static_batch(
            samples,
            orig_target_sizes,
            runner.fixed_batch_size,
        )
        trt_inputs = {"images": samples}
        if "orig_target_sizes" in runner.input_names:
            trt_inputs["orig_target_sizes"] = orig_target_sizes
        outputs = runner(trt_inputs)

        labels, boxes, scores = _parse_engine_outputs(
            outputs,
            orig_target_sizes,
            actual_batch_size,
            cfg,
            export_mode,
        )

        results = []
        for label, box, score in zip(labels, boxes, scores):
            keep = score > score_threshold
            results.append({
                "labels": label[keep].cpu(),
                "boxes": box[keep].cpu(),
                "scores": score[keep].cpu(),
            })

        res = {target["image_id"].item(): output for target, output in zip(targets, results)}
        coco_evaluator.update(res)

    metric_logger.synchronize_between_processes()
    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    if "bbox" in coco_evaluator.coco_eval:
        stats = coco_evaluator.coco_eval["bbox"].stats.tolist()
        print(
            f"{engine_path.name}: "
            f"mAP50-95={stats[0]:.4f}, mAP50={stats[1]:.4f}, mAP75={stats[2]:.4f}"
        )
        return stats
    return []


def _default_onnx_path(checkpoint: str) -> Path:
    return Path(checkpoint).with_suffix(".onnx")


def _engine_path(onnx_path: Path, precision: str, engine_dir: Optional[str]) -> Path:
    parent = Path(engine_dir) if engine_dir else onnx_path.parent
    return parent / f"{onnx_path.stem}.{precision}.engine"


def _select_cuda_device(gpu: Optional[int]) -> None:
    if gpu is None:
        current = torch.cuda.current_device()
    else:
        device_count = torch.cuda.device_count()
        if gpu < 0 or gpu >= device_count:
            visible = os.environ.get("CUDA_VISIBLE_DEVICES")
            visible_msg = f" CUDA_VISIBLE_DEVICES={visible!r}." if visible else ""
            raise ValueError(f"--gpu {gpu} is out of range for {device_count} visible CUDA device(s).{visible_msg}")
        torch.cuda.set_device(gpu)
        current = gpu

    name = torch.cuda.get_device_name(current)
    print(f"Using CUDA device {current}: {name}")


def main(args) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for TensorRT export/evaluation.")
    _select_cuda_device(args.gpu)

    cfg = _build_eval_cfg(args)
    img_h, img_w = cfg.yaml_cfg["eval_spatial_size"]
    print(f"Export/eval spatial size: {img_h}x{img_w}")
    onnx_path = Path(args.onnx) if args.onnx else _default_onnx_path(args.resume)

    if not args.skip_onnx or not onnx_path.exists():
        export_cfg = _build_eval_cfg(args)
        export_onnx(
            cfg=export_cfg,
            checkpoint=args.resume,
            output_file=onnx_path,
            batch_size=args.opt_batch,
            opset=args.opset,
            static_batch=args.static_batch,
            check=args.check,
            simplify=args.simplify,
            strict_load=args.strict_load,
            export_mode=args.export_mode,
        )
    else:
        print(f"Using existing ONNX: {onnx_path}")

    onnx_hw = _read_onnx_image_hw(onnx_path)
    if onnx_hw is not None and onnx_hw != (img_h, img_w):
        raise ValueError(
            f"ONNX images input is {onnx_hw[0]}x{onnx_hw[1]}, but requested/configured "
            f"resolution is {img_h}x{img_w}. Re-export ONNX or use matching --input-size."
        )

    built_engines: List[Path] = []
    for precision in args.precisions:
        engine_path = _engine_path(onnx_path, precision, args.engine_dir)
        if args.skip_build and engine_path.exists():
            built_engines.append(engine_path)
            print(f"Using existing {precision.upper()} engine: {engine_path}")
            continue

        try:
            built = build_engine(
                onnx_file=onnx_path,
                output_file=engine_path,
                precision=precision,
                min_batch=args.min_batch,
                opt_batch=args.opt_batch,
                max_batch=args.max_batch,
                image_hw=(img_h, img_w),
                workspace_gb=args.workspace,
                verbose=args.verbose,
                profiling_verbosity=args.profiling_verbosity,
            )
            if built is not None:
                built_engines.append(built)
        except Exception as exc:
            if precision == "fp8" and not args.strict_fp8:
                print(f"Skip FP8: build failed: {exc}")
            else:
                raise

    if args.no_eval:
        if args.dump_inspector:
            for engine_path in built_engines:
                dump_engine_inspector(engine_path, None, args.verbose)
        return

    if not built_engines:
        raise RuntimeError("No TensorRT engines were built/found for evaluation.")

    eval_cfg = _build_eval_cfg(args)
    for engine_path in built_engines:
        if args.dump_inspector:
            dump_engine_inspector(engine_path, None, args.verbose)
        evaluate_engine(engine_path, eval_cfg, args.score_threshold, args.verbose, args.eval_limit, args.export_mode)


def parse_args():
    parser = argparse.ArgumentParser(description="Export EdgeCrafter to TensorRT and evaluate mAP.")
    parser.add_argument(
        "--config",
        "-c",
        default="configs/ecdet/ecdet_m_yolo.yml",
        type=str,
        help="EdgeCrafter YAML config.",
    )
    parser.add_argument(
        "--resume",
        "-r",
        default="ecdet_m.pth",
        type=str,
        help="Checkpoint to export, e.g. best.pth.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="YOLO dataset root or data.yaml. The script reads it and resolves the val image folder.",
    )
    parser.add_argument("--update", "-u", nargs="*", default=[], help="YAML overrides, e.g. yolo_root=/data/ds.")
    parser.add_argument("--onnx", type=str, default=None, help="Output/input ONNX path.")
    parser.add_argument("--engine-dir", type=str, default=None, help="Directory for TensorRT engines.")
    parser.add_argument(
        "--export-mode",
        choices=["pixel", "normalized", "raw"],
        default="normalized",
        help=(
            "ONNX output contract: pixel keeps labels/boxes/scores with pixel boxes and orig_target_sizes input; "
            "normalized outputs labels/normalized_xyxy_boxes/scores from images only; "
            "raw outputs pred_logits/pred_boxes from images only."
        ),
    )
    parser.add_argument(
        "--precisions",
        nargs="+",
        default=["fp16", "fp8"],
        choices=["fp32", "fp16", "fp8"],
        help="TensorRT precision targets. Use fp32 for no FP16/FP8 quantization.",
    )
    parser.add_argument("--workspace", type=float, default=4.0, help="TensorRT workspace in GB.")
    parser.add_argument(
        "--profiling-verbosity",
        choices=["none", "layer_names_only", "detailed"],
        default="detailed",
        help="TensorRT profiling verbosity stored in the engine for inspector dumps.",
    )
    parser.add_argument("--opset", type=int, default=20)
    parser.add_argument(
        "--input-size",
        nargs="+",
        type=int,
        default=None,
        help="Export/eval resolution. Use S for SxS, or H W. Currently square and divisible by 32.",
    )
    parser.add_argument("--min-batch", type=int, default=1)
    parser.add_argument("--opt-batch", type=int, default=1)
    parser.add_argument("--max-batch", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--eval-limit", type=int, default=None, help="Evaluate only the first N validation images.")
    parser.add_argument("--gpu", type=int, default=None, help="Visible CUDA device index, e.g. --gpu 1.")
    parser.add_argument("--num-top-queries", type=int, default=None, help="Override PostProcessor.num_top_queries.")
    parser.add_argument("--score-threshold", type=float, default=0.0, help="Filter predictions before COCO eval.")
    parser.add_argument("--static-batch", action="store_true", help="Export ONNX without dynamic batch axes.")
    parser.add_argument("--skip-onnx", action="store_true", help="Use existing ONNX if present.")
    parser.add_argument("--skip-build", action="store_true", help="Use existing engine files if present.")
    parser.add_argument("--no-eval", action="store_true", help="Only export/build, do not evaluate mAP.")
    parser.add_argument("--check", action="store_true", help="Run ONNX checker after export.")
    parser.add_argument(
        "--simplify",
        action="store_true",
        default=False,
        help="Run onnxsim after export. Disabled by default; not recommended for TensorRT production builds.",
    )
    parser.add_argument("--strict-load", action="store_true", help="Require checkpoint tensors to match exactly.")
    parser.add_argument("--strict-fp8", action="store_true", help="Fail if FP8 build is requested but fails.")
    parser.add_argument("--dump-inspector", action="store_true", help="Dump TensorRT EngineInspector JSON for each engine.")
    parser.add_argument("--verbose", action="store_true", help="Verbose TensorRT logs.")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
