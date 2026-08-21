"""
Export EdgeCrafter checkpoints to TensorRT and evaluate mAP on the validation set.
"""

import argparse
import hashlib
import json
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
from tools.deployment.onnx_dataflow_fp16 import (
    DATAFLOW_FP16_POLICY,
    DEFAULT_DATA_MAX,
    DEFAULT_INIT_MAX,
    apply_dataflow_fp16_precision,
)
from tools.deployment.onnx_precision import read_precision_policy


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


def _configured_num_top_queries(cfg: YAMLConfig) -> int:
    postprocessor = cfg.yaml_cfg["postprocessor"]
    postprocessor_cfg = cfg.yaml_cfg[postprocessor] if isinstance(postprocessor, str) else postprocessor
    return int(postprocessor_cfg["num_top_queries"])


def _topk_detections(
    logits: torch.Tensor,
    boxes: torch.Tensor,
    num_classes: int,
    num_top_queries: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Sigmoid is monotonic; select logits first to avoid FP16 saturation before TopK.
    scores, flat_index = torch.topk(logits.flatten(1), num_top_queries, dim=-1)
    scores = torch.sigmoid(scores)
    labels = (flat_index - flat_index // num_classes * num_classes).to(torch.int32)
    box_index = flat_index // num_classes
    boxes = _batch_index_select(_box_cxcywh_to_xyxy(boxes), box_index)
    return labels, boxes, scores


class DeployModel(nn.Module):
    def __init__(self, cfg: YAMLConfig, export_mode: str, input_dtype: str = "float16") -> None:
        super().__init__()
        self.model = cfg.model.deploy()
        if input_dtype not in ("float32", "float16"):
            raise ValueError(f"Unsupported input dtype: {input_dtype}")
        self.input_dtype = input_dtype
        self.export_mode = export_mode
        self.num_classes = int(cfg.yaml_cfg["num_classes"])
        self.num_top_queries = _configured_num_top_queries(cfg)
        if export_mode == "pixel":
            self.postprocessor = cfg.postprocessor.deploy()

    def forward(self, images: torch.Tensor, orig_target_sizes: Optional[torch.Tensor] = None):
        if self.input_dtype == "float16":
            # Preserve the baseline model graph while making HALF the external
            # contract. The input values remain FP16-quantized; TensorRT may fold
            # this widening cast into its first FP16 convolution tactic.
            images = images.float()
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


def _materialize_reduce_axes_for_tensorrt(model_path: Path) -> int:
    """Store constant Reduce axes as initializers for TensorRT's ONNX parser."""
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper

    onnx_model = onnx.load(str(model_path))
    producers = {output: node for node in onnx_model.graph.node for output in node.output}
    constants = {
        initializer.name: numpy_helper.to_array(initializer)
        for initializer in onnx_model.graph.initializer
    }

    def evaluate_constant(name: str):
        if name in constants:
            return constants[name]
        node = producers.get(name)
        if node is None:
            raise ValueError(f"ONNX value is not constant: {name}")

        attributes = {attribute.name: attribute for attribute in node.attribute}
        if node.op_type == "Constant" and "value" in attributes:
            value = numpy_helper.to_array(attributes["value"].t)
        elif node.op_type == "Constant" and "value_ints" in attributes:
            value = np.asarray(attributes["value_ints"].ints, dtype=np.int64)
        elif node.op_type == "Constant" and "value_int" in attributes:
            value = np.asarray(attributes["value_int"].i, dtype=np.int64)
        elif node.op_type == "Reshape":
            value = np.reshape(
                evaluate_constant(node.input[0]),
                evaluate_constant(node.input[1]).astype(np.int64).tolist(),
            )
        elif node.op_type == "Cast":
            dtype = helper.tensor_dtype_to_np_dtype(attributes["to"].i)
            value = evaluate_constant(node.input[0]).astype(dtype)
        else:
            raise ValueError(f"Unsupported constant ONNX op {node.op_type} for value {name}")

        constants[name] = value
        return value

    materialized = 0
    for index, node in enumerate(onnx_model.graph.node):
        if not node.op_type.startswith("Reduce") or len(node.input) < 2 or not node.input[1]:
            continue
        axes = np.asarray(evaluate_constant(node.input[1]), dtype=np.int64)
        initializer_name = f"_trt_reduce_axes_{index}"
        onnx_model.graph.initializer.append(numpy_helper.from_array(axes, initializer_name))
        node.input[1] = initializer_name
        materialized += 1

    if materialized:
        onnx.save(onnx_model, str(model_path), save_as_external_data=False)
    return materialized


def _restore_optimizer_clip_bounds(model_path: Path, unoptimized_model) -> int:
    """Restore Clip bounds dropped by the PyTorch 2.9 ONNX optimizer."""
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper

    optimized_model = onnx.load(str(model_path))
    optimized_values = (
        {value.name for value in optimized_model.graph.input}
        | {value.name for value in optimized_model.graph.initializer}
        | {output for node in optimized_model.graph.node for output in node.output}
    )
    raw_producers = {
        output: node
        for node in unoptimized_model.graph.node
        for output in node.output
    }
    raw_constants = {
        initializer.name: numpy_helper.to_array(initializer)
        for initializer in unoptimized_model.graph.initializer
    }

    def evaluate_raw_constant(name: str):
        if name in raw_constants:
            return raw_constants[name]
        node = raw_producers.get(name)
        if node is None:
            raise ValueError(f"Unoptimized ONNX value is not constant: {name}")
        attributes = {attribute.name: attribute for attribute in node.attribute}
        if node.op_type == "Constant" and "value" in attributes:
            value = numpy_helper.to_array(attributes["value"].t)
        elif node.op_type == "Constant" and "value_ints" in attributes:
            value = np.asarray(attributes["value_ints"].ints, dtype=np.int64)
        elif node.op_type == "Constant" and "value_int" in attributes:
            value = np.asarray(attributes["value_int"].i, dtype=np.int64)
        elif node.op_type == "Cast":
            dtype = helper.tensor_dtype_to_np_dtype(attributes["to"].i)
            value = evaluate_raw_constant(node.input[0]).astype(dtype)
        else:
            raise ValueError(f"Unsupported raw constant ONNX op {node.op_type} for value {name}")
        raw_constants[name] = value
        return value

    restored = 0
    for node in optimized_model.graph.node:
        if node.op_type != "Clip" or len(node.input) < 3:
            continue
        missing_min = node.input[1] and node.input[1] not in optimized_values
        missing_max = node.input[2] and node.input[2] not in optimized_values
        if not missing_min and not missing_max:
            continue

        raw_clip = raw_producers.get(node.output[0])
        if raw_clip is None or raw_clip.op_type != "Min":
            raise ValueError(f"Can not recover Clip bounds for ONNX value {node.output[0]}")
        raw_maximum = raw_producers.get(raw_clip.input[0])
        if raw_maximum is None or raw_maximum.op_type != "Max":
            raise ValueError(f"Can not recover Clip minimum for ONNX value {node.output[0]}")

        if missing_min:
            minimum = np.asarray(evaluate_raw_constant(raw_maximum.input[1]))
            optimized_model.graph.initializer.append(numpy_helper.from_array(minimum, node.input[1]))
            optimized_values.add(node.input[1])
        if missing_max:
            maximum = np.asarray(evaluate_raw_constant(raw_clip.input[1]))
            optimized_model.graph.initializer.append(numpy_helper.from_array(maximum, node.input[2]))
            optimized_values.add(node.input[2])
        restored += 1

    if restored:
        onnx.save(optimized_model, str(model_path), save_as_external_data=False)
    return restored


def write_modelopt_calibration_data(
    cfg: YAMLConfig,
    output_path: Path,
    sample_count: int,
    export_mode: str,
    input_dtype: str,
) -> Path:
    """Write real validation tensors for ModelOpt calibration/activation analysis."""
    import numpy as np

    if sample_count <= 0:
        raise ValueError("ModelOpt calibration sample count must be positive.")

    image_chunks = []
    size_chunks = []
    collected = 0
    for samples, targets in cfg.val_dataloader:
        take = min(sample_count - collected, int(samples.shape[0]))
        image_chunks.append(samples[:take].detach().cpu())
        if export_mode == "pixel":
            size_chunks.append(
                torch.stack([target["orig_size"] for target in targets[:take]], dim=0).cpu()
            )
        collected += take
        if collected >= sample_count:
            break

    if collected != sample_count:
        raise ValueError(
            f"Validation set provided {collected} calibration samples; requested {sample_count}."
        )

    image_torch_dtype = torch.float16 if input_dtype == "float16" else torch.float32
    arrays = {
        "images": torch.cat(image_chunks, dim=0).to(image_torch_dtype).contiguous().numpy()
    }
    if export_mode == "pixel":
        arrays["orig_target_sizes"] = torch.cat(size_chunks, dim=0).float().contiguous().numpy()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    images = arrays["images"]
    print(
        f"Saved {sample_count} real validation sample(s) for ModelOpt calibration: {output_path} "
        f"shape={images.shape}, dtype={images.dtype}, range=[{images.min():.6g}, {images.max():.6g}]"
    )
    return output_path


def _write_fp16_calibration(
    cfg: YAMLConfig,
    output_path: Path,
    sample_count: int,
    export_mode: str,
    input_dtype: str,
) -> Path:
    """Backward-compatible name for the existing FP16 dataflow policy."""
    return write_modelopt_calibration_data(
        cfg,
        output_path,
        sample_count=sample_count,
        export_mode=export_mode,
        input_dtype=input_dtype,
    )


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
    input_dtype: str = "float16",
    onnx_exporter: str = "legacy",
    onnx_precision_policy: str = "baseline",
    fp16_report: Optional[Path] = None,
    fp16_calibration_data: Optional[Path] = None,
    fp16_calibration_samples: int = 1,
    fp16_data_max: float = DEFAULT_DATA_MAX,
    fp16_init_max: float = DEFAULT_INIT_MAX,
) -> Path:
    if onnx_exporter not in ("legacy", "dynamo"):
        raise ValueError(f"Unsupported ONNX exporter: {onnx_exporter}")
    if onnx_precision_policy not in ("baseline", "explicit-fp16-dataflow"):
        raise ValueError(f"Unsupported ONNX precision policy: {onnx_precision_policy}")
    if "ViTAdapter" in cfg.yaml_cfg:
        cfg.yaml_cfg["ViTAdapter"]["skip_load_backbone"] = True
    _load_checkpoint_model(cfg, checkpoint, strict=strict_load)

    model = DeployModel(cfg, export_mode, input_dtype=input_dtype).eval()
    img_h, img_w = cfg.yaml_cfg["eval_spatial_size"]
    image_dtype = torch.float16 if input_dtype == "float16" else torch.float32
    images = torch.rand(batch_size, 3, img_h, img_w, dtype=image_dtype)
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
    dynamic_shapes = None
    if not static_batch:
        if onnx_exporter == "legacy":
            dynamic_axes = {"images": {0: "N"}}
            if "orig_target_sizes" in input_names:
                dynamic_axes["orig_target_sizes"] = {0: "N"}
            dynamic_axes.update({name: {0: "N"} for name in output_names})
        else:
            dynamic_shapes = {"images": {0: "N"}}
            if "orig_target_sizes" in input_names:
                dynamic_shapes["orig_target_sizes"] = {0: "N"}

    output_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"ONNX exporter: {onnx_exporter}")
    with torch.no_grad():
        _ = model(*input_args)
        if onnx_exporter == "legacy":
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
        else:
            onnx_program = torch.onnx.export(
                model,
                input_args,
                input_names=input_names,
                output_names=output_names,
                dynamic_shapes=dynamic_shapes,
                opset_version=opset,
                verbose=False,
                external_data=False,
                dynamo=True,
                optimize=False,
            )
            unoptimized_model = onnx_program.model_proto
            onnx_program.optimize()
            onnx_program.save(str(output_file), external_data=False)

    if onnx_exporter == "dynamo":
        restored_clips = _restore_optimizer_clip_bounds(output_file, unoptimized_model)
        if restored_clips:
            print(f"Restored bounds for {restored_clips} ONNX Clip nodes after optimization.")
        materialized_axes = _materialize_reduce_axes_for_tensorrt(output_file)
        if materialized_axes:
            print(f"Materialized {materialized_axes} constant Reduce axes for TensorRT.")

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

    if onnx_precision_policy == "explicit-fp16-dataflow":
        calibration_path = (
            Path(fp16_calibration_data)
            if fp16_calibration_data
            else output_file.with_suffix(".fp16-calibration.npz")
        )
        if fp16_calibration_data is None:
            _write_fp16_calibration(
                cfg,
                calibration_path,
                sample_count=fp16_calibration_samples,
                export_mode=export_mode,
                input_dtype=input_dtype,
            )
        report = apply_dataflow_fp16_precision(
            output_file,
            calibration_path=calibration_path,
            report_path=fp16_report,
            data_max=fp16_data_max,
            init_max=fp16_init_max,
        )
        converted_initializers = report["converted_initializers"]["count"]
        print(
            "Applied calibrated dataflow FP16 policy: "
            f"{report['graph']['cast_node_count']} Cast nodes, "
            f"initializers={converted_initializers}."
        )
        print(f"Saved calibrated FP16 policy report: {report['report']}")

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


def _network_creation_flags(trt, strongly_typed: bool) -> int:
    flags = (
        0
        if _trt_major(trt) >= 10
        else (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    )
    if not strongly_typed:
        return flags
    if not hasattr(trt.NetworkDefinitionCreationFlag, "STRONGLY_TYPED"):
        raise RuntimeError(
            f"TensorRT {trt.__version__} does not expose STRONGLY_TYPED networks; "
            "explicit ONNX precision policy cannot be enforced."
        )
    return flags | (1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))


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


def _onnx_dtype_name(onnx, elem_type: int) -> str:
    mapping = {
        onnx.TensorProto.FLOAT: "float32",
        onnx.TensorProto.FLOAT16: "float16",
        onnx.TensorProto.INT32: "int32",
        onnx.TensorProto.INT64: "int64",
        onnx.TensorProto.BOOL: "bool",
    }
    dtype_name = mapping.get(elem_type)
    if dtype_name is None:
        raise TypeError(f"Unsupported ONNX tensor dtype: {onnx.TensorProto.DataType.Name(elem_type)}")
    return dtype_name


def _onnx_tensor_shape(value_info) -> Tuple[int, ...]:
    return tuple(int(dim.dim_value) if dim.dim_value > 0 else -1 for dim in value_info.type.tensor_type.shape.dim)


def _read_onnx_contract(onnx_file: Path) -> Dict[str, object]:
    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError("The onnx package is required to validate the input contract") from exc
    model = onnx.load(str(onnx_file), load_external_data=False)
    inputs = {
        value.name: _onnx_dtype_name(onnx, value.type.tensor_type.elem_type)
        for value in model.graph.input
    }
    outputs = {
        value.name: _onnx_dtype_name(onnx, value.type.tensor_type.elem_type)
        for value in model.graph.output
    }
    input_shapes = {value.name: _onnx_tensor_shape(value) for value in model.graph.input}
    output_shapes = {value.name: _onnx_tensor_shape(value) for value in model.graph.output}
    if "images" not in inputs:
        raise KeyError(f"ONNX model has no 'images' input: {onnx_file}")
    return {
        "inputs": inputs,
        "outputs": outputs,
        "input_shapes": input_shapes,
        "output_shapes": output_shapes,
    }


def _expected_io_dtypes(task: str, export_mode: str, input_dtype: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    if task == "segmentation" and export_mode != "pixel":
        raise NotImplementedError(f"--export-mode {export_mode} is only implemented for detection.")

    inputs = {"images": input_dtype}
    if export_mode == "pixel":
        inputs["orig_target_sizes"] = "float32"

    if export_mode == "raw":
        outputs = {"pred_logits": "float32", "pred_boxes": "float32"}
    else:
        outputs = {"labels": "int32", "boxes": "float32", "scores": "float32"}
        if task == "segmentation":
            outputs["masks"] = "float32"
    return inputs, outputs


def _validate_onnx_contract(
    contract: Dict[str, object],
    task: str,
    export_mode: str,
    input_dtype: str,
    image_hw: Tuple[int, int],
    static_batch: bool,
    opt_batch: int,
) -> None:
    expected_inputs, expected_outputs = _expected_io_dtypes(task, export_mode, input_dtype)
    if contract["inputs"] != expected_inputs:
        raise ValueError(f"ONNX input contract mismatch: got {contract['inputs']}, expected {expected_inputs}")
    if contract["outputs"] != expected_outputs:
        raise ValueError(f"ONNX output contract mismatch: got {contract['outputs']}, expected {expected_outputs}")

    input_shapes = contract["input_shapes"]
    image_shape = input_shapes["images"]
    expected_batch = opt_batch if static_batch else -1
    expected_image_shape = (expected_batch, 3, image_hw[0], image_hw[1])
    if image_shape != expected_image_shape:
        raise ValueError(f"ONNX images shape mismatch: got {image_shape}, expected {expected_image_shape}")
    if "orig_target_sizes" in expected_inputs:
        expected_size_shape = (expected_batch, 2)
        if input_shapes.get("orig_target_sizes") != expected_size_shape:
            raise ValueError(
                f"ONNX orig_target_sizes shape mismatch: got {input_shapes.get('orig_target_sizes')}, "
                f"expected {expected_size_shape}"
            )


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
    strongly_typed: bool = False,
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

    network = builder.create_network(_network_creation_flags(trt, strongly_typed))
    parser = trt.OnnxParser(network, logger)

    if not parser.parse_from_file(str(onnx_file)):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"Failed to parse ONNX file {onnx_file}:\n{errors}")

    typing_mode = "strongly typed" if strongly_typed else "weakly typed"
    print(f"TensorRT {trt.__version__}: building {precision.upper()} engine ({typing_mode})")
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        print(f'  input  {inp.name}: shape={tuple(inp.shape)} dtype={inp.dtype}')
    for i in range(network.num_outputs):
        out = network.get_output(i)
        print(f'  output {out.name}: shape={tuple(out.shape)} dtype={out.dtype}')

    if strongly_typed:
        # Strong typing takes tensor and compute precision from the ONNX graph.
        # TensorRT forbids the global FP16/FP8 builder flags in this mode.
        if precision not in {"fp16", "fp8"}:
            raise ValueError(
                "Strongly typed export currently supports explicit FP16 or FP8 ONNX graphs."
            )
    else:
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


def _dtype_name(dtype: torch.dtype) -> str:
    mapping = {
        torch.float32: "float32",
        torch.float16: "float16",
        torch.int32: "int32",
        torch.int64: "int64",
        torch.bool: "bool",
    }
    if dtype not in mapping:
        raise TypeError(f"Unsupported tensor dtype: {dtype}")
    return mapping[dtype]


def _shape_tuple(shape) -> Tuple[int, ...]:
    return tuple(int(dim) for dim in shape)


def _profile_tuple(profile) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]:
    return tuple(_shape_tuple(shape) for shape in profile)


def _read_engine_contract(engine_path: Path, verbose: bool) -> Dict[str, object]:
    import tensorrt as trt

    logger = _trt_logger(trt, verbose)
    trt.init_libnvinfer_plugins(logger, "")
    with engine_path.open("rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")

        profile_count = max(int(getattr(engine, "num_optimization_profiles", 1)), 1)
        inputs: Dict[str, str] = {}
        outputs: Dict[str, str] = {}
        input_shapes: Dict[str, Tuple[int, ...]] = {}
        output_shapes: Dict[str, Tuple[int, ...]] = {}
        profiles: Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]] = {}

        named_api = all(hasattr(engine, attr) for attr in (
            "num_io_tensors",
            "get_tensor_name",
            "get_tensor_mode",
            "get_tensor_dtype",
            "get_tensor_shape",
        ))
        if named_api:
            for i in range(engine.num_io_tensors):
                name = engine.get_tensor_name(i)
                dtype = _dtype_name(_torch_dtype_from_trt(trt, engine.get_tensor_dtype(name)))
                shape = _shape_tuple(engine.get_tensor_shape(name))
                if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    inputs[name] = dtype
                    input_shapes[name] = shape
                    if any(dim < 0 for dim in shape):
                        if not hasattr(engine, "get_tensor_profile_shape"):
                            raise RuntimeError("TensorRT named I/O API has no profile-shape query support.")
                        profiles[name] = _profile_tuple(engine.get_tensor_profile_shape(name, 0))
                else:
                    outputs[name] = dtype
                    output_shapes[name] = shape
        else:
            bindings_per_profile = engine.num_bindings // profile_count
            for i in range(bindings_per_profile):
                name = engine.get_binding_name(i)
                dtype = _dtype_name(_torch_dtype_from_trt(trt, engine.get_binding_dtype(i)))
                shape = _shape_tuple(engine.get_binding_shape(i))
                if engine.binding_is_input(i):
                    if hasattr(engine, "is_shape_binding") and engine.is_shape_binding(i):
                        raise TypeError(f"TensorRT shape input is not supported for reuse validation: {name}")
                    inputs[name] = dtype
                    input_shapes[name] = shape
                    if any(dim < 0 for dim in shape):
                        profiles[name] = _profile_tuple(engine.get_profile_shape(0, i))
                else:
                    outputs[name] = dtype
                    output_shapes[name] = shape

    if "images" not in inputs:
        raise KeyError(f"TensorRT engine has no 'images' input: {engine_path}")
    return {
        "inputs": inputs,
        "outputs": outputs,
        "input_shapes": input_shapes,
        "output_shapes": output_shapes,
        "profiles": profiles,
        "profile_count": profile_count,
    }


def _validate_batch_request(
    static_batch: bool,
    min_batch: int,
    opt_batch: int,
    max_batch: int,
    eval_batch_size: Optional[int] = None,
) -> None:
    if not 1 <= min_batch <= opt_batch <= max_batch:
        raise ValueError(
            f"Invalid batch profile: require 1 <= min <= opt <= max, got "
            f"{min_batch}/{opt_batch}/{max_batch}"
        )
    if static_batch and not min_batch == opt_batch == max_batch:
        raise ValueError(
            "Static batch requires --min-batch, --opt-batch and --max-batch to be equal, "
            f"got {min_batch}/{opt_batch}/{max_batch}"
        )
    if eval_batch_size is not None:
        valid_eval_batch = (
            eval_batch_size == opt_batch
            if static_batch
            else min_batch <= eval_batch_size <= max_batch
        )
        if not valid_eval_batch:
            raise ValueError(
                f"Evaluation batch size {eval_batch_size} is outside the requested "
                f"{'static batch' if static_batch else 'profile'} {min_batch}/{opt_batch}/{max_batch}"
            )


def _validate_engine_contract(
    engine_contract: Dict[str, object],
    onnx_contract: Dict[str, object],
    image_hw: Tuple[int, int],
    static_batch: bool,
    min_batch: int,
    opt_batch: int,
    max_batch: int,
) -> None:
    _validate_batch_request(static_batch, min_batch, opt_batch, max_batch)
    if engine_contract["profile_count"] != 1:
        raise ValueError(
            f"TensorRT runner requires exactly one optimization profile, got {engine_contract['profile_count']}"
        )

    for key in ("inputs", "outputs", "input_shapes"):
        if engine_contract[key] != onnx_contract[key]:
            raise ValueError(
                f"TensorRT {key} contract mismatch: got {engine_contract[key]}, "
                f"expected {onnx_contract[key]}"
            )

    engine_output_shapes = engine_contract["output_shapes"]
    onnx_output_shapes = onnx_contract["output_shapes"]
    if engine_output_shapes.keys() != onnx_output_shapes.keys():
        raise ValueError(
            f"TensorRT output shape names mismatch: got {engine_output_shapes.keys()}, "
            f"expected {onnx_output_shapes.keys()}"
        )
    for name, onnx_shape in onnx_output_shapes.items():
        engine_shape = engine_output_shapes[name]
        compatible = len(engine_shape) == len(onnx_shape) and all(
            expected < 0 or actual == expected
            for actual, expected in zip(engine_shape, onnx_shape)
        )
        if not compatible:
            raise ValueError(
                f"TensorRT output shape mismatch for {name}: got {engine_shape}, "
                f"expected ONNX-compatible {onnx_shape}"
            )

    expected_profiles = {}
    if not static_batch:
        img_h, img_w = image_hw
        expected_profiles["images"] = (
            (min_batch, 3, img_h, img_w),
            (opt_batch, 3, img_h, img_w),
            (max_batch, 3, img_h, img_w),
        )
        if "orig_target_sizes" in onnx_contract["inputs"]:
            expected_profiles["orig_target_sizes"] = (
                (min_batch, 2),
                (opt_batch, 2),
                (max_batch, 2),
            )
    if engine_contract["profiles"] != expected_profiles:
        raise ValueError(
            f"TensorRT optimization profile mismatch: got {engine_contract['profiles']}, "
            f"expected {expected_profiles}"
        )


def _validate_engine_top_queries(
    engine_contract: Dict[str, object],
    export_mode: str,
    num_top_queries: int,
) -> None:
    if export_mode == "raw":
        return
    if num_top_queries <= 0:
        raise ValueError(f"num_top_queries must be positive, got {num_top_queries}")

    output_shapes = engine_contract["output_shapes"]
    for name in ("labels", "boxes", "scores"):
        shape = output_shapes[name]
        if len(shape) < 2 or shape[1] != num_top_queries:
            raise ValueError(
                f"TensorRT {name} output has {shape[1] if len(shape) >= 2 else 'no'} query dimension, "
                f"expected {num_top_queries}: shape={shape}"
            )


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

    def input_torch_dtype(self, name: str) -> torch.dtype:
        if name not in self.input_names:
            raise KeyError(f"Unknown TensorRT input: {name}")
        return _torch_dtype_from_trt(self.trt, self._get_tensor_dtype(name))

    def __call__(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        stream = torch.cuda.current_stream().cuda_stream
        for name in self.input_names:
            if name not in inputs:
                raise KeyError(f"Missing TensorRT input: {name}")
        unexpected = sorted(set(inputs) - set(self.input_names))
        if unexpected:
            raise KeyError(f"Unexpected TensorRT inputs: {unexpected}")

        prepared_inputs = {}
        for name, tensor in inputs.items():
            expected_dtype = self.input_torch_dtype(name)
            if tensor.dtype != expected_dtype:
                raise TypeError(
                    f"TensorRT input {name!r} expects {expected_dtype}, got {tensor.dtype}. "
                    "Convert it at the producer/host boundary before H2D."
                )
            prepared_inputs[name] = tensor.contiguous().to(device=self.device, non_blocking=True)
        inputs = prepared_inputs

        for name in self.input_names:
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


def _cast_host_tensor(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if tensor.device.type != "cpu":
        raise ValueError(f"Expected a CPU tensor before H2D, got {tensor.device}")
    if tensor.dtype == dtype:
        return tensor.contiguous()

    converted = torch.empty(
        tuple(tensor.shape),
        dtype=dtype,
        device="cpu",
        pin_memory=tensor.is_pinned(),
    )
    converted.copy_(tensor)
    return converted


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
            _configured_num_top_queries(cfg),
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
        samples = _cast_host_tensor(samples, runner.input_torch_dtype("images"))
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        target_size_dtype = (
            runner.input_torch_dtype("orig_target_sizes")
            if "orig_target_sizes" in runner.input_names
            else torch.float32
        )
        orig_target_sizes = _cast_host_tensor(orig_target_sizes, target_size_dtype)
        samples, orig_target_sizes, actual_batch_size = _pad_static_batch(
            samples,
            orig_target_sizes,
            runner.fixed_batch_size,
        )
        orig_target_sizes_device = orig_target_sizes.to(device, dtype=torch.float32, non_blocking=True)
        trt_inputs = {"images": samples}
        if "orig_target_sizes" in runner.input_names:
            trt_inputs["orig_target_sizes"] = orig_target_sizes
        outputs = runner(trt_inputs)

        labels, boxes, scores = _parse_engine_outputs(
            outputs,
            orig_target_sizes_device,
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
            f"mAP50-95={stats[0]:.4f}, mAP50={stats[1]:.4f}, mAP75={stats[2]:.4f}; "
            f"AR1={stats[6]:.4f}, AR10={stats[7]:.4f}, AR100={stats[8]:.4f}; "
            f"AR-small={stats[9]:.4f}, AR-medium={stats[10]:.4f}, AR-large={stats[11]:.4f}"
        )
        return stats
    return []


def _default_onnx_path(checkpoint: str) -> Path:
    return Path(checkpoint).with_suffix(".onnx")


def _engine_path(onnx_path: Path, precision: str, engine_dir: Optional[str]) -> Path:
    parent = Path(engine_dir) if engine_dir else onnx_path.parent
    return parent / f"{onnx_path.stem}.{precision}.engine"


def _engine_manifest_path(engine_path: Path) -> Path:
    return engine_path.with_name(f"{engine_path.name}.manifest.json")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_engine_manifest(engine_path: Path, onnx_path: Path, precision: str) -> Path:
    manifest_path = _engine_manifest_path(engine_path)
    manifest = {
        "schema_version": 1,
        "precision": precision,
        "onnx_sha256": _sha256_file(onnx_path),
        "engine_sha256": _sha256_file(engine_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def _validate_engine_manifest(engine_path: Path, onnx_path: Path, precision: str) -> Path:
    manifest_path = _engine_manifest_path(engine_path)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Can not safely reuse TensorRT engine without its manifest: {manifest_path}. "
            "Rebuild the engine once without --skip-build."
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid TensorRT engine manifest: {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"TensorRT engine manifest must be a JSON object: {manifest_path}")

    expected = {
        "schema_version": 1,
        "precision": precision,
        "onnx_sha256": _sha256_file(onnx_path),
        "engine_sha256": _sha256_file(engine_path),
    }
    if manifest != expected:
        mismatches = {
            key: {"got": manifest.get(key), "expected": value}
            for key, value in expected.items()
            if manifest.get(key) != value
        }
        raise ValueError(
            f"TensorRT engine manifest does not match the requested ONNX/engine: {mismatches}. "
            "Rebuild without --skip-build."
        )
    return manifest_path


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
    if not args.onnx_only and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for TensorRT build/evaluation. Use --onnx-only for CPU export.")
    _validate_batch_request(
        args.static_batch,
        args.min_batch,
        args.opt_batch,
        args.max_batch,
        args.eval_batch_size,
    )
    if not args.onnx_only:
        _select_cuda_device(args.gpu)
    if args.fp16_report and args.onnx_precision_policy != "explicit-fp16-dataflow":
        raise ValueError("--fp16-report requires --onnx-precision-policy explicit-fp16-dataflow.")
    if args.fp16_calibration_data and args.onnx_precision_policy != "explicit-fp16-dataflow":
        raise ValueError(
            "--fp16-calibration-data requires --onnx-precision-policy explicit-fp16-dataflow."
        )

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
            input_dtype=args.input_dtype,
            onnx_exporter=args.onnx_exporter,
            onnx_precision_policy=args.onnx_precision_policy,
            fp16_report=Path(args.fp16_report) if args.fp16_report else None,
            fp16_calibration_data=(
                Path(args.fp16_calibration_data) if args.fp16_calibration_data else None
            ),
            fp16_calibration_samples=args.fp16_calibration_samples,
            fp16_data_max=args.fp16_data_max,
            fp16_init_max=args.fp16_init_max,
        )
    else:
        print(f"Using existing ONNX: {onnx_path}")

    actual_precision_policy = read_precision_policy(onnx_path)
    expected_precision_policy = {
        "baseline": "baseline",
        "explicit-fp16-dataflow": DATAFLOW_FP16_POLICY,
    }[args.onnx_precision_policy]
    if actual_precision_policy != expected_precision_policy:
        raise ValueError(
            f"ONNX precision policy mismatch: model has {actual_precision_policy!r}, "
            f"requested {expected_precision_policy!r}. Re-export without --skip-onnx."
        )

    onnx_contract = _read_onnx_contract(onnx_path)
    _validate_onnx_contract(
        onnx_contract,
        task=cfg.yaml_cfg["task"],
        export_mode=args.export_mode,
        input_dtype=args.input_dtype,
        image_hw=(img_h, img_w),
        static_batch=args.static_batch,
        opt_batch=args.opt_batch,
    )
    if args.onnx_only:
        print(f"ONNX-only export completed: {onnx_path}")
        return

    def validate_engine(engine_path: Path) -> Dict[str, object]:
        engine_contract = _read_engine_contract(engine_path, args.verbose)
        _validate_engine_contract(
            engine_contract,
            onnx_contract,
            image_hw=(img_h, img_w),
            static_batch=args.static_batch,
            min_batch=args.min_batch,
            opt_batch=args.opt_batch,
            max_batch=args.max_batch,
        )
        _validate_engine_top_queries(
            engine_contract,
            export_mode=args.export_mode,
            num_top_queries=_configured_num_top_queries(cfg),
        )
        return engine_contract

    built_engines: List[Path] = []
    for precision in args.precisions:
        engine_path = _engine_path(onnx_path, precision, args.engine_dir)
        if args.skip_build and engine_path.exists():
            _validate_engine_manifest(engine_path, onnx_path, precision)
            engine_contract = validate_engine(engine_path)
            built_engines.append(engine_path)
            print(
                f"Using existing {precision.upper()} engine with "
                f"{engine_contract['inputs']['images']} input and matching I/O/profile contract: "
                f"{engine_path}"
            )
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
                strongly_typed=actual_precision_policy == DATAFLOW_FP16_POLICY,
            )
            if built is not None:
                validate_engine(built)
                manifest_path = _write_engine_manifest(built, onnx_path, precision)
                print(f"Saved TensorRT engine manifest: {manifest_path}")
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
        default=["fp16"],
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
        "--onnx-exporter",
        choices=["legacy", "dynamo"],
        default="legacy",
        help="ONNX exporter backend (default: legacy for current TensorRT production compatibility).",
    )
    parser.add_argument(
        "--onnx-precision-policy",
        choices=["baseline", "explicit-fp16-dataflow"],
        default="baseline",
        help=(
            "ONNX internal precision policy. explicit-fp16-dataflow uses real validation "
            "activations plus minimal FP32 ranking/box-decoding islands and builds a "
            "strongly typed TensorRT network. baseline is retained for comparison tooling."
        ),
    )
    parser.add_argument(
        "--fp16-report",
        type=str,
        default=None,
        help="Optional JSON report path for explicit-fp16-dataflow.",
    )
    parser.add_argument(
        "--fp16-calibration-data",
        type=str,
        default=None,
        help=(
            "NPZ calibration inputs for explicit-fp16-dataflow. When omitted, real validation "
            "samples are collected through the configured val dataloader."
        ),
    )
    parser.add_argument(
        "--fp16-calibration-samples",
        type=int,
        default=1,
        help="Number of real validation samples to collect for dataflow FP16 calibration.",
    )
    parser.add_argument(
        "--fp16-data-max",
        type=float,
        default=DEFAULT_DATA_MAX,
        help="Maximum calibrated activation magnitude eligible for FP16 conversion.",
    )
    parser.add_argument(
        "--fp16-init-max",
        type=float,
        default=DEFAULT_INIT_MAX,
        help="Maximum initializer magnitude eligible for FP16 conversion.",
    )
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
    parser.add_argument(
        "--input-dtype",
        choices=["float32", "float16"],
        default="float16",
        help="ONNX/TensorRT images input contract (default: float16). Use float32 as a compatibility fallback.",
    )
    parser.add_argument(
        "--static-batch",
        action="store_true",
        help="Export ONNX without dynamic batch axes; min/opt/max batch values must be equal.",
    )
    parser.add_argument("--skip-onnx", action="store_true", help="Use existing ONNX if present.")
    parser.add_argument(
        "--onnx-only",
        action="store_true",
        help="Export and validate ONNX without requiring CUDA or building TensorRT engines.",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Use existing engine files only when their SHA-256 manifests match the ONNX and engine bytes.",
    )
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
