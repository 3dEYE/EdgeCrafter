"""
EdgeCrafter: Compact ViTs for Edge Dense Prediction via Task-Specialized Distillation
Copyright (c) 2026 The EdgeCrafter Authors. All Rights Reserved.
---------------------------------------------------------------------------------
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))

import torch
import torch.nn as nn

from engine.core import YAMLConfig


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
):
    scores = torch.sigmoid(logits)
    scores, flat_index = torch.topk(scores.flatten(1), num_top_queries, dim=-1)
    labels = (flat_index - flat_index // num_classes * num_classes).to(torch.int32)
    box_index = flat_index // num_classes
    boxes = _batch_index_select(_box_cxcywh_to_xyxy(boxes), box_index)
    return labels, boxes, scores


def main(args, ):
    """main
    """
    cfg = YAMLConfig(args.config, resume=args.resume)
    
    task = cfg.yaml_cfg['task']

    if args.resume:
        cfg.yaml_cfg['ViTAdapter']['skip_load_backbone'] = True
        checkpoint = torch.load(args.resume, map_location='cpu')
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']

        # NOTE load train mode state -> convert to deploy mode
        cfg.model.load_state_dict(state)

    else:
        # raise AttributeError('Only support resume to load model.state_dict by now.')
        print('not load model.state_dict, use default init state dict...')

    class Model(nn.Module):
        def __init__(self, ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.export_mode = args.export_mode
            self.num_classes = int(cfg.postprocessor.num_classes)
            self.num_top_queries = int(cfg.postprocessor.num_top_queries)
            if self.export_mode == 'pixel':
                self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes=None):
            outputs = self.model(images)
            if self.export_mode == 'raw':
                return outputs['pred_logits'], outputs['pred_boxes']

            if self.export_mode == 'normalized':
                return _topk_detections(
                    outputs['pred_logits'],
                    outputs['pred_boxes'],
                    self.num_classes,
                    self.num_top_queries,
                )

            outputs = self.postprocessor(outputs, orig_target_sizes)
            if len(outputs) == 4:
                labels, boxes, scores, masks = outputs
                return labels.to(torch.int32), boxes, scores, masks
            labels, boxes, scores = outputs
            return labels.to(torch.int32), boxes, scores

    model = Model()

    img_size = cfg.yaml_cfg["eval_spatial_size"]
    data = torch.rand(args.batch_size, 3, *img_size)
    size = torch.tensor([[img_size[1], img_size[0]]] * args.batch_size, dtype=torch.float32)

    if task == 'segmentation' and args.export_mode != 'pixel':
        raise NotImplementedError(f"--export-mode {args.export_mode} is only implemented for detection.")

    output_file = args.resume.replace('.pth', '.onnx') if args.resume else 'model.onnx'
    if args.export_mode == 'raw':
        input_args = (data,)
        input_names = ['images']
        output_names = ['pred_logits', 'pred_boxes']
    else:
        output_names = ['labels', 'boxes', 'scores'] + (['masks'] if task == 'segmentation' else [])
        if args.export_mode == 'pixel':
            input_args = (data, size)
            input_names = ['images', 'orig_target_sizes']
        else:
            input_args = (data,)
            input_names = ['images']

    dynamic_axes = None
    if not args.static_batch:
        dynamic_axes = {'images': {0: 'N'}}
        if 'orig_target_sizes' in input_names:
            dynamic_axes['orig_target_sizes'] = {0: 'N'}
        dynamic_axes.update({name: {0: 'N'} for name in output_names})

    _ = model(*input_args)
    
    torch.onnx.export(
        model,
        input_args,
        output_file,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
        verbose=False,
        do_constant_folding=True,
        external_data=args.external_data,
        dynamo=False,
    )

    if args.check:
        import onnx
        onnx_model = onnx.load(output_file)
        onnx.checker.check_model(onnx_model)
        print('Check export onnx model done...')

    if args.simplify:
        import onnx
        import onnxsim
        dynamic = True
        input_shapes = {'images': data.shape} if dynamic else None
        if dynamic and 'orig_target_sizes' in input_names:
            input_shapes['orig_target_sizes'] = size.shape
        onnx_model_simplify, check = onnxsim.simplify(output_file, test_input_shapes=input_shapes)
        onnx.save(onnx_model_simplify, output_file, save_as_external_data=args.external_data)
        print(f'Simplify onnx model {check}...')


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', default='configs/dfine/dfine_hgnetv2_l_coco.yml', type=str, )
    parser.add_argument('--resume', '-r', type=str, )
    parser.add_argument('--opset', type=int, default=20,)
    parser.add_argument('--export-mode', choices=['normalized', 'pixel', 'raw'], default='normalized',
                        help='Default normalized exports one images input and labels/normalized_xyxy_boxes/scores outputs.')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Example batch size used during export tracing.')
    parser.add_argument('--static-batch', action='store_true',
                        help='Export a fixed batch dimension instead of dynamic N.')
    parser.add_argument('--check',  action='store_true')
    parser.add_argument('--simplify',  action='store_true')
    parser.add_argument('--external-data', action='store_true',
                        help='Save ONNX weights to an external .data file. Disabled by default.')
    args = parser.parse_args()
    main(args)
