"""
EdgeCrafter: Compact ViTs for Edge Dense Prediction via Task-Specialized Distillation
Copyright (c) 2026 The EdgeCrafter Authors. All Rights Reserved.
---------------------------------------------------------------------------------
DEIMv2: Real-Time Object Detection Meets DINOv3
Copyright (c) 2025 The DEIMv2 Authors. All Rights Reserved.
---------------------------------------------------------------------------------
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""
import json
import os
import shlex
import sys
import warnings
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import argparse
import torch.distributed as distributed

from engine.core import YAMLConfig, yaml_utils
from engine.misc import dist_utils
from engine.solver import TASKS

debug=False

warnings.filterwarnings("ignore")
warnings.filterwarnings("always", message=".*stop_epoch.*")
# Dataset problems must not be swallowed by the blanket ignore above, otherwise a
# run on broken annotations looks clean.
warnings.filterwarnings("always", message="(?i).*invalid label.*")

if debug:
    import torch
    def custom_repr(self):
        return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'
    original_repr = torch.Tensor.__repr__
    torch.Tensor.__repr__ = custom_repr


def save_run_args(args, cfg) -> None:
    command_argv = getattr(sys, 'orig_argv', None) or [sys.executable, *sys.argv]
    run = {
        'started_at': datetime.now().astimezone().isoformat(timespec='seconds'),
        'working_directory': os.getcwd(),
        'command': shlex.join(command_argv),
        'argv': list(sys.argv),
        'args': vars(args),
        'resolved_config': cfg.yaml_cfg,
        'distributed': {
            'world_size': dist_utils.get_world_size(),
            'local_world_size': os.environ.get('LOCAL_WORLD_SIZE'),
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        },
    }

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    args_path = output_dir / 'args.json'
    metadata = {'runs': []}
    if args_path.exists():
        try:
            with args_path.open('r', encoding='utf-8') as f:
                metadata = json.load(f)
            if not isinstance(metadata, dict) or not isinstance(metadata.get('runs'), list):
                raise ValueError('expected an object with a runs array')
        except (json.JSONDecodeError, ValueError) as exc:
            backup_name = f'args.invalid-{datetime.now():%Y%m%dT%H%M%S%f}.json'
            backup_path = args_path.with_name(backup_name)
            args_path.replace(backup_path)
            print(f'Warning: invalid {args_path} was moved to {backup_path}: {exc}')
            metadata = {'runs': []}

    metadata['runs'].append(run)
    temp_path = args_path.with_suffix('.json.tmp')
    with temp_path.open('w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2, default=str)
        f.write('\n')
    temp_path.replace(args_path)


def reserve_output_dir(output_dir: str) -> Path:
    """Create and return an unused output directory.

    The requested name is used for the first run. Later runs receive the first
    available ``_vN`` suffix, starting with ``_v2``. Creating the directory
    with ``exist_ok=False`` also prevents concurrent launches from selecting
    the same path.
    """
    base_path = Path(output_dir)
    candidate = base_path
    version = 2

    while True:
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            # A parent component may be a file. In that case changing only the
            # final directory name cannot help, so preserve the original error
            # instead of looping forever.
            if not os.path.lexists(candidate):
                raise
            candidate = base_path.with_name(f'{base_path.name}_v{version}')
            version += 1


def configure_output_dir(cfg) -> None:
    """Reserve a run directory and share its path with all distributed ranks."""
    requested_path = str(cfg.output_dir)
    selected_path = None

    if dist_utils.is_main_process():
        selected_path = str(reserve_output_dir(requested_path))

    if dist_utils.is_dist_available_and_initialized():
        selected_paths = [selected_path]
        distributed.broadcast_object_list(selected_paths, src=0)
        selected_path = selected_paths[0]

    cfg.output_dir = selected_path
    cfg.yaml_cfg['output_dir'] = selected_path

    if Path(selected_path) != Path(requested_path):
        print(f'Output directory already exists; using {selected_path}')


def main(args, ) -> None:
    """main
    """
    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)

    assert not all([args.tuning, args.resume]), \
        'Only support from_scrach or resume or tuning at one time'


    update_dict = yaml_utils.parse_cli(args.update) # update cfg from command line
    update_dict.update({k: v for k, v in args.__dict__.items() \
        if k not in ['update', ] and v is not None})

    cfg = YAMLConfig(args.config, **update_dict)

    if cfg.input_dtype == 'float16' and not cfg.use_amp:
        raise ValueError(
            'float16 input requires AMP; keep the defaults or use '
            '--no-amp together with --input-dtype float32'
        )

    if args.resume or args.tuning:
        if 'ViTAdapter' in cfg.yaml_cfg:
            cfg.yaml_cfg['ViTAdapter']['skip_load_backbone'] = True

    # Resume training in place because ECSolver may need best.pth from the
    # existing run at the augmentation-stage boundary. Evaluation is safe to
    # version even when it loads its weights through --resume.
    if not args.resume or args.test_only:
        configure_output_dir(cfg)

    print('cfg: ', cfg.__dict__)

    if not args.test_only and dist_utils.is_main_process():
        try:
            save_run_args(args, cfg)
        except (OSError, TypeError, ValueError) as exc:
            print(f'Warning: could not save training arguments: {exc}')

    solver = TASKS[cfg.yaml_cfg['task']](cfg)

    if args.test_only:
        solver.val()
    else:
        solver.fit()

    dist_utils.cleanup()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # priority 0
    parser.add_argument('-c', '--config', type=str, default='')
    parser.add_argument('-r', '--resume', type=str, help='resume from checkpoint')
    parser.add_argument('-t', '--tuning', type=str, help='tuning from checkpoint')
    parser.add_argument('-d', '--device', type=str, help='device',)
    parser.add_argument('--seed', type=int, default=0, help='exp reproducibility')
    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument('--use-amp', dest='use_amp', action='store_true', help='enable mixed precision training (default)')
    amp_group.add_argument('--no-amp', dest='use_amp', action='store_false', help='disable mixed precision training')
    parser.set_defaults(use_amp=True)
    parser.add_argument(
        '--input-dtype',
        choices=['float32', 'float16'],
        default='float16',
        help='Image dtype used at the training device boundary (default: float16).',
    )
    parser.add_argument('--output-dir', type=str, help='output directoy')
    parser.add_argument('--summary-dir', type=str, help='tensorboard summry')
    parser.add_argument('--test-only', action='store_true', default=False,)

    # priority 1
    parser.add_argument('-u', '--update', nargs='+', help='update yaml config')

    # env
    parser.add_argument('--print-method', type=str, default='builtin', help='print method')
    parser.add_argument('--print-rank', type=int, default=0, help='print rank id')

    parser.add_argument('--local-rank', type=int, help='local rank id')
    args = parser.parse_args()

    main(args)
