"""
EdgeCrafter: Compact ViTs for Edge Dense Prediction via Task-Specialized Distillation
Copyright (c) 2026 The EdgeCrafter Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import copy
import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader

from ._config import BaseConfig
from .workspace import create
from .yaml_utils import load_config, merge_config, merge_dict


class YAMLConfig(BaseConfig):
    def __init__(self, cfg_path: str, **kwargs) -> None:
        super().__init__()

        cfg = load_config(cfg_path)
        cfg = merge_dict(cfg, kwargs)
        self.resolve_yolo_num_classes(cfg)

        self.yaml_cfg = copy.deepcopy(cfg)
        self.reset_cfg()

        for k in super().__dict__:
            if not k.startswith('_') and k in cfg:
                self.__dict__[k] = cfg[k]

    @property
    def global_cfg(self, ):
        return merge_config(self.yaml_cfg, inplace=False, overwrite=False)

    @property
    def model(self, ) -> torch.nn.Module:
        if self._model is None and 'model' in self.yaml_cfg:
            self._model = create(self.yaml_cfg['model'], self.global_cfg)
        return super().model

    @property
    def postprocessor(self, ) -> torch.nn.Module:
        if self._postprocessor is None and 'postprocessor' in self.yaml_cfg:
            self._postprocessor = create(self.yaml_cfg['postprocessor'], self.global_cfg)
        return super().postprocessor

    @property
    def criterion(self, ) -> torch.nn.Module:
        if self._criterion is None and 'criterion' in self.yaml_cfg:
            self._criterion = create(self.yaml_cfg['criterion'], self.global_cfg)
        return super().criterion

    @property
    def optimizer(self, ) -> optim.Optimizer:
        if self._optimizer is None and 'optimizer' in self.yaml_cfg:
            params = self.get_optim_params(self.yaml_cfg['optimizer'], self.model)
            self._optimizer = create('optimizer', self.global_cfg, params=params)
        return super().optimizer

    @property
    def lr_scheduler(self, ) -> optim.lr_scheduler.LRScheduler:
        if self._lr_scheduler is None and 'lr_scheduler' in self.yaml_cfg:
            self._lr_scheduler = create('lr_scheduler', self.global_cfg, optimizer=self.optimizer)
        return super().lr_scheduler

    @property
    def lr_warmup_scheduler(self, ) -> optim.lr_scheduler.LRScheduler:
        if self._lr_warmup_scheduler is None and 'lr_warmup_scheduler' in self.yaml_cfg :
            self._lr_warmup_scheduler = create('lr_warmup_scheduler', self.global_cfg, lr_scheduler=self.lr_scheduler)
        return super().lr_warmup_scheduler

    @property
    def train_dataloader(self, ) -> DataLoader:
        if self._train_dataloader is None and 'train_dataloader' in self.yaml_cfg:
            self._train_dataloader = self.build_dataloader('train_dataloader')
        return super().train_dataloader

    @property
    def val_dataloader(self, ) -> DataLoader:
        if self._val_dataloader is None and 'val_dataloader' in self.yaml_cfg:
            self._val_dataloader = self.build_dataloader('val_dataloader')
        return super().val_dataloader

    @property
    def ema(self, ) -> torch.nn.Module:
        if self._ema is None and self.yaml_cfg.get('use_ema', False):
            self._ema = create('ema', self.global_cfg, model=self.model)
        return super().ema

    @property
    def scaler(self, ):
        if self._scaler is None and self.yaml_cfg.get('use_amp', False):
            self._scaler = create('scaler', self.global_cfg)
        return super().scaler

    @property
    def evaluator(self, ):
        if self._evaluator is None and 'evaluator' in self.yaml_cfg:
            if self.yaml_cfg['evaluator']['type'] == 'CocoEvaluator':
                from ..data import get_coco_api_from_dataset
                base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
                self._evaluator = create('evaluator', self.global_cfg, coco_gt=base_ds)
            else:
                raise NotImplementedError(f"{self.yaml_cfg['evaluator']['type']}")
        return super().evaluator

    @staticmethod
    def get_optim_params(cfg: dict, model: nn.Module):
        """
        E.g.:
            ^(?=.*a)(?=.*b).*$  means including a and b
            ^(?=.*(?:a|b)).*$   means including a or b
            ^(?=.*a)(?!.*b).*$  means including a, but not b
        """
        assert 'type' in cfg, ''
        cfg = copy.deepcopy(cfg)

        if 'params' not in cfg:
            return model.parameters()

        assert isinstance(cfg['params'], list), ''

        param_groups = []
        visited = []
        for pg in cfg['params']:
            pattern = pg['params']
            params = {k: v for k, v in model.named_parameters() if v.requires_grad and len(re.findall(pattern, k)) > 0}
            pg['params'] = params.values()
            param_groups.append(pg)
            visited.extend(list(params.keys()))
            # print(params.keys())

        names = [k for k, v in model.named_parameters() if v.requires_grad]

        if len(visited) < len(names):
            unseen = set(names) - set(visited)
            params = {k: v for k, v in model.named_parameters() if v.requires_grad and k in unseen}
            param_groups.append({'params': params.values()})
            visited.extend(list(params.keys()))
            # print(params.keys())

        assert len(visited) == len(names), ''

        return param_groups

    @staticmethod
    def get_rank_batch_size(cfg):
        """compute batch size for per rank if total_batch_size is provided.
        """
        assert ('total_batch_size' in cfg or 'batch_size' in cfg) \
            and not ('total_batch_size' in cfg and 'batch_size' in cfg), \
                '`batch_size` or `total_batch_size` should be choosed one'

        total_batch_size = cfg.get('total_batch_size', None)
        if total_batch_size is None:
            bs = cfg.get('batch_size')
        else:
            from ..misc import dist_utils
            assert total_batch_size % dist_utils.get_world_size() == 0, \
                'total_batch_size should be divisible by world size'
            bs = total_batch_size // dist_utils.get_world_size()
        return bs

    def build_dataloader(self, name: str):
        bs = self.get_rank_batch_size(self.yaml_cfg[name])
        global_cfg = self.global_cfg
        if 'total_batch_size' in global_cfg[name]:
            # pop unexpected key for dataloader init
            _ = global_cfg[name].pop('total_batch_size')
        # Runtime policy belongs to the loader builder rather than dataset
        # descriptions. Explicit command-line/config overrides still win.
        runtime_defaults = {
            'num_workers': 'auto',
            'max_workers': None if name == 'train_dataloader' else 4,
            'pin_memory': True,
            'persistent_workers': True,
            # Torch's default. Anything lower starves the GPU whenever a batch
            # takes longer to build than the previous step takes to run.
            'prefetch_factor': 2,
        }
        loader_cfg = global_cfg[name]
        runtime_kwargs = {
            key: loader_cfg.get(key, value)
            for key, value in runtime_defaults.items()
        }
        loader = create(name, global_cfg, batch_size=bs, **runtime_kwargs)
        loader.shuffle = self.yaml_cfg[name].get('shuffle', False)
        return loader

    @staticmethod
    def resolve_yolo_num_classes(cfg: dict):
        yolo_root_value = cfg.get('yolo_root', None)
        yolo_root = None
        if yolo_root_value is not None:
            if not isinstance(yolo_root_value, (str, Path)):
                raise TypeError(
                    'yolo_root must be a filesystem path, '
                    f'got {type(yolo_root_value).__name__}'
                )

            yolo_root_text = str(yolo_root_value).strip()
            if not yolo_root_text:
                raise ValueError('yolo_root must not be empty')
            if yolo_root_text.startswith('='):
                raise ValueError(
                    f"Invalid yolo_root={yolo_root_value!r}: the path starts with '='. "
                    "When overriding it from the command line, use "
                    "`yolo_root=/path/to/dataset` with a single '='."
                )

            yolo_root = Path(yolo_root_text).expanduser()
            if not yolo_root.exists():
                raise FileNotFoundError(
                    f"YOLO dataset root does not exist: {yolo_root} "
                    f"(from yolo_root={yolo_root_value!r}). "
                    'Check the path and make sure the dataset is mounted.'
                )
            if not yolo_root.is_dir():
                raise NotADirectoryError(
                    f"YOLO dataset root is not a directory: {yolo_root} "
                    f"(from yolo_root={yolo_root_value!r})"
                )

        data_file = cfg.get('yolo_data_file', None)
        if data_file is not None:
            data_path = Path(data_file).expanduser()
            if not data_path.exists():
                raise FileNotFoundError(
                    f'YOLO data file does not exist: {data_path} '
                    f'(from yolo_data_file={data_file!r})'
                )
            if not data_path.is_file():
                raise IsADirectoryError(
                    f'YOLO data file is not a file: {data_path} '
                    f'(from yolo_data_file={data_file!r})'
                )

        if cfg.get('num_classes', None) is not None:
            return

        if data_file is None and yolo_root is not None:
            for name in ('data.yaml', 'data.yml', 'data_win.yaml', 'dataset.yaml', 'dataset.yml'):
                candidate = yolo_root / name
                if candidate.exists():
                    data_file = candidate
                    cfg['yolo_data_file'] = str(candidate)
                    break

        if data_file is None:
            for loader_name in ('train_dataloader', 'val_dataloader'):
                dataset_cfg = cfg.get(loader_name, {}).get('dataset', {})
                if dataset_cfg.get('type') == 'YOLODetection':
                    data_file = dataset_cfg.get('data_file', None)
                    if data_file is not None:
                        break

        if data_file is None:
            if yolo_root is not None:
                raise FileNotFoundError(
                    f'Could not infer num_classes for the YOLO dataset at {yolo_root}: '
                    'no data.yaml, data.yml, data_win.yaml, dataset.yaml, or dataset.yml '
                    'was found. Add one of these files, pass '
                    '`yolo_data_file=/path/to/data.yaml`, or set `num_classes` explicitly.'
                )
            return

        data_path = Path(data_file).expanduser()
        # Dataset-level data_file entries and files discovered under yolo_root
        # reach this branch after the explicit global yolo_data_file check above.
        if not data_path.exists():
            raise FileNotFoundError(f'YOLO data file does not exist: {data_path}')
        if not data_path.is_file():
            raise IsADirectoryError(f'YOLO data file is not a file: {data_path}')

        with data_path.open('r', encoding='utf-8') as f:
            data_cfg = yaml.safe_load(f) or {}

        num_classes = data_cfg.get('nc', None)
        if num_classes is None:
            names = data_cfg.get('names', None)
            if isinstance(names, (list, tuple, dict)):
                num_classes = len(names)

        if num_classes is None:
            raise ValueError(f"Can not infer num_classes from YOLO data file: {data_file}")

        cfg['num_classes'] = int(num_classes)

    def reset_cfg(self):
        """reset tranforms size according to input size, and check stop_epoch for training transforms.
        """
        input_size = self.yaml_cfg['eval_spatial_size'][0]

        def simple_glom(data, path):
            for key in path.split("."):
                data = data[key]
            return data

        train_ops = simple_glom(self.yaml_cfg, 'train_dataloader.dataset.transforms.ops')
        val_ops = simple_glom(self.yaml_cfg, 'val_dataloader.dataset.transforms.ops')

        for ops in [train_ops, val_ops]:
            for op in ops:
                t = op.get("type")
                if t == "Mosaic":
                    op["output_size"] = input_size // 2
                elif t == "Resize":
                    op["size"] = (input_size, input_size)

        stop_aug_epoch = simple_glom(self.yaml_cfg, 'train_dataloader.dataset.transforms.stop_epoch')
        epochs = self.yaml_cfg['epochs']
        no_aug_epoch = epochs - stop_aug_epoch
        if not 0 <= no_aug_epoch <= 5:
            self.yaml_cfg['train_dataloader']['dataset']['transforms']['stop_epoch'] = epochs - 2

            import warnings
            warnings.warn(
                "'stop_epoch' was not correctly set for training transforms. "
                "Automatically adjusted to: epochs - 2.",
                UserWarning)
