"""
EdgeCrafter: Compact ViTs for Edge Dense Prediction via Task-Specialized Distillation
Copyright (c) 2026 The EdgeCrafter Authors. All Rights Reserved.
---------------------------------------------------------------------------------
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright (c) 2024 D-FINE authors. All Rights Reserved.
"""
import math
import multiprocessing as mp
import os
import random
from collections import defaultdict, deque
from copy import deepcopy
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
import torchvision.transforms.v2 as VT
from PIL import Image, ImageDraw
from torch.utils.data import default_collate
from torchvision.transforms.v2 import InterpolationMode
from torchvision.transforms.v2 import functional as VF

from ..core import register

torchvision.disable_beta_transforms_warning()


__all__ = [
    'DataLoader',
    'BaseCollateFunction',
    'BatchImageCollateFunction',
    'batch_image_collate_fn'
]


def _cgroup_cpu_count():
    """Return a cgroup CPU quota as an integer worker budget when present."""
    quota_files = (
        (Path('/sys/fs/cgroup/cpu.max'), None),
        (
            Path('/sys/fs/cgroup/cpu/cpu.cfs_quota_us'),
            Path('/sys/fs/cgroup/cpu/cpu.cfs_period_us'),
        ),
    )
    for quota_path, period_path in quota_files:
        try:
            if period_path is None:
                quota_text, period_text = quota_path.read_text().split()[:2]
                if quota_text == 'max':
                    continue
            else:
                quota_text = quota_path.read_text().strip()
                period_text = period_path.read_text().strip()
            quota = int(quota_text)
            period = int(period_text)
            if quota > 0 and period > 0:
                return max(1, math.ceil(quota / period))
        except (FileNotFoundError, OSError, ValueError):
            continue
    return None


def available_cpu_count():
    """Return the CPUs this process is allowed to use.

    ``sched_getaffinity`` respects Linux container/cpuset limits.  Newer Python
    versions expose the same intent through ``process_cpu_count``; ``cpu_count``
    remains the portable fallback.
    """
    candidates = []
    if hasattr(os, 'sched_getaffinity'):
        try:
            candidates.append(len(os.sched_getaffinity(0)))
        except (OSError, TypeError):
            pass

    process_cpu_count = getattr(os, 'process_cpu_count', None)
    count = process_cpu_count() if process_cpu_count is not None else os.cpu_count()
    if count:
        candidates.append(count)
    cgroup_count = _cgroup_cpu_count()
    if cgroup_count:
        candidates.append(cgroup_count)
    return max(1, min(candidates, default=1))


def auto_num_workers(max_workers=None):
    """Use available CPUs, shared evenly by local distributed ranks.

    Windows uses spawn rather than fork, so every worker imports its own copy of
    the Python/PyTorch stack. Keep its automatic default bounded; an explicit
    integer ``num_workers`` remains available for deliberate overrides.
    """
    try:
        local_world_size = max(1, int(os.environ.get('LOCAL_WORLD_SIZE', '1')))
    except ValueError:
        local_world_size = 1
    workers = available_cpu_count() // local_world_size
    platform_limit = 8 if os.name == 'nt' else None
    limits = [limit for limit in (platform_limit, max_workers) if limit is not None]
    return min(workers, *limits) if limits else workers


class _WorkerInitializer:
    """Prevent every DataLoader process from creating another CPU thread pool."""

    def __init__(self, user_init_fn=None):
        self.user_init_fn = user_init_fn

    def __call__(self, worker_id):
        torch.set_num_threads(1)
        try:
            import cv2
            cv2.setNumThreads(1)
        except ImportError:
            pass

        if self.user_init_fn is not None:
            self.user_init_fn(worker_id)


@register()
class DataLoader(data.DataLoader):
    __inject__ = ['dataset', 'collate_fn']

    def __init__(self, dataset, batch_size=1, shuffle=None, sampler=None,
                 batch_sampler=None, num_workers=0, collate_fn=None,
                 pin_memory=False, drop_last=False, timeout=0,
                 worker_init_fn=None, multiprocessing_context=None,
                 generator=None, *, prefetch_factor=None,
                 persistent_workers=False, pin_memory_device='', in_order=True,
                 max_workers=None):
        requested_workers = num_workers
        if isinstance(num_workers, str):
            if num_workers.lower() != 'auto':
                raise ValueError("num_workers must be an integer or 'auto'")
            num_workers = auto_num_workers(max_workers=max_workers)

        num_workers = int(num_workers)
        if num_workers < 0:
            raise ValueError('num_workers must be non-negative')

        if num_workers == 0:
            persistent_workers = False
            prefetch_factor = None
            multiprocessing_context = None
        else:
            if isinstance(worker_init_fn, _WorkerInitializer):
                constrained_worker_init_fn = worker_init_fn
            else:
                constrained_worker_init_fn = _WorkerInitializer(worker_init_fn)
            worker_init_fn = constrained_worker_init_fn

        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            multiprocessing_context=multiprocessing_context,
            generator=generator,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            pin_memory_device=pin_memory_device,
            in_order=in_order,
        )

        if isinstance(requested_workers, str):
            print(
                f'DataLoader num_workers=auto resolved to {num_workers} '
                f'(available_cpus={available_cpu_count()}, '
                f'local_world_size={os.environ.get("LOCAL_WORLD_SIZE", "1")}, '
                f'max_workers={max_workers})'
            )

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for n in ['dataset', 'batch_size', 'num_workers', 'drop_last', 'pin_memory',
                  'persistent_workers', 'prefetch_factor', 'collate_fn']:
            format_string += "\n"
            format_string += "    {0}: {1}".format(n, getattr(self, n))
        format_string += "\n)"
        return format_string

    def set_epoch(self, epoch):
        self._epoch = epoch
        self.dataset.set_epoch(epoch)
        self.collate_fn.set_epoch(epoch)

    @property
    def epoch(self):
        return self._epoch if hasattr(self, '_epoch') else -1

    @property
    def shuffle(self):
        return self._shuffle

    @shuffle.setter
    def shuffle(self, shuffle):
        assert isinstance(shuffle, bool), 'shuffle must be a boolean'
        self._shuffle = shuffle


@register()
def batch_image_collate_fn(items):
    """only batch image
    """
    return torch.cat([x[0][None] for x in items], dim=0), [x[1] for x in items]


class BaseCollateFunction(object):
    def set_epoch(self, epoch):
        if not hasattr(self, '_epoch_shared'):
            self._epoch_shared = mp.Value('q', int(epoch), lock=False)
        else:
            self._epoch_shared.value = int(epoch)

    @property
    def epoch(self):
        return self._epoch_shared.value if hasattr(self, '_epoch_shared') else -1

    def __call__(self, items):
        raise NotImplementedError('')


@register() 
class BatchImageCollateFunction(BaseCollateFunction):
    def __init__(
        self, 
        mixup_prob=0.0,
        mixup_epoch=0,
    ) -> None:
        super().__init__()
        self.mixup_prob, self.mixup_epoch = mixup_prob, mixup_epoch

    def apply_mixup(self, images, targets):
        """
        Applies Mixup augmentation to the batch if conditions are met.

        Args:
            images (torch.Tensor): Batch of images.
            targets (list[dict]): List of target dictionaries corresponding to images.

        Returns:
            tuple: Updated images and targets
        """
        beta = round(random.uniform(0.45, 0.55), 6)
        # Apply Mixup if within specified epoch range and probability threshold
        if random.random() < self.mixup_prob and self.epoch < self.mixup_epoch:
            # Generate mixup ratio
            beta = round(random.uniform(0.45, 0.55), 6)

            # Mix images
            images = images.roll(shifts=1, dims=0).mul_(1.0 - beta).add_(images.mul(beta))

            # Prepare targets for Mixup
            shifted_targets = targets[-1:] + targets[:-1]
            updated_targets = deepcopy(targets)

            for i in range(len(targets)):
                # Combine boxes, labels, and areas from original and shifted targets
                updated_targets[i]['boxes'] = torch.cat([targets[i]['boxes'], shifted_targets[i]['boxes']], dim=0)
                updated_targets[i]['labels'] = torch.cat([targets[i]['labels'], shifted_targets[i]['labels']], dim=0)
                updated_targets[i]['area'] = torch.cat([targets[i]['area'], shifted_targets[i]['area']], dim=0)
                if 'masks' in targets[i]:
                    updated_targets[i]['masks'] = torch.cat([targets[i]['masks'], shifted_targets[i]['masks']], dim=0)

                # Add mixup ratio to targets
                updated_targets[i]['mixup'] = torch.tensor(
                    [beta] * len(targets[i]['labels']) + [1.0 - beta] * len(shifted_targets[i]['labels']), 
                    dtype=torch.float32
                    )
            targets = updated_targets
            
        return images, targets

    def __call__(self, items):
        images = torch.cat([x[0][None] for x in items], dim=0)
        targets = [x[1] for x in items]
        images, targets = self.apply_mixup(images, targets)

        return images, targets
