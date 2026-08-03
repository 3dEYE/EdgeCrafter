import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.data.dataloader import (  # noqa: E402
    BaseCollateFunction,
    DataLoader,
    available_cpu_count,
    auto_num_workers,
)
from engine.data.dataset._dataset import DetDataset  # noqa: E402
from engine.core.yaml_config import YAMLConfig  # noqa: E402
from engine.misc import dist_utils  # noqa: E402


class EpochDataset(DetDataset):
    def __len__(self):
        return 4

    def __getitem__(self, index):
        return self.epoch


class EpochCollate(BaseCollateFunction):
    def __call__(self, items):
        return torch.tensor(items), self.epoch


class DataLoaderRuntimeTest(unittest.TestCase):
    def test_yaml_builder_owns_runtime_defaults(self):
        config = YAMLConfig.__new__(YAMLConfig)
        config.yaml_cfg = {
            'train_dataloader': {'batch_size': 8, 'shuffle': True},
        }
        global_cfg = {
            'train_dataloader': {'type': 'DataLoader', 'batch_size': 8},
        }
        fake_loader = SimpleNamespace(shuffle=None)

        with patch.object(
            YAMLConfig, 'global_cfg', new_callable=lambda: property(lambda _: global_cfg)
        ), patch('engine.core.yaml_config.create', return_value=fake_loader) as create_mock:
            loader = config.build_dataloader('train_dataloader')

        runtime_kwargs = create_mock.call_args.kwargs
        self.assertEqual(runtime_kwargs['num_workers'], 'auto')
        self.assertIsNone(runtime_kwargs['max_workers'])
        self.assertTrue(runtime_kwargs['pin_memory'])
        self.assertTrue(runtime_kwargs['persistent_workers'])
        self.assertEqual(runtime_kwargs['prefetch_factor'], 2)
        self.assertTrue(loader.shuffle)

    def test_auto_workers_are_shared_between_local_ranks(self):
        with patch.dict(os.environ, {'LOCAL_WORLD_SIZE': '4'}), patch(
            'engine.data.dataloader.available_cpu_count', return_value=16
        ):
            self.assertEqual(auto_num_workers(), 4)

    def test_auto_workers_do_not_oversubscribe_when_ranks_exceed_cpus(self):
        with patch.dict(os.environ, {'LOCAL_WORLD_SIZE': '8'}), patch(
            'engine.data.dataloader.available_cpu_count', return_value=4
        ):
            self.assertEqual(auto_num_workers(), 0)

    def test_cgroup_quota_limits_available_cpus(self):
        with patch('engine.data.dataloader._cgroup_cpu_count', return_value=2), patch(
            'engine.data.dataloader.os.cpu_count', return_value=32
        ):
            self.assertEqual(available_cpu_count(), 2)

    def test_auto_workers_honor_explicit_runtime_cap(self):
        with patch('engine.data.dataloader.available_cpu_count', return_value=32):
            self.assertEqual(auto_num_workers(max_workers=4), 4)

    def test_zero_workers_disables_multiprocessing_only_options(self):
        loader = DataLoader(
            EpochDataset(),
            num_workers=0,
            persistent_workers=True,
            prefetch_factor=4,
            in_order=False,
            collate_fn=EpochCollate(),
        )
        self.assertFalse(loader.persistent_workers)
        self.assertIsNone(loader.prefetch_factor)

    def test_persistent_workers_observe_epoch_updates(self):
        loader = DataLoader(
            EpochDataset(),
            batch_size=2,
            num_workers=2,
            persistent_workers=True,
            prefetch_factor=2,
            collate_fn=EpochCollate(),
        )
        try:
            loader.set_epoch(3)
            dataset_epochs, collate_epoch = next(iter(loader))
            self.assertEqual(dataset_epochs.tolist(), [3, 3])
            self.assertEqual(collate_epoch, 3)

            loader.set_epoch(7)
            dataset_epochs, collate_epoch = next(iter(loader))
            self.assertEqual(dataset_epochs.tolist(), [7, 7])
            self.assertEqual(collate_epoch, 7)
        finally:
            if loader._iterator is not None:
                loader._iterator._shutdown_workers()

    def test_distributed_wrapper_preserves_runtime_options(self):
        loader = DataLoader(
            EpochDataset(),
            batch_size=2,
            num_workers=1,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
            in_order=False,
            collate_fn=EpochCollate(),
        )
        with patch.object(dist_utils, 'is_dist_available_and_initialized', return_value=True), \
             patch.object(torch.distributed, 'get_world_size', return_value=1), \
             patch.object(torch.distributed, 'get_rank', return_value=0):
            wrapped = dist_utils.warp_loader(loader, shuffle=True)

        self.assertEqual(wrapped.num_workers, 1)
        self.assertTrue(wrapped.pin_memory)
        self.assertTrue(wrapped.persistent_workers)
        self.assertEqual(wrapped.prefetch_factor, 4)
        self.assertEqual(wrapped.in_order, loader.in_order)
        self.assertIs(wrapped.worker_init_fn, loader.worker_init_fn)


if __name__ == '__main__':
    unittest.main()
