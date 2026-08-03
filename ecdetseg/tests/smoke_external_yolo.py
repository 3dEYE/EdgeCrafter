import argparse
import sys
import time
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.core import YAMLConfig  # noqa: E402


# Entries with one row per object; they must stay aligned with `labels` through
# Mosaic, SanitizeBoundingBoxes and the mixup merge.
PER_OBJECT_KEYS = ('boxes', 'area', 'iscrowd', 'masks')
# Entries describing the sample as a whole. Mosaic merges four samples, so these
# must not grow to four ids / four size pairs.
IMAGE_LEVEL_NUMEL = {'image_id': 1, 'idx': 1, 'orig_size': 2, 'size': 2}


def _check_batch(batch, expected_batch_size):
    images, targets = batch
    assert images.shape == (expected_batch_size, 3, 640, 640), images.shape
    assert torch.isfinite(images).all()
    assert len(targets) == expected_batch_size
    for target in targets:
        assert {'boxes', 'labels', 'area', 'orig_size'} <= target.keys()
        assert target['boxes'].ndim == 2 and target['boxes'].shape[-1] == 4
        assert target['labels'].ndim == 1
        assert torch.isfinite(target['boxes']).all()

        num_objects = target['labels'].shape[0]
        for key in PER_OBJECT_KEYS:
            if key in target:
                assert target[key].shape[0] == num_objects, (key, target[key].shape, num_objects)
        for key, numel in IMAGE_LEVEL_NUMEL.items():
            if key in target:
                assert target[key].numel() == numel, (key, target[key].shape)
    return images, targets


def _mixup_merge_trials(collate_fn, items, trials=32):
    """Count how many collate calls merged each sample with its neighbour.

    The collate function no longer tags mixed targets, so the merge is detected by
    its only observable effect: every sample gains the previous sample's objects.
    """
    baseline = [target['labels'].shape[0] for _, target in items]
    if sum(baseline) == 0:
        return None
    expected = [baseline[i] + baseline[i - 1] for i in range(len(baseline))]

    merged = 0
    for _ in range(trials):
        _, targets = collate_fn(items)
        counts = [target['labels'].shape[0] for target in targets]
        if counts == baseline:
            continue
        assert counts == expected, (counts, baseline, expected)
        for i, target in enumerate(targets):
            # Object counts alone cannot tell the shift direction apart when every
            # sample holds the same number of objects, so compare the boxes: the
            # merged target must be own rows followed by the previous sample's.
            own, previous = items[i][1]['boxes'], items[i - 1][1]['boxes']
            assert torch.equal(target['boxes'][:own.shape[0]], own), i
            assert torch.equal(target['boxes'][own.shape[0]:], previous), i
            for key in PER_OBJECT_KEYS:
                if key in target:
                    assert target[key].shape[0] == target['labels'].shape[0], key
        merged += 1
    return merged


def _worker_pids(loader):
    return tuple(worker.pid for worker in loader._iterator._workers)


def _shutdown(loader):
    if loader is not None and loader._iterator is not None:
        loader._iterator._shutdown_workers()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_root', type=Path)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--batches', type=int, default=8)
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    data_file = dataset_root / 'data.yaml'
    config_path = PROJECT_ROOT / 'configs' / 'ecdet' / 'ecdet_m.yml'
    dataset_override = {
        'type': 'YOLODetection',
        'validate_on_init': False,
        'strict': True,
    }
    config = YAMLConfig(
        str(config_path),
        yolo_root=str(dataset_root),
        yolo_data_file=str(data_file),
        num_classes=7,
        train_dataloader={
            'total_batch_size': args.batch_size,
            'dataset': {**dataset_override, 'split': 'train'},
        },
        val_dataloader={
            'total_batch_size': args.batch_size,
            'dataset': {**dataset_override, 'split': 'val'},
        },
    )

    train_loader = None
    val_loader = None
    try:
        train_loader = config.train_dataloader
        print(
            'TRAIN SETTINGS:',
            f'workers={train_loader.num_workers}',
            f'pin_memory={train_loader.pin_memory}',
            f'persistent_workers={train_loader.persistent_workers}',
            f'prefetch_factor={train_loader.prefetch_factor}',
        )

        train_loader.set_epoch(0)
        start = time.perf_counter()
        iterator = iter(train_loader)
        batches = []
        for _ in range(args.batches):
            try:
                batch = next(iterator)
            except StopIteration:  # dataset shorter than --batches
                break
            batches.append(_check_batch(batch, args.batch_size))
        assert batches, 'no batches produced'
        elapsed = time.perf_counter() - start
        first_epoch_pids = _worker_pids(train_loader)
        # Un-collated samples, so the object counts are the pre-mixup baseline.
        items = [train_loader.dataset[i] for i in range(args.batch_size)]
        collate_cfg = config.yaml_cfg['train_dataloader']['collate_fn']
        mixup_prob = collate_cfg.get('mixup_prob', 0.0)
        mixup_on = _mixup_merge_trials(train_loader.collate_fn, items)
        if mixup_prob > 0:
            assert mixup_on is None or mixup_on > 0, 'mixup never fired before mixup_epoch'
        print(
            'TRAIN EPOCH 0:',
            f'batches={len(batches)}',
            f'mixup_merges={mixup_on if mixup_on is not None else "n/a (no objects)"}/32',
            f'elapsed={elapsed:.3f}s',
            f'throughput={args.batch_size * len(batches) / elapsed:.2f} samples/s',
        )

        # Mixup is off from `mixup_epoch` onwards; read it instead of hardcoding an
        # epoch that silently stops matching when the recipe is rescaled.
        mixup_epoch = collate_cfg['mixup_epoch']
        train_loader.set_epoch(mixup_epoch)
        images, _ = _check_batch(next(iter(train_loader)), args.batch_size)
        second_epoch_pids = _worker_pids(train_loader)
        assert first_epoch_pids == second_epoch_pids, (first_epoch_pids, second_epoch_pids)
        mixup_off = _mixup_merge_trials(train_loader.collate_fn, items)
        assert mixup_off in (0, None), f'mixup still fired at epoch {mixup_epoch}: {mixup_off}'
        print(
            f'TRAIN EPOCH {mixup_epoch}:',
            'persistent_worker_pids_unchanged=True',
            'mixup_disabled=True',
            f'images_pinned={images.is_pinned()}',
        )

        val_loader = config.val_dataloader
        val_images, _ = _check_batch(next(iter(val_loader)), args.batch_size)
        print(
            'VAL:',
            f'workers={val_loader.num_workers}',
            f'prefetch_factor={val_loader.prefetch_factor}',
            f'images_pinned={val_images.is_pinned()}',
        )
    finally:
        _shutdown(train_loader)
        _shutdown(val_loader)


if __name__ == '__main__':
    main()
