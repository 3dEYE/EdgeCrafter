"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import multiprocessing as mp

import torch
import torch.utils.data as data


class DetDataset(data.Dataset):
    def __getitem__(self, index):
        img, target = self.load_item(index)
        if self.transforms is not None:
            img, target, _ = self.transforms(img, target, self)
        return img, target

    def load_item(self, index):
        raise NotImplementedError("Please implement this function to return item before `transforms`.")

    def set_epoch(self, epoch) -> None:
        if not hasattr(self, '_epoch_shared'):
            self._epoch_shared = mp.Value('q', int(epoch), lock=False)
        else:
            self._epoch_shared.value = int(epoch)

    @property
    def epoch(self):
        return self._epoch_shared.value if hasattr(self, '_epoch_shared') else -1
