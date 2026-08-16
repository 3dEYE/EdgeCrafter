import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.solver.ec_engine import train_one_epoch  # noqa: E402


class RecordingSGD(torch.optim.SGD):
    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)
        self.step_lrs = []

    def step(self, closure=None):
        self.step_lrs.append(self.param_groups[0]['lr'])
        return super().step(closure)


class ZeroFirstStepScheduler:
    def __init__(self):
        self.steps = []

    def step(self, current_iter, optimizer):
        self.steps.append(current_iter)
        for group in optimizer.param_groups:
            group['lr'] = 0.0
        return optimizer


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, samples, targets=None):
        return {'value': samples * self.weight}


class TinyCriterion(torch.nn.Module):
    def forward(self, outputs, targets, **metas):
        return {'loss_test': outputs['value'].sum()}


class TrainOneEpochSchedulerTest(unittest.TestCase):
    def test_self_scheduler_sets_lr_before_first_optimizer_step(self):
        model = TinyModel()
        optimizer = RecordingSGD(model.parameters(), lr=1.0)
        scheduler = ZeroFirstStepScheduler()
        data_loader = [(torch.ones(1), [{}])]

        train_one_epoch(
            True,
            scheduler,
            model,
            TinyCriterion(),
            data_loader,
            optimizer,
            torch.device('cpu'),
            epoch=0,
            print_freq=10,
            input_dtype='float32',
        )

        self.assertEqual(scheduler.steps, [0])
        self.assertEqual(optimizer.step_lrs, [0.0])
        self.assertEqual(model.weight.item(), 1.0)


if __name__ == '__main__':
    unittest.main()
