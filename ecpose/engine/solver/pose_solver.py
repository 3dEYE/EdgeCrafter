import datetime
import json
import logging
import math
import time

import torch

from ..misc import dist_utils, stats
from ..misc.metrics import BestMetricHolder
from ..optim.lr_scheduler import FlatCosineLRScheduler
from ._solver import BaseSolver
from .pose_engine import evaluate, train_one_epoch


def safe_barrier():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
    else:
        pass

def safe_get_rank():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    else:
        return 0


def _metric_value(evaluation_stats, metric_name):
    if metric_name not in evaluation_stats:
        available = ", ".join(sorted(evaluation_stats))
        raise KeyError(f"Primary metric {metric_name!r} is missing; available metrics: {available}")
    value = evaluation_stats[metric_name]
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError(f"Primary metric {metric_name!r} is empty")
        value = value[0]
    return float(value)


def _is_better(value, best_value, mode):
    if not math.isfinite(value):
        return False
    return value < best_value if mode == "min" else value > best_value

class PoseSolver(BaseSolver):
    def train(self,):
        self._setup()
        self.criterion = self.cfg.criterion
        self.optimizer = self.cfg.optimizer
        self.lr_scheduler = self.cfg.lr_scheduler
        self.lr_warmup_scheduler = self.cfg.lr_warmup_scheduler

        # Load datasets
        self.train_dataloader = dist_utils.warp_loader(
            self.cfg.train_dataloader, shuffle=self.cfg.train_dataloader.shuffle
        )
        self.val_dataloader = dist_utils.warp_loader(
            self.cfg.val_dataloader, shuffle=self.cfg.val_dataloader.shuffle
        )

        self.evaluator = self.cfg.evaluator

        # Enable self-defined flat-cosine scheduler for pose if requested
        self.self_lr_scheduler = False
        if hasattr(self.cfg, "lrsheduler") and self.cfg.lrsheduler is not None:
            iter_per_epoch = len(self.train_dataloader)
            print("     ## Using Self-defined Scheduler-{} (pose) ## ".format(self.cfg.lrsheduler))
            self.lr_scheduler = FlatCosineLRScheduler(
                self.optimizer,
                self.cfg.lr_gamma,
                iter_per_epoch,
                total_epochs=self.cfg.epoches,
                warmup_iter=self.cfg.warmup_iter,
                flat_epochs=self.cfg.flat_epoch,
                no_aug_epochs=self.cfg.no_aug_epoch,
            )
            self.self_lr_scheduler = True

        self.best_map_holder = BestMetricHolder(use_ema=self.cfg.use_ema)
        if self.cfg.resume:
            print(f'Resume checkpoint from {self.cfg.resume}')
            self.load_resume_state(self.cfg.resume)

    def fit(self,):
        self.train()
        args = self.cfg
        n_parameters, model_stats = stats(self.cfg)
        
        print(model_stats)
        # print("-" * 42 + "Model Structrue" + "-" * 43)
        # print(self.model)
        
        # print("-" * 42 + "Check Shape of feats" + "-" * 43)
        # model = self.model.module if hasattr(self.model, 'module') else self.model
        # device = next(model.parameters()).device  
        # with torch.no_grad():
        #     feats = model.backbone(torch.randn(1, 3, 640, 640).to(device))
        #     for i, f in enumerate(feats):
        #         print(i, f.shape)

        print("-" * 42 + "Start training" + "-" * 43)
        
        
        primary_metric = getattr(args, "primary_metric", "coco_eval_keypoints")
        primary_metric_mode = getattr(args, "primary_metric_mode", "max").lower()
        if primary_metric_mode not in {"min", "max"}:
            raise ValueError("primary_metric_mode must be 'min' or 'max'")
        initial_best = math.inf if primary_metric_mode == "min" else -math.inf
        stage_best = {1: initial_best, 2: initial_best}
        global_best = initial_best
        global_best_epoch = -1
        stop_epoch = self.train_dataloader.collate_fn.stop_epoch
        # evaluate again before resume training
        if self.last_epoch > 0:
            module = self.ema.module if self.ema else self.model
            test_stats = evaluate(
                module,
                self.postprocessor,
                self.evaluator,
                self.val_dataloader,
                self.device
            )
            resumed_value = _metric_value(test_stats, primary_metric)
            if math.isfinite(resumed_value):
                resumed_stage = 1 if self.last_epoch < stop_epoch else 2
                stage_best[resumed_stage] = resumed_value
                global_best = resumed_value
                global_best_epoch = self.last_epoch
            resume_stat = {'epoch': global_best_epoch, primary_metric: global_best}
            print(f'best_stat: {resume_stat}')

        start_time = time.time()
        start_epoch = self.last_epoch + 1
        for epoch in range(start_epoch, args.epoches):
            epoch_start_time = time.time()

            self.train_dataloader.set_epoch(epoch)
            # self.train_dataloader.dataset.set_epoch(epoch)
            if dist_utils.is_dist_avail_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)
            train_stats = train_one_epoch(
                self.self_lr_scheduler,
                self.lr_scheduler,
                self.model, 
                self.criterion, 
                self.train_dataloader, 
                self.optimizer, 
                self.cfg.train_dataloader.batch_size,
                args.grad_accum_steps,
                self.device, 
                epoch,
                args.clip_max_norm, 
                writer=self.writer, 
                warmup_scheduler=self.lr_warmup_scheduler, 
                ema=self.ema,
                args=args
                )

            if not self.self_lr_scheduler:
                if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                    self.lr_scheduler.step()

            if self.output_dir:
                checkpoint_paths = [self.output_dir / 'checkpoint.pth']
                # extra checkpoint before LR drop and every 100 epochs
                if (epoch + 1) % args.checkpoint_freq == 0:
                    checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    # weights = {
                    #     'model': self.state_dict(),
                    #     'ema': self.ema.state_dict() if self.ema is not None else None,
                    #     'optimizer': self.optimizer.state_dict(),
                    #     'lr_scheduler': self.lr_scheduler.state_dict(),
                    #     'warmup_scheduler': self.lr_warmup_scheduler.state_dict() if self.lr_warmup_scheduler is not None else None,
                    #     'epoch': epoch,
                    #     'args': args,
                    # }
                    dist_utils.save_on_master(self.state_dict(), checkpoint_path)


            
            # eval ema model if exists
            if self.ema is not None:
                test_stats = evaluate(
                    self.ema.module, 
                    self.postprocessor, 
                    self.evaluator,
                    self.val_dataloader, 
                    self.device, 
                    self.writer
                )
                for k in test_stats:
                    if self.writer and dist_utils.is_main_process():
                        for i, v in enumerate(test_stats[k]):
                            self.writer.add_scalar(f'Test/ema_{k}_{i}'.format(k), v, epoch)
                eval_stats = test_stats
            else:
                # eval regular model
                test_stats = evaluate(
                    self.model, 
                    self.postprocessor, 
                    self.evaluator,
                    self.val_dataloader, 
                    self.device, 
                    self.writer
                )
                # Log regular model results
                for k in test_stats:
                    if self.writer and dist_utils.is_main_process():
                        for i, v in enumerate(test_stats[k]):
                            self.writer.add_scalar(f'Test/regular_{k}_{i}'.format(k), v, epoch)
                eval_stats = test_stats
            
            current_value = _metric_value(eval_stats, primary_metric)
            stage = 1 if epoch < stop_epoch else 2
            improved = _is_better(current_value, stage_best[stage], primary_metric_mode)
            if improved:
                stage_best[stage] = current_value
                if self.output_dir:
                    dist_utils.save_on_master(
                        self.state_dict(), self.output_dir / f'best_stg{stage}.pth'
                    )
            elif stage == 2 and self.ema is not None:
                self.ema.decay -= 0.0001
                print(f'Refresh EMA at epoch {epoch} with decay {self.ema.decay}')

            if _is_better(current_value, global_best, primary_metric_mode):
                global_best = current_value
                global_best_epoch = epoch
            best_stat_print = {'epoch': global_best_epoch, primary_metric: global_best}
            print(f'best_stat: {best_stat_print}')


            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'test_{k}': v for k, v in eval_stats.items()},
                'epoch': epoch,
                'n_parameters': n_parameters
            }
            
            # Add EMA results to log if available
            if self.ema is not None:
                log_stats.update({f'test_{k}': v for k, v in test_stats.items()})

            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")
                      
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

    def val(self, ):
        self.eval()
        module = self.ema.module if self.ema else self.model
        test_stats = evaluate(
                module,
                self.postprocessor,
                self.evaluator,
                self.val_dataloader,
                self.device,
            )

        # if self.output_dir:
        #     dist_utils.save_on_master(coco_evaluator.coco_eval["keypoints"].eval, self.output_dir / "eval.pth")

        return
