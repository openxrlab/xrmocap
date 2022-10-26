# yapf: disable

import logging
import numpy as np
import os
import random
import time
import torch
import torch.distributed as dist
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
from mmcv.runner import get_dist_info, load_checkpoint
from torch.utils.data import DistributedSampler
from typing import Union
from xrprimer.utils.log_utils import get_logger

from xrmocap.core.evaluation.builder import build_evaluation
from xrmocap.data.dataset.builder import build_dataset
from xrmocap.model.architecture.builder import build_architecture
from xrmocap.utils.distribute_utils import (
    get_rank, is_main_process, time_synchronized,
)
from xrmocap.utils.mvp_utils import (
    AverageMeter, get_total_grad_norm, match_name_keywords, norm2absolute,
    save_checkpoint, set_cudnn,
)

# yapf: enable


class MVPTrainer():

    def __init__(self,
                 distributed: bool,
                 model_path: Union[None, str],
                 gpu_idx: int,
                 train_dataset: str,
                 test_dataset: str,
                 cudnn_setup: dict,
                 dataset_setup: dict,
                 evaluation_setup: dict,
                 mvp_setup: dict,
                 pretrained_backbone: Union[None, str] = None,
                 finetune_model: Union[None, str] = None,
                 resume: bool = False,
                 final_output_dir: Union[None, str] = './output',
                 lr_decay_epoch: list = [30],
                 test_model_file: Union[None, str] = None,
                 end_epoch: int = 30,
                 optimizer: Union[None, str] = None,
                 weight_decay: float = 0.0,
                 lr: float = 0.2,
                 logger: Union[None, str, logging.Logger] = None,
                 device: str = 'cuda',
                 seed: int = 42,
                 workers: int = 4,
                 train_batch_size: int = 1,
                 test_batch_size: int = 1,
                 lr_linear_proj_mult: float = 0.1,
                 model_root: str = './weight',
                 inference_conf_thr: list = [0.0],
                 print_freq: int = 100,
                 clip_max_norm: float = 0.1,
                 optimize_backbone: bool = False) -> None:
        """Create a trainer for training the Multi-view Pose Transformer(MVP).

        Args:
            distributed (bool):
                If distributed training is used.
            model_path (Union[None, str]):
                Path to the model weight,for evaluation.
            gpu_idx (int):
                Index of current GPU when using distributed training.
            train_dataset (str):
                Name of the train dataset.
            test_dataset (str):
                Name of the test dataset.
            lr (float, optional):
                Learning rate. Defaults to 0.2.
            weight_decay (float, optional):
                Weight decay. Defaults to 0.0.
            optimizer (Union[None, str], optional):
                Type of optimizer. Defaults to None.
            end_epoch (int, optional):
                End epoch of trainig. Defaults to 30.
            pretrained_backbone (Union[None, str], optional):
                Path to pretrained backbone weights
                if using pretrained model. Defaults to None.
            finetune_model (Union[None, str], optional):
                Path to a pretrained model weights to be finetuned.
                Defaults to None.
            resume (bool, optional):
                If auto resume from checkpoints is used.
                Defaults to False.
            final_output_dir (Union[None, str], optional):
                Path to output folder. Defaults to './output'.
            lr_decay_epoch (list, optional):
                Lr decay milestones. Defaults to [30].
            test_model_file (Union[None, str], optional):
                Path to test model weight. Defaults to None.
            cudnn_setup (dict):
                Dict of parameters to setup cudnn.
            dataset_setup (dict):
                Dict of parameters to setup the dataset.
            mvp_setup (dict):
                Dict of parameters to setup the MVP architecture.
            logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be selected.
                Defaults to None.
            device (str, optional):
                Training device, 'cuda' for GPU usage.
                Defaults to 'cuda'.
            seed (int, optional):
                Fix random seed between experiments if needed.
                Defaults to 42.
            workers (int, optional):
                Number of workers.. Defaults to 4.
            train_batch_size (int, optional):
                Batch size in training. Defaults to 1.
            test_batch_size (int, optional):
                Batch size in testing. Defaults to 1.
            lr_linear_proj_mult (float, optional):
                LR factor for linear projection related weights.
                Defaults to 0.1.
            model_root (str, optional):
                Root folder for pretrained weights. Defaults to './weight'.
            inference_conf_thr (list, optional):
                List of confidence threshold to filter non-human
                keypoints. Defaults to [0.0].
            print_freq (int, optional):
                Printing frequency during training. Defaults to 100.
            clip_max_norm (float, optional):
                Gradient clipping. Defaults to 0.1.
            optimize_backbone (bool, optional):
                Set it to be True to train the whole model jointly.
                Defaults to False.
        """

        self.logger = get_logger(logger)
        self.device = device
        self.seed = seed
        self.distributed = distributed
        self.model_path = model_path
        self.gpu_idx = gpu_idx

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.lr = lr
        self.lr_linear_proj_mult = lr_linear_proj_mult
        self.weight_decay = weight_decay
        self.optimizer = optimizer
        self.end_epoch = end_epoch
        self.pretrained_backbone = pretrained_backbone
        self.model_root = model_root
        self.finetune_model = finetune_model
        self.resume = resume
        self.lr_decay_epoch = lr_decay_epoch
        self.inference_conf_thr = inference_conf_thr
        self.test_model_file = test_model_file
        self.clip_max_norm = clip_max_norm
        self.print_freq = print_freq
        self.workers = workers
        self.final_output_dir = final_output_dir
        self.optimize_backbone = optimize_backbone

        self.cudnn_setup = cudnn_setup
        self.dataset_setup = dataset_setup
        self.mvp_setup = mvp_setup
        self.evaluation_setup = evaluation_setup

        seed = self.seed + get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def get_optimizer(self, model_without_ddp, weight_decay, optim_type):
        lr = self.lr
        if model_without_ddp.backbone is not None:
            for params in model_without_ddp.backbone.parameters():
                # Set it to be True to train the whole model jointly
                # Default to false to avoid OOM
                params.requires_grad = self.optimize_backbone

        lr_linear_proj_mult = self.lr_linear_proj_mult
        lr_linear_proj_names = ['reference_points', 'sampling_offsets']
        param_dicts = [{
            'params': [
                p for n, p in model_without_ddp.named_parameters()
                if not match_name_keywords(n, lr_linear_proj_names)
                and p.requires_grad
            ],
            'lr':
            lr,
        }, {
            'params': [
                p for n, p in model_without_ddp.named_parameters()
                if match_name_keywords(n, lr_linear_proj_names)
                and p.requires_grad
            ],
            'lr':
            lr * lr_linear_proj_mult,
        }]

        if optim_type == 'adam':
            optimizer = optim.Adam(param_dicts, lr=lr)
        elif optim_type == 'adamw':
            optimizer = optim.AdamW(param_dicts, lr=lr, weight_decay=1e-4)

        return optimizer

    def train(self):

        if is_main_process():
            self.logger.info('Loading data ..')

        train_dataset_cfg = dict(type='MVPDataset', logger=self.logger)
        train_dataset_cfg.update(self.dataset_setup.train_dataset_setup)
        train_dataset_cfg.update(self.dataset_setup.base_dataset_setup)
        train_dataset = build_dataset(train_dataset_cfg)

        test_dataset_cfg = dict(type='MVPDataset', logger=self.logger)
        test_dataset_cfg.update(self.dataset_setup.test_dataset_setup)
        test_dataset_cfg.update(self.dataset_setup.base_dataset_setup)
        test_dataset = build_dataset(test_dataset_cfg)

        n_views = train_dataset.n_views

        if self.distributed:
            rank, world_size = get_dist_info()
            sampler_train = DistributedSampler(train_dataset)
            sampler_val = DistributedSampler(
                test_dataset, world_size, rank, shuffle=False)
        else:
            sampler_train = torch.utils.data.RandomSampler(train_dataset)
            sampler_val = torch.utils.data.SequentialSampler(test_dataset)

        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, self.train_batch_size, drop_last=True)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_sampler=batch_sampler_train,
            num_workers=self.workers,
            pin_memory=True)

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.test_batch_size,
            sampler=sampler_val,
            pin_memory=True,
            num_workers=self.workers)

        set_cudnn(self.cudnn_setup.benchmark, self.cudnn_setup.deterministic,
                  self.cudnn_setup.enable)

        if is_main_process():
            self.logger.info('Constructing models ..')

        mvp_cfg = dict(
            type='MviewPoseTransformer', is_train=True, logger=self.logger)
        mvp_cfg.update(self.mvp_setup)
        model = build_architecture(mvp_cfg)

        eval_cfg = dict(
            type='MVPEvaluation',
            test_loader=test_loader,
            print_freq=self.print_freq,
            final_output_dir=self.final_output_dir,
            logger=self.logger)
        eval_cfg.update(self.evaluation_setup)
        evaluation = build_evaluation(eval_cfg)

        model.to(self.device)
        model.criterion.to(self.device)

        model_without_ddp = model
        if self.distributed:
            if is_main_process():
                self.logger.info('Distributed ..')
            model = torch.nn.parallel.\
                DistributedDataParallel(model, device_ids=[self.gpu_idx],
                                        find_unused_parameters=True)
            model_without_ddp = model.module

        optimizer = self.get_optimizer(model_without_ddp, self.weight_decay,
                                       self.optimizer)

        end_epoch = self.end_epoch

        if self.pretrained_backbone:
            # Load pretrained poseresnet weight for panoptic only
            checkpoint_file = os.path.join(self.model_root,
                                           self.pretrained_backbone)
            load_checkpoint(
                model_without_ddp.backbone,
                checkpoint_file,
                map_location='cpu',
                logger=self.logger)
        if self.finetune_model is not None:
            # Load the checkpoint with state_dict only
            checkpoint_file = os.path.join(self.model_root,
                                           self.finetune_model)
            checkpoint = load_checkpoint(
                model_without_ddp,
                checkpoint_file,
                map_location='cpu',
                logger=self.logger)
            start_epoch = 0
            best_precision = checkpoint['precision'] \
                if 'precision' in checkpoint else 0
        if self.resume:
            # Load the checkpoint with full dict keys
            checkpoint_file = os.path.join(self.final_output_dir,
                                           'checkpoint.pth.tar')
            checkpoint = load_checkpoint(
                model_without_ddp,
                checkpoint_file,
                map_location='cpu',
                logger=self.logger)
            start_epoch = checkpoint['epoch']
            best_precision = checkpoint['precision'] \
                if 'precision' in checkpoint else 0
            optimizer.load_state_dict(checkpoint['optimizer'])

        else:
            start_epoch, checkpoint, best_precision = 0, None, 0

        # list for step decay
        if isinstance(self.lr_decay_epoch, list):
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=self.lr_decay_epoch, gamma=0.1)
            if checkpoint is not None and 'lr_scheduler' in checkpoint:
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        # int for cosine decay
        else:
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, self.lr_decay_epoch, eta_min=1e-5)
            if checkpoint is not None and 'lr_scheduler' in checkpoint:
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

        n_parameters = sum(p.numel() for p in model.parameters()
                           if p.requires_grad)
        if is_main_process():
            self.logger.info(f'Number of params: {n_parameters}')

        for epoch in range(start_epoch, end_epoch):
            current_lr = optimizer.param_groups[0]['lr']
            if is_main_process():
                self.logger.info(f'Epoch: {epoch}, current lr: {current_lr}')
            train_3d(
                model,
                optimizer,
                train_loader,
                epoch,
                self.logger,
                self.final_output_dir,
                self.clip_max_norm,
                self.print_freq,
                n_views=n_views,
                device=self.device)

            lr_scheduler.step()

            for thr in self.inference_conf_thr:
                precision = evaluation.run(model, threshold=thr, is_train=True)

                if is_main_process():
                    if precision > best_precision:
                        best_precision = precision
                        best_model = True
                    else:
                        best_model = False
                    if isinstance(self.lr_decay_epoch, list):
                        self.logger.info(
                            f'saving checkpoint to {self.final_output_dir} '
                            f'(Best: {best_model})')
                        save_checkpoint(
                            {
                                'epoch': epoch + 1,
                                'state_dict': model.module.state_dict(),
                                'lr_scheduler': lr_scheduler.state_dict(),
                                'precision': best_precision,
                                'optimizer': optimizer.state_dict(),
                            }, best_model, self.final_output_dir)
                    else:
                        self.logger.info(
                            f'saving checkpoint to {self.final_output_dir} '
                            f'(Best: {best_model})')
                        save_checkpoint(
                            {
                                'epoch': epoch + 1,
                                'state_dict': model.module.state_dict(),
                                'precision': best_precision,
                                'optimizer': optimizer.state_dict(),
                            }, best_model, self.final_output_dir)
                dist.barrier()

        if is_main_process():
            final_model_state_file = os.path.join(self.final_output_dir,
                                                  'final_state.pth.tar')
            self.logger.info(
                f'saving final model state to {final_model_state_file}')
            torch.save(model.module.state_dict(), final_model_state_file)

    def eval(self):

        if is_main_process():
            self.logger.info('Loading data ..')

        test_dataset_cfg = dict(type='MVPDataset', logger=self.logger)
        test_dataset_cfg.update(self.dataset_setup.test_dataset_setup)
        test_dataset_cfg.update(self.dataset_setup.base_dataset_setup)
        test_dataset = build_dataset(test_dataset_cfg)

        if self.distributed:
            rank, world_size = get_dist_info()
            sampler_val = DistributedSampler(
                test_dataset, world_size, rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(test_dataset)

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.test_batch_size,
            sampler=sampler_val,
            pin_memory=True,
            num_workers=self.workers)

        set_cudnn(self.cudnn_setup.benchmark, self.cudnn_setup.deterministic,
                  self.cudnn_setup.enable)

        if is_main_process():
            self.logger.info('Constructing models ..')

        mvp_cfg = dict(
            type='MviewPoseTransformer', is_train=False, logger=self.logger)
        mvp_cfg.update(self.mvp_setup)
        model = build_architecture(mvp_cfg)

        eval_cfg = dict(
            type='MVPEvaluation',
            test_loader=test_loader,
            print_freq=self.print_freq,
            final_output_dir=self.final_output_dir,
            logger=self.logger)
        eval_cfg.update(self.evaluation_setup)
        evaluation = build_evaluation(eval_cfg)

        model.to(self.device)
        model.criterion.to(self.device)

        if self.distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[self.gpu_idx], find_unused_parameters=True)

        if self.model_path is not None:
            if is_main_process():
                self.logger.info(f'Load saved models state {self.model_path}')

            load_checkpoint(
                model.module,
                self.model_path,
                logger=self.logger,
                map_location='cpu')

        elif os.path.isfile(
                os.path.join(self.final_output_dir, self.test_model_file)):
            test_model_file = \
                os.path.join(self.final_output_dir,
                             self.test_model_file)
            if is_main_process():
                self.logger.info(
                    f'Load default best models state {test_model_file}')
            model.module.load_state_dict(torch.load(test_model_file))
        else:
            raise ValueError('Check the model file for testing!')

        for thr in self.inference_conf_thr:
            evaluation.run(model=model, threshold=thr, is_train=False)


def train_3d(model,
             optimizer,
             loader,
             epoch: int,
             logger: Union[None, str, logging.Logger],
             output_dir: str,
             clip_max_norm: float,
             print_freq: int,
             n_views: int = 5,
             device: str = 'cuda'):
    """Train one epoch.

    Args:
        model: Model to be trained.
        optimizer:
            Optimizer used in training.
        loader:
            Dataloader.
        epoch (int):
            Current epoch.
        logger (Union[None, str, logging.Logger]):
            Logger for logging. If None, root logger will be selected.
        output_dir (str):
            Path to output folder.
        clip_max_norm (float):
            Gradient clipping.
        print_freq (int):
            Printing frequency during training.
        n_views (int, optional):
            Number of views. Defaults to 5.
        device (str, optional):
            Device name. Defaults to 'cuda'.
    """
    logger = get_logger(logger)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_ce = AverageMeter()
    class_error = AverageMeter()
    loss_per_kp = AverageMeter()
    loss_pose_perprojection = AverageMeter()
    cardinality_error = AverageMeter()

    model.train()

    if model.module.backbone is not None:
        # Comment out this line if you want to train 2D backbone jointly
        model.module.backbone.eval()

    end = time.time()
    for i, (inputs, meta) in enumerate(loader):
        assert len(inputs) == n_views
        inputs = [i.to(device) for i in inputs]
        meta = [{
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in t.items()
        } for t in meta]
        data_time.update(time_synchronized() - end)
        end = time_synchronized()

        out, loss_dict, losses = model(views=inputs, meta=meta)

        n_kps = loader.dataset.n_kps
        bs, n_queries = out['pred_logits'].shape[:2]

        src_poses = out['pred_poses']['outputs_coord']. \
            view(bs, n_queries, n_kps, 3)
        src_poses = norm2absolute(src_poses, model.module.grid_size,
                                  model.module.grid_center)
        score = out['pred_logits'][:, :, 1:2].sigmoid()
        score = score.unsqueeze(2).expand(-1, -1, n_kps, -1)

        loss_ce.update(loss_dict['loss_ce'].sum().item())
        class_error.update(loss_dict['class_error'].sum().item())

        loss_per_kp.update(loss_dict['loss_per_kp'].sum().item())

        if 'loss_pose_perprojection' in loss_dict:
            loss_pose_perprojection.update(
                loss_dict['loss_pose_perprojection'].sum().item())

        cardinality_error.update(loss_dict['cardinality_error'].sum().item())

        if losses > 0:
            optimizer.zero_grad()
            losses.backward()
            if clip_max_norm > 0:
                grad_total_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), clip_max_norm)
            else:
                grad_total_norm = get_total_grad_norm(model.parameters(),
                                                      clip_max_norm)

            optimizer.step()

        batch_time.update(time_synchronized() - end)
        end = time_synchronized()

        if i % print_freq == 0 and is_main_process():
            gpu_memory_usage = torch.cuda.memory_allocated(0)
            speed = len(inputs) * inputs[0].size(0) / batch_time.val

            msg = \
                f'Epoch: [{epoch}][{i}/{len(loader)}]\t' \
                f'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                f'Speed: {speed:.1f} samples/s\t' \
                f'Data: {data_time.val:.3f}s ' f'({data_time.avg:.3f}s)\t' \
                f'loss_ce: {loss_ce.val:.7f} ' f'({loss_ce.avg:.7f})\t' \
                f'class_error: {class_error.val:.7f} ' \
                f'({class_error.avg:.7f})\t' \
                f'loss_per_kp: {loss_per_kp.val:.6f} ' \
                f'({loss_per_kp.avg:.6f})\t' \
                f'loss_pose_perprojection: ' \
                f'{loss_pose_perprojection.val:.6f} ' \
                f'({loss_pose_perprojection.avg:.6f})\t' \
                f'cardinality_error: {cardinality_error.val:.6f} ' \
                f'({cardinality_error.avg:.6f})\t' \
                f'Memory {gpu_memory_usage:.1f}\t' \
                f'gradnorm {grad_total_norm:.2f}'

            logger.info(msg)
