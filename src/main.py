import os
import sys
import datetime
import time
import logging
import math
import json
from pathlib import Path
import subprocess

import omegaconf
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import models as torchvision_models

from src.datasets.augs import DataAugmentationDINO
from src.datasets.bdd100k import BDD100KDataset
from src.datasets.wt import WalkingToursDataset, WalkingToursDecodedDataset
import src.utils as utils
from src.midway import MidwayLoss
import src.vision_transformer as vits
from src.vision_transformer import DINOHead

log = logging.getLogger(__name__)


def main(args):
    world_size, rank = utils.init_distributed()

    utils.fix_random_seeds(args.seed)
    log.info("git:\n  {}\n".format(utils.get_sha()))
    resolved_args = omegaconf.OmegaConf.to_container(args, resolve=True, throw_on_missing=True)
    log.info("{}".format(resolved_args).replace(', ', ',\n'))

    cudnn.benchmark = True

    if utils.is_main_process() and args.wandb:
        name = (args.name + "//" + args.sub_name) if hasattr(args, "sub_name") else (".LOCAL" + "//" + args.name)
        wandb_run = utils.init_wandb(resolved_args, name, args.experiment_dir, None, 'midway-network')

    # ============ preparing data ... ============
    transform = DataAugmentationDINO(
        args.global_crops_scale,
        args.local_crops_scale,
        args.local_crops_number,
        args.initial_crop_scale,
        args.global_crop_resolution,
        motion_crop_scale=args.motion_crop_scale,
        motion_crop_resolution=args.motion_crop_resolution,
        motion_crop_ratio=args.motion_crop_ratio,
        motion_crop_jitter=args.motion_crop_jitter,
        motion_crop_aug=args.motion_crop_aug,
        motion_affine_aug_params=args.motion_affine_aug_params,
    )
    epoch_increment = 1
    dataset = None
    if args.dataset == 'bdd100k':
        dataset = BDD100KDataset(args.data_path,
                                 transform=transform,
                                 delta_t=args.delta_t,
                                 meta_info_file=args.meta_info_file,
                                 repeat_sample=args.repeat_sample,
                                 backend=args.backend,
                                 )
        epoch_increment = args.repeat_sample
    elif args.dataset == 'walking_tours':
        dataset = WalkingToursDataset(args.data_path,
                                      transform=transform,
                                      delta_t=args.delta_t,
                                      repeat_sample=args.repeat_sample,
                                      backend=args.backend,
                                      dataset_fraction=args.dataset_fraction
        )
    elif args.dataset == 'walking_tours_decoded':
        dataset = WalkingToursDecodedDataset(args.data_path,
                                             transform=transform,
                                             delta_t=args.delta_t,
                                             dataset_fraction=args.dataset_fraction
        )

    use_motion_crops = args.motion_crop_scale is not None
    if args.delta_t is not None:
        global_crops_number = 4
    else:
        global_crops_number = 2

    sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=3 if args.num_workers > 0 else None
    )
    dataset_len = len(dataset)
    log.info(f"Data loaded: there are {dataset_len} images.")

    # ============ building student and teacher networks ... ============
    # we changed the name DeiT-S for ViT-S to avoid confusions
    args.arch = args.arch.replace("deit", "vit")
    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if args.arch in vits.__dict__.keys():
        student = vits.__dict__[args.arch](
            img_size=args.global_crop_resolution,
            patch_size=args.patch_size,
            drop_path_rate=args.drop_path_rate,  # stochastic depth
            num_register_tokens=args.num_register_tokens,
        )
        teacher = vits.__dict__[args.arch](
            img_size=args.global_crop_resolution,
            patch_size=args.patch_size,
            num_register_tokens=args.num_register_tokens,
        )
        embed_dim = student.embed_dim
    # otherwise, we check if the architecture is in torchvision models
    elif args.arch in torchvision_models.__dict__.keys():
        student = torchvision_models.__dict__[args.arch]()
        teacher = torchvision_models.__dict__[args.arch]()
        embed_dim = student.fc.weight.shape[1]
    # otherwise if the network is a XCiT
    elif args.arch in torch.hub.list("facebookresearch/xcit:main"):
        student = torch.hub.load('facebookresearch/xcit:main', args.arch,
                                 pretrained=False, drop_path_rate=args.drop_path_rate)
        teacher = torch.hub.load('facebookresearch/xcit:main', args.arch, pretrained=False)
        embed_dim = student.embed_dim
    else:
        log.info(f"Unknown architecture: {args.arch}")
    patch_grid_size = (student.patch_embed.num_patches_h, student.patch_embed.num_patches_w)

    # multi-crop wrapper handles forward with inputs of different resolutions
    loss_level = []
    feature_levels = sorted(list(set(args.motion_feature_levels + loss_level)))

    total_enc_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    log.info(f'Number of trainable encoder parameters: {total_enc_params}')

    student = utils.MultiCropWrapper(student,
        DINOHead(
            embed_dim,
            args.out_dim,
            use_bn=args.use_bn_in_head,
            norm_last_layer=args.norm_last_layer,
        ) if args.dino_weight > 0 else nn.Identity(),
        feature_levels,
    )
    teacher = utils.MultiCropWrapper(teacher,
        DINOHead(
            embed_dim,
            args.out_dim,
            use_bn=args.use_bn_in_head,
        ) if args.dino_weight > 0 else nn.Identity(), 
        feature_levels,
    )
    total_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    log.info(f'Number of trainable encoder+head parameters: {total_params}')

    # move networks to gpu
    gpu = rank % torch.cuda.device_count()
    device = torch.device(gpu)
    student, teacher = student.to(device), teacher.to(device)
    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[gpu])
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    student = nn.parallel.DistributedDataParallel(student, device_ids=[gpu])

    # ============ preparing losses ... ============
    teacher_temp_schedule = utils.linear_schedule(
        args.teacher_temp,
        args.teacher_temp,
        args.epochs,
        len(data_loader),
        args.warmup_teacher_temp_epochs,
        args.warmup_teacher_temp
    )
    total_dino_crops = global_crops_number + args.local_crops_number * (1 + int(args.delta_t is not None))

    if args.dino_weight > 0:
        dino_loss = DINOLoss(
            args.out_dim,
            total_dino_crops + 2 * int(args.use_dino_motion_crops), 
            global_crops_number + 2 * int(args.use_dino_motion_crops),
            teacher_temp_schedule,
            use_dino_motion_crops=args.use_dino_motion_crops,
        ).to(device)
    else:
        dino_loss = None

    model_configs = dict(args.midway_model_configs)
    model_configs.update({
        'feature_level_idx': [feature_levels.index(i) for i in args.motion_feature_levels],
        'embed_dim': embed_dim,
        'out_dim': model_configs['motion_out_dim'],
        'use_bn': args.use_bn_in_head,
        'teacher_temp_schedule': teacher_temp_schedule,
        'patch_grid_size': patch_grid_size,
        'use_motion_crops': use_motion_crops,
        'patch_size': args.patch_size,
    })
    motion_loss = MidwayLoss(
        **model_configs
    ).to(device)
    if utils.has_batchnorms(motion_loss):
        motion_loss = nn.SyncBatchNorm.convert_sync_batchnorm(motion_loss)
    total_motion_params = sum(p.numel() for p in motion_loss.parameters() if p.requires_grad)
    log.info(f'Number of trainable latent motion parameters: {total_motion_params}')
    motion_loss = nn.parallel.DistributedDataParallel(motion_loss, device_ids=[gpu])
    motion_loss_without_ddp = motion_loss.module

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        student.load_state_dict(checkpoint['student'], strict=False)
        if motion_loss and 'motion_loss' in checkpoint:
            motion_loss.load_state_dict(checkpoint['motion_loss'], strict=False)
        log.info(f"Loaded pre-trained model from {args.checkpoint}")

    # teacher and student start with the same weights
    msg = teacher_without_ddp.load_state_dict(student.module.state_dict(), strict=False)
    log.info(msg)
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    log.info(f"Student and Teacher are built: they are both {args.arch} network.")

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student)
    if motion_loss:
        params_groups += utils.get_params_groups(motion_loss)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
    else:
        raise NotImplementedError(f"Optimizer {args.optimizer} not implemented.")
    
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    repeat_sample = args.repeat_sample if args.repeat_sample is not None else 1

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size() * repeat_sample) / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                               args.epochs, dataset_len)
    log.info(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.experiment_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        dino_loss=dino_loss,
        motion_loss=motion_loss
    )
    start_epoch = to_restore["epoch"]

    start_time = time.time()
    log.info("Starting DINO training !")
    for epoch in range(start_epoch, args.epochs, epoch_increment):
        if args.dataset == 'walking_tours':
            dataset = WalkingToursDataset(args.data_path,
                                          transform=transform,
                                          delta_t=args.delta_t,
                                          repeat_sample=args.repeat_sample,
                                          backend=args.backend,
                                          dataset_fraction=args.dataset_fraction)
            sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
            data_loader = torch.utils.data.DataLoader(
                dataset,
                sampler=sampler,
                batch_size=args.batch_size_per_gpu,
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=True,
                prefetch_factor=3 if args.num_workers > 0 else None
            )
        elif args.dataset == 'walking_tours_decoded':
            dataset = WalkingToursDecodedDataset(args.data_path,
                                                 transform=transform,
                                                 delta_t=args.delta_t,
                                                 dataset_fraction=args.dataset_fraction)
            sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
            data_loader = torch.utils.data.DataLoader(
                dataset,
                sampler=sampler,
                batch_size=args.batch_size_per_gpu,
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=True,
                prefetch_factor=3 if args.num_workers > 0 else None
            )
        # ============ training one epoch of DINO ... ============
        train_stats = train_one_epoch(student, teacher, teacher_without_ddp, dino_loss,
            motion_loss, data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
            epoch, epoch_increment, fp16_scaler, args, device)

        # ============ saving models ... ============
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + epoch_increment,
            'args': args
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        if dino_loss is not None:
            save_dict['dino_loss'] = dino_loss.state_dict()
        if motion_loss is not None:
            save_dict['motion_loss'] = motion_loss.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.experiment_dir, 'checkpoint.pth'))
        if args.saveckp_freq and ((epoch + epoch_increment) % args.saveckp_freq == 0 or (epoch + epoch_increment) >= args.epochs):
            utils.save_on_master(save_dict, os.path.join(args.experiment_dir, f'checkpoint{epoch + epoch_increment:04}.pth'))
        
        # ============ evaluation ... ============
        if args.eval and args.eval_freq and utils.is_main_process() and ((epoch + epoch_increment) % args.eval_freq == 0 or (epoch + epoch_increment) >= args.epochs):
            ckpt_path = os.path.join(args.experiment_dir, f'checkpoint{epoch + epoch_increment:04}.pth')
            if not os.path.exists(ckpt_path):
                utils.save_on_master(save_dict, ckpt_path)
            submit_eval(args, epoch + epoch_increment, ckpt_path)

        # ============ writing logs ... ============
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.experiment_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    log.info('Training time {}'.format(total_time_str))


def train_one_epoch(student, teacher, teacher_without_ddp, dino_loss, motion_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule, epoch,
                    epoch_increment, fp16_scaler, args, device):
    student.train()
    teacher.train()
    motion_loss.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    dummy_iterations = 10
    for it_, images_ in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + (it_ * epoch_increment)  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]
            
        # adjust the number of dummy iterations to avoid bottlenecking on the GPU
        ratio = metric_logger.data_time.avg / metric_logger.iter_time.avg
        if ratio > 0.6:
            dummy_iterations = min(1000, dummy_iterations + 10)
        elif ratio < 0.3 and dummy_iterations > 0:
            dummy_iterations = max(0, dummy_iterations - 10)
        dummy = utils.gpu_intensive_dummy_operation(device, iterations=dummy_iterations)

        images = []
        for im in images_:
            im = im.to(device, non_blocking=True)
            if len(im.shape) == 5:
                im = im.flatten(0, 1)
            images.append(im)
        epoch_int = int(it_ * epoch_increment / len(data_loader)) + epoch
        # teacher and student forward passes + compute dino loss
        if args.delta_t is not None:
            global_crops_number = 4
        else:
            global_crops_number = 2

        no_motion_crops = args.motion_crop_scale is None
        if no_motion_crops or args.use_dino_motion_crops:
            dino_start_idx = 0
        else:
            dino_start_idx = 2

        motion_idx = 2
        if args.use_dino_motion_crops:
            global_crops_number += 2
        
        loss_masks = None
        if not no_motion_crops or args.use_dino_motion_crops:
            if dict(args.motion_affine_aug_params):
                affine_masks = images[:2]
                images = images[2:]
                loss_masks = torch.cat(affine_masks, dim=0)
        
        num_motion_losses = len(args.midway_weights)

        with torch.cuda.amp.autocast(fp16_scaler is not None):
            if args.dino_weight > 0 or no_motion_crops:
                teacher_dino_output, teacher_motion_output = teacher(images[dino_start_idx:dino_start_idx+global_crops_number], compute_features=no_motion_crops)  # only the 4 global views pass through the teacher
                student_dino_output, student_motion_output = student(images[dino_start_idx:], compute_features=no_motion_crops)
                img_targets = images[dino_start_idx:dino_start_idx+global_crops_number]
            _dino_loss = torch.tensor(0., device=device)
            _motion_loss = torch.tensor(0., device=device)
            if args.dino_weight > 0:
                _dino_loss = args.dino_weight * dino_loss(student_dino_output, teacher_dino_output, it)
            midway_weights = [0 for _ in range(num_motion_losses)]
            if epoch_int >= args.midway_start_epoch:
                midway_weights = list(args.midway_weights)
            if not no_motion_crops:
                teacher_motion_output = teacher(images[:motion_idx], use_head=False, compute_features=True)[1]
                student_motion_output = student(images[:motion_idx], use_head=False, compute_features=True)[1]
            _motion_loss, motion_losses, metric_dict = motion_loss(student_motion_output, teacher_motion_output, midway_weights, it, loss_masks)
        loss = _dino_loss + _motion_loss

        loss_item = loss.item()
        if not math.isfinite(loss_item):
            log.warning("Loss is {}, stopping training".format(loss_item))
            sys.exit(1)

        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
            if motion_loss:
                for param_q, param_k in zip(motion_loss.module.student_parameters, motion_loss.module.teacher_parameters):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        dino_loss_item = _dino_loss.item()
        motion_loss_item = _motion_loss.item()
        motion_losses_item = {f'motion_loss_{i}': motion_losses[i].item() for i in range(num_motion_losses)}
        metric_dict_item = {k: v.item() for k, v in metric_dict.items()}

        torch.cuda.synchronize()
        metric_logger.update(loss=loss_item)
        metric_logger.update(dino_loss=dino_loss_item)
        metric_logger.update(motion_loss=motion_loss_item)
        metric_logger.update(**motion_losses_item)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
        metric_logger.update(**metric_dict_item)


        if it_ % 10 == 0 and utils.is_main_process() and args.wandb:
            motion_losses_item = {f'train/{k}': v for k, v in motion_losses_item.items()}
            utils.wandb_log({"train/loss": loss_item,
                             "train/dino_loss": dino_loss_item,
                             "train/motion_loss": motion_loss_item,
                             **motion_losses_item,
                             "train/lr": optimizer.param_groups[0]["lr"],
                             "train/wd": optimizer.param_groups[0]["weight_decay"],
                             "train/step": it},
                             step=it)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    log.info(f"Averaged stats: {metric_logger}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, nteachercrops,
                 teacher_temp_schedule, student_temp=0.1,
                 center_momentum=0.9, use_dino_motion_crops=False):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.nteachercrops = nteachercrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = teacher_temp_schedule

        self.use_dino_motion_crops = use_dino_motion_crops

    def forward(self, student_output, teacher_output, epoch_iter):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch_iter]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(self.nteachercrops)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if self.use_dino_motion_crops:
                    if v < 2 and iq < 2:
                        if v == iq:
                            # we skip cases where student and teacher operate on the same view
                            continue
                    elif v >= 2 and iq >= 2:
                        if v == iq or (v < self.nteachercrops and abs(v - iq) == 2):
                            # we skip cases where student and teacher operate on the same view (including from different frames)
                            continue
                    else:
                        continue
                elif v == iq or (v < self.nteachercrops and abs(v - iq) == 2):
                    # we skip cases where student and teacher operate on the same view (including from different frames)
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


def submit_eval(args, epoch, ckpt_path):
    eval_dir = os.path.join(args.experiment_dir, f'eval-epoch{epoch}')
    os.makedirs(eval_dir, exist_ok=True)
    ckpt_out = os.path.join(eval_dir, f'checkpoint_epoch{epoch}_eval.pth')
    eval_script = args.eval_script
    config = args.eval_config

    output_log = os.path.join(eval_dir, 'slurm-%j.out')
    error_log = os.path.join(eval_dir, 'slurm-%j.err')
    sbatch_command = ['sbatch', '--export=NONE', f'--output={output_log}', f'--error={error_log}',
                    f'{eval_script}', ckpt_path, ckpt_out, config, eval_dir]
    subprocess.Popen(sbatch_command)
    log.info(f"Submitted evaluation job: {' '.join(sbatch_command)}")
