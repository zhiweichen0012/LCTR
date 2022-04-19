# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
from operator import mod, truediv
from pickle import TRUE
import sys
from typing import Iterable, List, Optional
import pdb
import os
import torch.nn.functional as F

import torch
import torch.nn as nn
from torch.nn.modules import conv
from torch.nn.modules.activation import Threshold
import numpy as np
from numpy.lib.function_base import delete

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

import utils
from vis_tools import *

import matplotlib.pyplot as plt


T_vis_path = os.path.join(os.getcwd(), 'vis_test')


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    model_ema: Optional[ModelEma] = None,
    mixup_fn: Optional[Mixup] = None,
    set_training_mode=True,
):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets, paths in metric_logger.log_every(
        data_loader, print_freq, header
    ):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = nn.CrossEntropyLoss()(outputs, targets)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = (
            hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        )
        loss_scaler(
            loss,
            optimizer,
            clip_grad=max_norm,
            parameters=model.parameters(),
            create_graph=is_second_order,
        )

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


@torch.no_grad()
def evaluate(data_loader, model, device, args=None, threshold_loc=-1):
    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    LocSet = []
    IoUSet = []
    IoUSetTop5 = []

    for images, target, paths, bboxes in metric_logger.log_every(
        data_loader, 10, header
    ):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        bboxes = bboxes.to(device, non_blocking=True)  # xyxy

        with torch.cuda.amp.autocast():
            patch_size = (16, 16)
            output, attns_avg, attns_conv, cls_patch_attns = model(
                images, attnBack=True
            )
            loss = criterion(output, target)

            w_featmap = images.shape[-2] // patch_size[0]
            h_featmap = images.shape[-1] // patch_size[1]
            nh = attns_avg.shape[1]
            batch = images.shape[0]
            _, pre_logits = output.topk(5, 1, True, True)

            for _b in range(batch):
                img = images[_b, :]
                im_name = os.path.basename(paths[_b]).split('.')[0]
                w, h = (
                    img.shape[1] - img.shape[1] % patch_size[0],
                    img.shape[2] - img.shape[2] % patch_size[1],
                )
                img = img[:, :w, :h].unsqueeze(0)
                attn_conv = attns_conv[_b, _b]
                cls_patch_attn_avg = torch.mean(cls_patch_attns, dim=1)[_b]
                attn_avg = cls_patch_attn_avg

                if threshold_loc != -1:
                    predict_box, cam_b = return_box_cam(
                        attn_avg=attn_avg, attn_conv=attn_conv, th=threshold_loc
                    )

                    # * compute loc acc
                    max_iou = -1
                    iou = utils.IoU(bboxes[_b], predict_box)
                    if iou > max_iou:
                        max_iou = iou
                    LocSet.append(max_iou)
                    temp_loc_iou = max_iou
                    if pre_logits[_b][0] != target[_b]:
                        max_iou = -1
                    IoUSet.append(max_iou)
                    max_iou = -1
                    for i in range(5):
                        if pre_logits[_b][i] == target[_b]:
                            max_iou = temp_loc_iou
                            break
                    IoUSetTop5.append(max_iou)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    if threshold_loc != -1:
        # * compute cls loc acc
        loc_acc_top1 = np.sum(np.array(IoUSet) >= 0.5) / len(IoUSet)
        loc_acc_top5 = np.sum(np.array(IoUSetTop5) >= 0.5) / len(IoUSetTop5)
        loc_acc_gt = np.sum(np.array(LocSet) >= 0.5) / len(LocSet)
        print(
            '*Loc Acc@1 {top1:.3f} Acc@5 {top5:.3f} GT {gt:.3f} TH {th:.3f} TestNum {tn:d}'.format(
                top1=loc_acc_top1 * 100,
                top5=loc_acc_top5 * 100,
                gt=loc_acc_gt * 100,
                th=threshold_loc,
                tn=len(LocSet),
            )
        )

    # * gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(
        '*Cls Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'.format(
            top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss
        )
    )
    if threshold_loc != -1:
        print(
            '{top1_loc:.3f} {top5_loc:.3f} {gt_loc:.3f} {th:.3f} {top1_cls.global_avg:.3f} {top5_cls.global_avg:.3f}'.format(
                top1_loc=loc_acc_top1 * 100,
                top5_loc=loc_acc_top5 * 100,
                gt_loc=loc_acc_gt * 100,
                th=threshold_loc,
                top1_cls=metric_logger.acc1,
                top5_cls=metric_logger.acc5,
            )
        )
    del images
    del target
    del paths
    del bboxes
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
