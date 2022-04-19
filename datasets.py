# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import json
import torch
import pdb
import sys
import cv2
import numpy as np

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

from typing import Any, Callable, cast, Dict, List, Optional, Tuple

IMG_EXTENSIONS = (
    '.jpg',
    '.jpeg',
    '.png',
    '.ppm',
    '.bmp',
    '.pgm',
    '.tif',
    '.tiff',
    '.webp',
)


class myDataset(ImageFolder):
    def __init__(self, root: str, transform: Optional[Callable]):
        super().__init__(root, transform=transform)

        self.gt_dic = self._init_gt(root)

    def _init_gt(self, root):
        gt_dic = {}
        if "test" in root and "CUB" in root:
            box_file = os.path.join(root[:-5], 'list', 'test_boxes.txt')
            f2 = open(box_file)
            lines = f2.readlines()
            for l in lines:
                _id = l.strip().split(' ')[0]
                bbox = [
                    float(l.strip().split(' ')[2]),
                    float(l.strip().split(' ')[3]),
                    float(l.strip().split(' ')[4]),
                    float(l.strip().split(' ')[5]),
                ]
                gt_dic[_id] = bbox  # xywh
        elif "val" in root and "IMNET" in root:
            box_file = os.path.join(
                root[: root.find('val') - 1], 'list', 'val_boxes.txt'
            )
            f2 = open(box_file)
            lines = f2.readlines()
            for l in lines:
                _id = l.strip().split('	')[0]
                bbox = [
                    float(l.strip().split('	')[2]),
                    float(l.strip().split('	')[3]),
                    float(l.strip().split('	')[4]),
                    float(l.strip().split('	')[5]),
                ]
                bbox[2] = bbox[2] - bbox[0]
                bbox[3] = bbox[3] - bbox[1]
                gt_dic[_id] = bbox  # xywh
        return gt_dic

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        width, height = sample.size
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if "test" in path or "val" in path:
            if "CUB" in path:
                bbox = self.gt_dic[path[path.find("test") + 5 :]]
            else:
                bbox = self.gt_dic[path[path.find('val') + 4 :]]
            resize_min = 256.0
            if width > height:
                n_width = width * resize_min / height
                n_height = resize_min
            else:
                n_width = resize_min
                n_height = height * resize_min / width
            bbox[0] = bbox[0] / width * n_width
            bbox[1] = bbox[1] / height * n_height
            bbox[2] = bbox[2] / width * n_width
            bbox[3] = bbox[3] / height * n_height
            crop_wh = 224
            temp_crop_x = int(round((n_width - crop_wh + 1) / 2.0))
            temp_crop_y = int(round((n_height - crop_wh + 1) / 2.0))
            bbox[0] -= temp_crop_x
            bbox[1] -= temp_crop_y
            bbox[2] += bbox[0]
            bbox[3] += bbox[1]
            bbox = np.clip(np.array(bbox), 0, crop_wh)
            return sample, target, path, bbox
        return sample, target, path


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val_cls')
        dataset = myDataset(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == 'CUB':
        root = os.path.join(args.data_path, 'train' if is_train else 'test')
        dataset = myDataset(root, transform=transform)
        nb_classes = 200

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            transform.transforms[0] = transforms.RandomCrop(args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(
                size
            ),  # to maintain same ratio w.r.t. 224 images , interpolation=3
        )
        t.append(transforms.CenterCrop(args.input_size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
