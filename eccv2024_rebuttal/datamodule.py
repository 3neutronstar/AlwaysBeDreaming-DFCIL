# -*- coding: utf-8 -*-
import torch
import torchvision.transforms as T

import cl_lite.core as cl


class DataModule(cl.SplitedDataModule):
    finetuning: bool = False

    @property
    def train_transforms(self):
        return [
            T.RandomCrop(self.dims[-2:], padding=4),
            T.RandomHorizontalFlip(),
        ]

        
class PASS_DataModule(cl.SplitedDataModule):
    finetuning: bool = False

    @property
    def train_transforms(self):
        if self.dataset.startswith('imagenet'):
            transform=[
            T.RandomResizedCrop(self.dims[-2:]),
            T.RandomHorizontalFlip(),
            ]
            print("Setting for ImageNet")
        else:
            transform= [
            T.RandomCrop(self.dims[-2:], padding=4),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.24705882352941178),
            ]
        return transform
