# -*- coding: utf-8 -*-
import torch
import torchvision.transforms as T

import cl_lite.core as cl


class DataModule(cl.SplitedDataModule):
    finetuning: bool = False

    @property
    def transform(self):
        if self.dataset.startswith('imagenet'):
            transform=[
            T.RandomResizedCrop(self.dims[-2:]),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
            print("Setting for ImageNet")
        else:
            return super().transform()

        return T.Compose(transform)


    # @property
    # def train_transforms(self):
    #     if self.dataset.startswith('imagenet'):
    #         transform=[
    #         T.RandomResizedCrop(self.dims[-2:]),
    #         T.RandomHorizontalFlip(),
    #         T.ToTensor(),
    #         T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #         ]
    #         print("Setting for ImageNet")
    #         return transform
    #     else:
    #         return [
    #         T.RandomCrop(self.dims[-2:], padding=4),
    #         T.RandomHorizontalFlip(),
    #         ]

    @property
    def val_transforms(self):
        if self.dataset.startswith('imagenet'):
            transform=[
            T.CenterCrop(self.dims[-2:]),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
            print("Setting for ImageNet")
            return transform
        else:
            return super().val_transforms()
        
    # def setup(self, stage=None):
    #     try: # path
    #         print("HI")
    #         self._source, self.class_order = ds.split_task(
    #             ds.create(self.dataset, root=self.root),
    #             self.num_tasks,
    #             self.init_task_splits,
    #             self.class_order,
    #             self.task_seed,
    #         )
    #     except:
    #         print("HI22222222")
    #         if self.dataset=='tiny-imagenet200':
    #             dataset=TinyImageNet200(root=self.root)
    #         elif self.dataset=='imagenet100':
    #             dataset=ds.ImageNet100(root=self.root)

                
    #         self._source, self.class_order = ds.split_task(
    #             dataset,
    #             self.num_tasks,
    #             self.init_task_splits,
    #             self.class_order,
    #             self.task_seed,
    #         )



    #     indices = {c: i for i, c in enumerate(self.class_order)}
    #     indices = [indices[c] for c, _ in enumerate(self.class_order)]
    #     reindexed = indices != self.class_order
    #     self._class_indices = torch.tensor(indices) if reindexed else None

    #     self.switch_task(self.current_task, force=True)

    #     self.setup_memory()


class PASS_DataModule(cl.SplitedDataModule):
    finetuning: bool = False

    @property
    def transforms(self):
        if self.dataset.startswith('imagenet'):
            transform=[
            T.RandomResizedCrop(self.dims[-2:]),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
            print("Setting for ImageNet")
        else:
            transform= [
            T.RandomCrop(self.dims[-2:], padding=4),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.24705882352941178),
            # T.ToTensor(),
            # T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            ]
        return T.Compose(transform)
        
    @property
    def val_transforms(self):
        if self.dataset.startswith('imagenet'):
            transform=[
            T.CenterCrop(self.dims[-2:]),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
            print("Setting for ImageNet")
            return transform
        else:
            return [
            T.CenterCrop(self.dims[-2:]),
            # T.ToTensor(),
            ]