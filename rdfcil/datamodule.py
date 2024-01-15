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