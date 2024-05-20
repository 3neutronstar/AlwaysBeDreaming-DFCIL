# -*- coding: utf-8 -*-

import os
from collections import OrderedDict
from copy import deepcopy
from math import log, sqrt
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cl_lite.backbone as B
import cl_lite.core as cl
from cl_lite.deep_inversion import GenerativeInversion
from cl_lite.head import DynamicSimpleHead
from cl_lite.mixin import FeatureHookMixin
from cl_lite.nn import freeze

from datamodule import DataModule
from mixin import FinetuningMixin

import torch
import torch.nn as nn

import torch.nn.functional as F
from cl_lite.backbone.resnet_cifar import CifarResNet
from cl_lite.backbone.resnet import ResNet

class ISCF_ResNet(CifarResNet):
    def __init__(self, n=5, nf=16, channels=3, preact=False, zero_residual=True, pooling_config=..., downsampling="stride", final_layer=False, all_attentions=False, last_relu=False, **kwargs):
        super().__init__(n, nf, channels, preact, zero_residual, pooling_config, downsampling, final_layer, all_attentions, last_relu, **kwargs)        
    def forward_feat(self, x):
        x = self.conv_1_3x3(x)
        x = F.relu(self.bn_1(x), inplace=True)

        feats_s1, x1 = self.stage_1(x)
        feats_s2, x2 = self.stage_2(x1)
        feats_s3, x3 = self.stage_3(x2)
        x4 = self.stage_4(x3)

        return x4,[x1, x2, x3]


class ISCF_ResNet18(ResNet):
    def __init__(
        self,
        block,
        layers,
        zero_init_residual=True,
        nf=64,
        last_relu=False,
        initial_kernel=3,
        **kwargs
    ):
        super(ISCF_ResNet18, self).__init__(block, layers,zero_init_residual,nf,last_relu,initial_kernel, **kwargs)

    def forward_feat(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x_1 = self.layer1(x)
        x_2 = self.layer2(self.end_relu(x_1))
        x_3 = self.layer3(self.end_relu(x_2))
        x_4 = self.layer4(self.end_relu(x_3))

        return x_4, [x_1, x_2, x_3]

class SP(nn.Module):
    def __init__(self,reduction='mean'):
        super(SP,self).__init__()
        self.reduction=reduction

    def forward(self,fm_s,fm_t):
        fm_s = fm_s.view(fm_s.size(0),-1)
        G_s = torch.mm(fm_s,fm_s.t())
        norm_G_s =F.normalize(G_s,p=2,dim=1)

        fm_t = fm_t.view(fm_t.size(0),-1)
        G_t = torch.mm(fm_t,fm_t.t())
        norm_G_t = F.normalize(G_t,p=2,dim=1)
        loss = F.mse_loss(norm_G_s,norm_G_t,reduction=self.reduction)
        return loss

class ISCFModule(FeatureHookMixin, FinetuningMixin, cl.Module):
    datamodule: DataModule
    evaluator_cls = cl.ILEvaluator

    def __init__(
        self,
        base_lr: float = 0.1,
        momentum: float = 0.9,
        weight_decay: float = 0.0005,
        lr_factor: float = 0.1,
        milestones: List[int] = [80, 120],
        finetuning_epochs: int = 0,
        finetuning_lr: float = 0.005,
        lambda_ce: float = 0.5,
        lambda_lkd: float = 0.15,
        lambda_sp: float = 0.5,
        lambda_gce: float = 1.0,
        lambda_weq: float = 1.0,
        num_inv_iters: int = 5000,
        inv_lr: float = 0.001,
        inv_tau: float = 1000.0,
        inv_alpha_pr: float = 0.001,
        inv_alpha_rf: float = 50.0,
        inv_resume_from: str = None,
        fc_bias: bool = False,
    ):
        """Module of joint project

        Args:
            base_lr: Base learning rate
            momentum: Momentum value for SGD optimizer
            weight_decay: Weight decay value
            lr_factor: Learning rate decay factor
            milestones: Milestones for reducing learning rate
            finetuning_epochs: the number of finetuning epochs,
            finetuning_lr: the learning rate of finetuning,
            lambda_ce: the scale factor of cross entropy loss,
            lambda_lkd: the scale factor of stablility knowledge distillation,
            lambda_rkd: the scale factor of relation knowledge distillation,
            num_inv_iters: number of inversion iterations
            inv_lr: inversion learning rate
            inv_tau: temperature of inversion cross entropy loss
            inv_alpha_pr: factor of inversion image prior regularization
            inv_alpha_rf: factor of inversion feature statistics regularization
            inv_resume_from: resume inversion from a checkpoint
        """

        super().__init__()
        self.save_hyperparameters()

    def register_losses(self):
        self.register_loss(
            "ce",
            nn.functional.cross_entropy,
            ["prediction", "target"],#,"lcl_weight"],
        )

        # if self.model_old is None:
        #     return


        # self.set_loss_factor(
        #     "ce", self.hparams.lambda_ce
        # )

        # self.register_loss(
        #     "gce",
        #     nn.functional.cross_entropy,
        #     ["gce_prediction", "gce_target","gce_weight"],
        #     self.hparams.lambda_gce
        # )


    def update_old_model(self):
        self.backbone.zero_grad()
        self.head.zero_grad()
        model_old = [("backbone", self.backbone),("head", self.head)]
        self.model_old = deepcopy(nn.Sequential(OrderedDict(model_old))).eval()
        freeze(self.model_old)
        if self.hparams.num_inv_iters > 0:
            self.inversion = GenerativeInversion(
                model=deepcopy(self.model_old),
                dataset=self.datamodule.dataset,
                batch_size=self.datamodule.batch_size,
                max_iters=self.hparams.num_inv_iters,
                lr=self.hparams.inv_lr,
                tau=self.hparams.inv_tau,
                alpha_pr=self.hparams.inv_alpha_pr,
                alpha_rf=self.hparams.inv_alpha_rf,
            )
        else:
            mem = self.datamodule.update_memory(self.model_old)#, num_old_classes=n_cur-n_old) # MS
            self.datamodule.memory_set(mem)
        self.model_old=deepcopy(self.backbone).eval()
        self.model_old.head=deepcopy(self.head).eval()
        freeze(self.model_old)

        self.sp = SP(reduction='mean')

    def init_setup(self, stage=None):
        if self.datamodule.dataset.startswith("imagenet"):
            from cl_lite.backbone.resnet import BasicBlock
            self.backbone = ISCF_ResNet18(BasicBlock, [2, 2, 2, 2])
        else:
            self.backbone = ISCF_ResNet()
        kwargs = dict(num_features=self.backbone.num_features, bias=self.hparams.fc_bias)
        self.head = DynamicSimpleHead(**kwargs)
        self.model_old, self.inversion, self.rkd = None, None, nn.Identity()

        for task_id in range(0, self.datamodule.current_task + 1):
            if task_id > 0 and task_id == self.datamodule.current_task:
                self.update_old_model()  # load from checkpoint
            self.head.append(self.datamodule[task_id].num_classes)

    def setup(self, stage=None):
        current_task = self.datamodule.current_task
        resume_from_checkpoint = self.trainer.resume_from_checkpoint

        if current_task == 0 or resume_from_checkpoint is not None:
            self.init_setup(stage)
            self.print(f"=> Network Overview \n {self}")
        else:
            self.update_old_model()
            self.head.append(self.datamodule.num_classes)

        self.register_losses()


    def forward(self, input):
        output = self.backbone(input)
        output = self.head(output)
        return output

    def on_train_start(self):
        super().on_train_start()
        if self.model_old is not None and self.hparams.num_inv_iters > 0:
            ckpt_path = self.hparams.inv_resume_from
            if ckpt_path is None:
                self.inversion()
                log_dir = self.trainer.logger.log_dir
                ckpt_path = os.path.join(log_dir, "inversion.ckpt")
                print("\n==> Saving inversion states to", ckpt_path)
                torch.save(self.inversion.state_dict(), ckpt_path)
            else:
                print("\n==> Restoring inversion states from", ckpt_path)
                state = torch.load(ckpt_path, map_location=self.device)
                self.inversion.load_state_dict(state)
                self.hparams.inv_resume_from = None

    def training_step(self, batch, batch_idx):
        input, target = batch
        if self.finetuning and self.datamodule.current_task == 0:
            zeros = torch.zeros_like(input, requires_grad=True)
            return zeros.sum()

        target_t = self.datamodule.transform_target(target)
        target_all = target_t

        n_cur = self.head.num_classes
        if self.hparams.num_inv_iters > 0:
            if self.model_old is not None:
                _ = self.model_old.eval() if self.model_old.training else None
                _ = self.inversion.eval() if self.inversion.training else None
                n_old = self.model_old.head.num_classes

                input_rh, target_rh = self.inversion.sample(input.shape[0])
                target_all = torch.cat([target_t, target_rh])

                input_int=torch.cat([input, input_rh])
                # z,middles = self.backbone.forward_feat(input_int)
                # outputs=self.head(z)
            else:
                input_int=input
        else:
            input_int=input

        kwargs = dict(
            input=input_int,
            target=target_all,
            prediction=self(input_int),
        )

        loss, loss_dict = self.compute_loss(**kwargs)
        loss_dict={f"loss/{key}": val for key, val in loss_dict.items()}

        self.log_dict(loss_dict)
        # indices, counts = target_all.cpu().unique(return_counts=True)
        #self.cls_count[indices] += counts

        return loss

    def training_epoch_end(self, *args, **kwargs):
        return super().training_epoch_end(*args, **kwargs)

    def configure_optimizers(self):
        module = nn.ModuleList([self.backbone, self.head, self.rkd])
        optimizer = optim.SGD(
            module.parameters(),
            lr=self.hparams.base_lr,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=self.hparams.milestones,
            gamma=self.hparams.lr_factor,
        )

        scheduler = self.add_finetuning_lr_scheduler(optimizer, scheduler)

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
