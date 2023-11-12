from __future__ import print_function
import torch
import torch.nn as nn
from torch.nn import functional as F
import models
from utils.distillation_rkd import RKDAngleLoss
from utils.metric import AverageMeter, Timer
import numpy as np
from .datafree_helper import Teacher
from .default import NormalNN, weight_reset, accumulate_acc, loss_fn_kd
import copy
from torch.optim import Adam

class DeepInversionGenBN(NormalNN):

    def __init__(self, learner_config):
        super(DeepInversionGenBN, self).__init__(learner_config)
        self.inversion_replay = False
        self.previous_teacher = None
        self.dw = self.config['DW']
        self.device = 'cuda' if self.gpu else 'cpu'
        self.power_iters = self.config['power_iters']
        self.deep_inv_params = self.config['deep_inv_params']
        self.kd_criterion = nn.MSELoss(reduction="none")

        # gen parameters
        self.generator = self.create_generator()
        self.generator_optimizer = Adam(params=self.generator.parameters(), lr=self.deep_inv_params[0])
        self.beta = self.config['beta']

        # repeat call for generator network
        if self.gpu:
            self.cuda_gen()
        
    ##########################################
    #           MODEL TRAINING               #
    ##########################################

    def learn_batch(self, train_loader, train_dataset, model_save_dir, val_loader=None):
        
        self.pre_steps()

        # try to load model
        need_train = True
        if not self.overwrite:
            try:
                self.load_model(model_save_dir)
                need_train = False
            except:
                pass

        # trains
        if need_train:
            if self.reset_optimizer:  # Reset optimizer before learning each task
                self.log('Optimizer is reset!')
                self.init_optimizer()

            # data weighting
            self.data_weighting(train_dataset)

            # Evaluate the performance of current task
            self.log('Epoch:{epoch:.0f}/{total:.0f}'.format(epoch=0,total=self.config['schedule'][-1]))
            if val_loader is not None:
                self.validation(val_loader)

            # losses = [AverageMeter() for i in range(3)]
            losses = [AverageMeter() for i in range(5)]
            acc = AverageMeter()
            accg = AverageMeter()
            batch_time = AverageMeter()
            batch_timer = Timer()
            self.save_gen = False
            self.save_gen_later = False
            for epoch in range(self.config['schedule'][-1]):
                self.epoch=epoch
                if epoch > 0: self.scheduler.step()
                for param_group in self.optimizer.param_groups:
                    self.log('LR:', param_group['lr'])
                batch_timer.tic()
                for i, (x, y, task)  in enumerate(train_loader):

                    # verify in train mode
                    self.model.train()

                    # send data to gpu
                    if self.gpu:
                        x =x.cuda()
                        y = y.cuda()

                    # data replay
                    if self.inversion_replay:
                        x_replay, y_replay, y_replay_hat = self.sample(self.previous_teacher, len(x), self.device)

                    # if KD
                    if self.inversion_replay:
                        y_hat = self.previous_teacher.generate_scores(x, allowed_predictions=np.arange(self.last_valid_out_dim))
                        _, y_hat_com = self.combine_data(((x, y_hat),(x_replay, y_replay_hat)))
                    else:
                        y_hat_com = None

                    # combine inputs and generated samples for classification
                    if self.inversion_replay:
                        x_com, y_com = self.combine_data(((x, y),(x_replay, y_replay)))
                    else:
                        x_com, y_com = x, y

                    # sd data weighting (NOT online learning compatible)
                    if self.dw:
                        dw_cls = self.dw_k[y_com.long()]
                    else:
                        dw_cls = None

                    # model update
                    loss, loss_class, loss_other, output= self.update_model(x_com, y_com, y_hat_com, dw_force = dw_cls, kd_index = np.arange(len(x), len(x_com)))
                    # loss, loss_class, loss_kd, output= self.update_model(x_com, y_com, y_hat_com, dw_force = dw_cls, kd_index = np.arange(len(x), len(x_com)))

                    
                    # measure elapsed time
                    batch_time.update(batch_timer.toc()) 

                    # measure accuracy and record loss
                    y_com = y_com.detach()
                    accumulate_acc(output[:self.batch_size], y_com[:self.batch_size], task, acc, topk=(self.top_k,))
                    if self.inversion_replay: accumulate_acc(output[self.batch_size:], y_com[self.batch_size:], task, accg, topk=(self.top_k,))
                    losses[0].update(loss,  y_com.size(0)) 
                    losses[1].update(loss_class,  y_com.size(0))
                    if type(loss_other)==tuple:
                        loss_kd, loss_middle, loss_balancing = loss_other
                        losses[2].update(loss_kd,  y_com.size(0))
                        losses[3].update(loss_middle,  y_com.size(0))
                        losses[4].update(loss_balancing,  y_com.size(0))
                    else:
                        loss_kd = loss_other
                        losses[2].update(loss_kd,  y_com.size(0))
                    batch_timer.tic()

                # eval update
                self.log('Epoch:{epoch:.0f}/{total:.0f}'.format(epoch=self.epoch+1,total=self.config['schedule'][-1]))
                # self.log(' * Loss {loss.avg:.3f} | CE Loss {lossb.avg:.3f} | KD Loss {lossc.avg:.3f}'.format(loss=losses[0],lossb=losses[1],lossc=losses[2]))
                self.log(' * Loss {loss.avg:.4e} | CE Loss {lossb.avg:.4e} | LKD Loss {lossc.avg:.4e} | SP Loss {lossd.avg:.4e} | WEQ Reg {losse.avg:.4e}'.format(loss=losses[0],lossb=losses[1],lossc=losses[2],lossd=losses[3],losse=losses[4]))
                self.log(' * Train Acc {acc.avg:.3f} | Train Acc Gen {accg.avg:.3f}'.format(acc=acc,accg=accg))

                # Evaluate the performance of current task
                if val_loader is not None:
                    self.validation(val_loader)

                # reset
                losses = [AverageMeter() for i in range(5)]
                acc = AverageMeter()
                accg = AverageMeter()
                self.epoch_end()

            if self.config['finetuning_epochs'] > 0:
                self.log('===============fine-tuning================')
                for ft_epoch in range(self.config['finetuning_epochs']):
                    losses = AverageMeter()
                    acc = AverageMeter()
                    batch_time = AverageMeter()
                    batch_timer = Timer()
                    self.save_gen = False
                    self.save_gen_later = False
                    ft_optim=Adam(params=self.model.parameters(), lr=self.config['finetuning_lr'])
                    for i, (x, y, task)  in enumerate(train_loader):
                        self.model.train()
                        if self.gpu:
                            x =x.cuda()
                            y = y.cuda()

                        if self.inversion_replay:
                            x_replay, y_replay, y_replay_hat = self.sample(self.previous_teacher, len(x), self.device)
                        if self.inversion_replay:
                            x_com, y_com = self.combine_data(((x, y),(x_replay, y_replay)))
                        else:
                            x_com, y_com = x, y

                        loss = self.finetuning_update_model(x_com,y_com)

                        ft_optim.zero_grad()
                        loss.backward()
                        # step
                        ft_optim.step()
                        losses.update(loss,  y_com.size(0)) 
                    self.log(' [{} epoch] Train Loss {loss.avg:.3f}'.format(ft_epoch, loss=losses))
                    if val_loader is not None:
                        self.validation(val_loader)
                    self.epoch_end()
        self.model.eval()
        self.last_last_valid_out_dim = self.last_valid_out_dim
        self.last_valid_out_dim = self.valid_out_dim
        self.first_task = False

        # for eval
        if self.previous_teacher is not None:
            self.previous_previous_teacher = self.previous_teacher
        
        # new teacher
        if (self.out_dim == self.valid_out_dim): need_train = False
        self.previous_teacher = Teacher(solver=copy.deepcopy(self.model), generator=self.generator, gen_opt = self.generator_optimizer, img_shape = (-1, train_dataset.nch,train_dataset.im_size, train_dataset.im_size), iters = self.power_iters, deep_inv_params = self.deep_inv_params, class_idx = np.arange(self.valid_out_dim), train = need_train, config = self.config)
        self.sample(self.previous_teacher, self.batch_size, self.device, return_scores=False)
        if len(self.config['gpuid']) > 1:
            self.previous_linear = copy.deepcopy(self.model.module.last)
        else:
            self.previous_linear = copy.deepcopy(self.model.last)
        self.inversion_replay = True

        try:
            return batch_time.avg
        except:
            return None

    def finetuning_update_model(self, x_com, y_com):
        logits = self.model.forward(x_com)
        class_idx = np.arange(self.batch_size)
        loss = self.criterion(logits[class_idx], y_com[class_idx].long(), torch.ones(y_com[class_idx].size()).cuda())
        return loss

    def update_model(self, inputs, targets, target_scores = None, dw_force = None, kd_index = None):

        loss_kd = torch.zeros((1,), requires_grad=True).cuda()

        if dw_force is not None:
            dw_cls = dw_force
        elif self.dw:
            dw_cls = self.dw_k[targets.long()]
        else:
            dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]

        # forward pass
        logits = self.forward(inputs)

        # classification 
        class_idx = np.arange(self.batch_size)
        loss_class = self.criterion(logits[class_idx], targets[class_idx].long(), dw_cls[class_idx])

        # KD old
        if target_scores is not None:
            loss_kd = self.mu * loss_fn_kd(logits[class_idx], target_scores[class_idx], dw_cls[class_idx], np.arange(self.last_valid_out_dim).tolist(), self.DTemp)

        # KD new
        if target_scores is not None:
            target_scores = F.softmax(target_scores[:, :self.last_valid_out_dim] / self.DTemp, dim=1)
            target_scores = [target_scores]
            target_scores.append(torch.zeros((len(targets),self.valid_out_dim-self.last_valid_out_dim), requires_grad=True).cuda())
            target_scores = torch.cat(target_scores, dim=1)
            loss_kd += self.mu * loss_fn_kd(logits[kd_index], target_scores[kd_index], dw_cls[kd_index], np.arange(self.valid_out_dim).tolist(), self.DTemp, soft_t = True)

        total_loss = loss_class + loss_kd
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return total_loss.detach(), loss_class.detach(), loss_kd.detach(), logits

    ##########################################
    #             MODEL UTILS                #
    ##########################################

    def combine_data(self, data):
        x, y = [],[]
        for i in range(len(data)):
            x.append(data[i][0])
            y.append(data[i][1])
        x, y = torch.cat(x), torch.cat(y)
        return x, y

    def save_model(self, filename):
        
        model_state = self.generator.state_dict()
        for key in model_state.keys():  # Always save it to cpu
            model_state[key] = model_state[key].cpu()
        self.log('=> Saving generator model to:', filename)
        torch.save(model_state, filename + 'generator.pth')
        super(DeepInversionGenBN, self).save_model(filename)

    def load_model(self, filename):
        self.generator.load_state_dict(torch.load(filename + 'generator.pth'))
        if self.gpu:
            self.generator = self.generator.cuda()
        self.generator.eval()
        super(DeepInversionGenBN, self).load_model(filename)

    def create_generator(self):
        cfg = self.config

        # Define the backbone (MLP, LeNet, VGG, ResNet ... etc) of model
        generator = models.__dict__[cfg['gen_model_type']].__dict__[cfg['gen_model_name']]()
        return generator

    def print_model(self):
        super(DeepInversionGenBN, self).print_model()
        self.log(self.generator)
        self.log('#parameter of generator:', self.count_parameter_gen())
    
    def reset_model(self):
        super(DeepInversionGenBN, self).reset_model()
        self.generator.apply(weight_reset)

    def count_parameter_gen(self):
        return sum(p.numel() for p in self.generator.parameters())

    def count_memory(self, dataset_size):
        return self.count_parameter() + self.count_parameter_gen() + self.memory_size * dataset_size[0]*dataset_size[1]*dataset_size[2]

    def cuda_gen(self):
        self.generator = self.generator.cuda()
        return self

    def sample(self, teacher, dim, device, return_scores=True):
        return teacher.sample(dim, device, return_scores=return_scores)

    def epoch_end(self):
        pass

class DeepInversionLWF(DeepInversionGenBN):

    def __init__(self, learner_config):
        super(DeepInversionLWF, self).__init__(learner_config)
        self.kl_loss = nn.KLDivLoss(reduction='batchmean').cuda()

    def update_model(self, inputs, targets, target_scores = None, dw_force = None, kd_index = None):
        
        loss_kd = torch.zeros((1,), requires_grad=True).cuda()

        if dw_force is not None:
            dw_cls = dw_force
        elif self.dw:
            dw_cls = self.dw_k[targets.long()]
        else:
            dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]

        # forward pass
        logits = self.forward(inputs)

        # classification 
        class_idx = np.arange(self.batch_size)
        loss_class = self.criterion(logits[class_idx], targets[class_idx].long(), dw_cls[class_idx])

        # KD
        if target_scores is not None:
            loss_kd = self.mu * loss_fn_kd(logits, target_scores, dw_cls, np.arange(self.last_valid_out_dim).tolist(), self.DTemp)

        total_loss = loss_class + loss_kd
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return total_loss.detach(), loss_class.detach(), loss_kd.detach(), logits

class AlwaysBeDreaming(DeepInversionGenBN):

    def __init__(self, learner_config):
        super(AlwaysBeDreaming, self).__init__(learner_config)
        self.kl_loss = nn.KLDivLoss(reduction='batchmean').cuda()

    def update_model(self, inputs, targets, target_scores = None, dw_force = None, kd_index = None):
        
        # class balancing
        mappings = torch.ones(targets.size(), dtype=torch.float32)
        if self.gpu:
            mappings = mappings.cuda()
        rnt = 1.0 * self.last_valid_out_dim / self.valid_out_dim
        mappings[:self.last_valid_out_dim] = rnt
        mappings[self.last_valid_out_dim:] = 1-rnt
        dw_cls = mappings[targets.long()]

        # forward pass
        logits_pen = self.model.forward(x=inputs, pen=True)
        if len(self.config['gpuid']) > 1:
            logits = self.model.module.last(logits_pen)
        else:
            logits = self.model.last(logits_pen)
        
        # classification 
        class_idx = np.arange(self.batch_size)
        if self.inversion_replay:

            # local classification
            loss_class = self.criterion(logits[class_idx,self.last_valid_out_dim:self.valid_out_dim], (targets[class_idx]-self.last_valid_out_dim).long(), dw_cls[class_idx]) 

            # ft classification  
            with torch.no_grad():             
                feat_class = self.model.forward(x=inputs, pen=True).detach()
            if len(self.config['gpuid']) > 1:
                loss_class += self.criterion(self.model.module.last(feat_class), targets.long(), dw_cls)
            else:
                loss_class += self.criterion(self.model.last(feat_class), targets.long(), dw_cls)
            
        else:
            loss_class = self.criterion(logits[class_idx], targets[class_idx].long(), dw_cls[class_idx])

        # KD
        if target_scores is not None:

            # hard - linear
            kd_index = np.arange(2 * self.batch_size)
            dw_KD = self.dw_k[-1 * torch.ones(len(kd_index),).long()]
            logits_KD = self.previous_linear(logits_pen[kd_index])[:,:self.last_valid_out_dim]
            logits_KD_past = self.previous_linear(self.previous_teacher.generate_scores_pen(inputs[kd_index]))[:,:self.last_valid_out_dim]
            loss_kd = self.mu * (self.kd_criterion(logits_KD, logits_KD_past).sum(dim=1) * dw_KD).mean() / (logits_KD.size(1))
        else:
            loss_kd = torch.zeros((1,), requires_grad=True).cuda()
            
        total_loss = loss_class + loss_kd
        self.optimizer.zero_grad()
        total_loss.backward()

        # step
        self.optimizer.step()

        return total_loss.detach(), loss_class.detach(), loss_kd.detach(), logits
    
#####################################
############### sp-kd ###############
#####################################

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
    
class ISCF(DeepInversionGenBN):

    def __init__(self, learner_config):
        super(ISCF, self).__init__(learner_config)
        #SPKD loss definition
        self.md_criterion = SP(reduction='none')


    def update_model(self, inputs, targets, target_scores = None, dw_force = None, kd_index = None):
        task_step=self.valid_out_dim-self.last_valid_out_dim
        # class balancing
        mappings = torch.ones(targets.size(), dtype=torch.float32)
        if self.gpu:
            mappings = mappings.cuda()

        rnt = 1.0 * self.last_valid_out_dim / self.valid_out_dim
        mappings[:self.last_valid_out_dim] = rnt
        mappings[self.last_valid_out_dim:] = 1-rnt
        dw_cls = mappings[targets.long()]

        # forward pass
        logits_pen,m = self.model.forward(inputs, middle=True)

        if len(self.config['gpuid']) > 1:
            logits = self.model.module.last(logits_pen)
        else:
            logits = self.model.last(logits_pen)
        
        # classification 
        class_idx = np.arange(self.batch_size) # real
        if self.inversion_replay:
            # local classification - LCE loss: the logit dimension is from last_valid_out_dim to valid_out_dim
            loss_class = self.criterion(logits[class_idx,self.last_valid_out_dim:self.valid_out_dim], (targets[class_idx]-self.last_valid_out_dim).long(), dw_cls[class_idx]) 
            
            # ft classification  
            if len(self.config['gpuid']) > 1:
                loss_class += self.criterion(self.model.module.last(logits_pen.detach()), targets.long(), dw_cls)
            else:
                loss_class += self.criterion(self.model.last(logits_pen.detach()), targets.long(), dw_cls)
        
        #first task local classification when we do not use any synthetic data     
        else:
            loss_class = self.criterion(logits[class_idx], targets[class_idx].long(), dw_cls[class_idx])
        
        add_index= np.arange(2*self.batch_size) # real n fake
        if self.previous_teacher: # after 2nd task
            with torch.no_grad():
                logits_prev, pm = self.previous_teacher.solver.forward(inputs[add_index],middle=True)
            #SPKD - Intermediate KD
            if len(pm)==3:
                out1_pm,out2_pm,out3_pm=pm
                out1_m,out2_m,out3_m=m
                loss_sp = (self.md_criterion(out1_m[add_index],out1_pm)+self.md_criterion(out2_m[add_index],out2_pm)+self.md_criterion(out3_m[add_index],out3_pm))/3.
            else: # for imagenet
                out1_pm,out2_pm,out3_pm,out4_pm=pm
                out1_m,out2_m,out3_m,out4_m=m
                loss_sp = (self.md_criterion(out1_m[add_index],out1_pm)+self.md_criterion(out2_m[add_index],out2_pm)+self.md_criterion(out3_m[add_index],out3_pm)+self.md_criterion(out4_m[add_index],out4_pm))/4.
            
            loss_sp = loss_sp.mean()*self.config['sp_mu']

            # Logit KD for maintaining the output probability 
            with torch.no_grad():
                # logits_prevpen = self.previous_teacher.solver.forward(inputs[add_index],pen=True)
                logits_prev=self.previous_linear(logits_prev)[:,:self.last_valid_out_dim].detach()

            loss_lkd=(F.mse_loss(logits[add_index,:self.last_valid_out_dim],logits_prev,reduction='none').sum(dim=1)) * self.mu / task_step
            loss_lkd=loss_lkd.mean()
        else:
            loss_sp=torch.zeros((1,), requires_grad=True).cuda()
            loss_lkd = torch.zeros((1,), requires_grad=True).cuda()

        # weight equalizer for balancing the average norm of weight 
        if self.previous_teacher:
            if len(self.config['gpuid']) > 1:
                last_weights=self.model.module.last.weight[:self.valid_out_dim,:].detach()
                last_bias=self.model.module.last.bias[:self.valid_out_dim].detach().unsqueeze(-1)
                cur_weights=self.model.module.last.weight[:self.valid_out_dim,:] 
                cur_bias=self.model.module.last.bias[:self.valid_out_dim].unsqueeze(-1) 
            else:
                last_weights=self.model.last.weight[:self.valid_out_dim,:].detach()
                last_bias=self.model.last.bias[:self.valid_out_dim].detach().unsqueeze(-1)
                cur_weights=self.model.last.weight[:self.valid_out_dim,:]
                cur_bias=self.model.last.bias[:self.valid_out_dim].unsqueeze(-1) 

            last_params=torch.cat([last_weights,last_bias],dim=1)
            cur_params=torch.cat([cur_weights,cur_bias],dim=1)
            weq_regularizer=F.mse_loss(last_params.norm(dim=1,keepdim=True).mean().expand(self.valid_out_dim),cur_params.norm(dim=1))
            weq_regularizer*=self.config['weq_mu']
        else:
            weq_regularizer=torch.zeros((1,),requires_grad=True).cuda()

        # calculate the 5 losses - LCE + SPKD + LKD + FT + WEQ, loss_class include the LCE and FT losses
        total_loss = loss_class + loss_lkd + loss_sp + weq_regularizer

        self.optimizer.zero_grad()
        total_loss.backward()
        # step
        self.optimizer.step()

        return total_loss.detach(), loss_class.detach(), (loss_lkd.detach(), loss_sp.detach(), weq_regularizer.detach()), logits
    
class RDFCIL(DeepInversionGenBN):
    def __init__(self, learner_config):
        super(RDFCIL, self).__init__(learner_config)
        self.rkd = RKDAngleLoss(
            self.model.in_planes, # backbone num feature
            proj_dim=2 * self.model.in_planes, # backbone num feature
        ).to(self.device)
        if len(self.config['gpuid']) > 1:
            self.cls_count = torch.zeros(self.model.module.last.weight.shape[0])
            self.cls_weight = torch.ones(self.model.module.last.weight.shape[0]).to(self.device)
        else:
            self.cls_count = torch.zeros(self.model.last.weight.shape[0])
            self.cls_weight = torch.ones(self.model.last.weight.shape[0]).to(self.device)

        self.ce_mu= self.config['ce_mu']

    def update_model(self, inputs, targets, target_scores = None, dw_force = None, kd_index = None):
        task_step=self.valid_out_dim-self.last_valid_out_dim

        rdfcil_alpha=np.log2(self.config['num_classes']/2+1)
        rdfcil_beta2=self.last_valid_out_dim / self.config['num_classes']
        rdfcil_beta=np.sqrt(rdfcil_beta2)


        # class balancing
        mappings = torch.ones(targets.size(), dtype=torch.float32)
        if self.gpu:
            mappings = mappings.cuda()

        rnt = 1.0 * self.last_valid_out_dim / self.valid_out_dim
        mappings[:self.last_valid_out_dim] = 1 # rnt
        mappings[self.last_valid_out_dim:] = 1 # 1-rnt
        dw_cls = mappings[targets.long()]

        # forward pass
        logits_pen, m = self.model.forward(inputs, middle=True)

        if len(self.config['gpuid']) > 1:
            logits = self.model.module.last(logits_pen)
        else:
            logits = self.model.last(logits_pen)
        
        # classification 
        class_idx = np.arange(self.batch_size) # real
        if self.inversion_replay:
            # local classification - LCE loss: the logit dimension is from last_valid_out_dim to valid_out_dim
            loss_class = self.criterion(logits[class_idx,self.last_valid_out_dim:self.valid_out_dim], (targets[class_idx]-self.last_valid_out_dim).long(), dw_cls[class_idx]) * (1+1/rdfcil_alpha) / rdfcil_beta * self.ce_mu
            
            # # ft classification  
            # if len(self.config['gpuid']) > 1:
            #     loss_class += self.criterion(self.model.module.last(logits_pen.detach()), targets.long(), self.cls_weight[:self.valid_out_dim])
            # else:
            #     loss_class += self.criterion(self.model.last(logits_pen.detach()), targets.long(), self.cls_weight[:self.valid_out_dim])
        #first task local classification when we do not use any synthetic data     
        else:
            loss_class = self.criterion(logits[class_idx], targets[class_idx].long(), dw_cls)


        #first task local classification when we do not use any synthetic data     
        add_index= np.arange(2*self.batch_size) # real n fake
        if self.previous_teacher: # after 2nd task
            with torch.no_grad():
                logits_prev_pen, pm = self.previous_teacher.solver.forward(inputs[add_index],middle=True)
                # logits_prevpen = self.previous_teacher.solver.forward(inputs[add_index],pen=True)
                logits_prev=self.previous_linear(logits_prev_pen)[:,:self.last_valid_out_dim].detach()

            loss_hkd = F.l1_loss(logits[self.batch_size:,:self.last_valid_out_dim],logits_prev[self.batch_size:,:self.last_valid_out_dim]) * self.mu * rdfcil_alpha * rdfcil_beta

            loss_rkd = self.rkd(logits_pen[class_idx],logits_prev_pen[class_idx]).mean() * self.config['rkd_mu'] * rdfcil_alpha * rdfcil_beta

        else:
            loss_hkd = torch.zeros((1,), requires_grad=True).cuda()
            loss_rkd = torch.zeros((1,), requires_grad=True).cuda()

        total_loss = loss_class + loss_hkd + loss_rkd

        self.optimizer.zero_grad()
        total_loss.backward()
        # step
        self.optimizer.step()

        indices, counts = targets.cpu().unique(return_counts=True)
        self.cls_count[indices] += counts


        return total_loss.detach(), loss_class.detach(), loss_hkd.detach() + loss_rkd.detach(), logits


    def finetuning_update_model(self, x_com, y_com):
        rdfcil_alpha=np.log2(self.config['num_classes']/2+1)
        rdfcil_beta2=self.last_valid_out_dim / self.config['num_classes']
        rdfcil_beta=np.sqrt(rdfcil_beta2)
        

        # forward pass
        logits = self.forward(x_com)

        # classification 
        indices, counts = y_com.cpu().unique(return_counts=True)
        self.cls_count[indices] += counts
        loss = 0
        if self.previous_teacher:
            with torch.no_grad():
                logits_prev_pen, pm = self.previous_teacher.solver.forward(x_com,middle=True)
                # logits_prevpen = self.previous_teacher.solver.forward(inputs[add_index],pen=True)
                logits_prev=self.previous_linear(logits_prev_pen)[:,:self.last_valid_out_dim].detach()

            loss_hkd = F.l1_loss(logits[:,:self.last_valid_out_dim],logits_prev[:,:self.last_valid_out_dim]) * self.mu * rdfcil_alpha * rdfcil_beta
            loss += loss_hkd

        loss += F.cross_entropy(logits[:,:self.valid_out_dim], y_com.long(), self.cls_weight.to(self.device)[:self.valid_out_dim]) * self.ce_mu
        return loss

    def epoch_end(self):
        cls_weight = self.cls_count.sum() / self.cls_count.clamp(min=1)
        self.cls_weight = cls_weight.div(cls_weight.min()).to(self.device)