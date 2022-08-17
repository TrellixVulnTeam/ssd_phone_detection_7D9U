from collections import OrderedDict
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import glog
import torch.optim as optim
import numpy as np
from tqdm.auto import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data.dataloader import DataLoader
# from custom_nn import torch_mergebn_layers
from datetime import datetime
from functools import reduce
import re
import inspect
from sync_batchnorm import convert_model
from collections import OrderedDict
from cnnc_util import torch_lowrank_layers,torch_binarize_layers,torch_mergebn_layers,freeze,unfreeze
import math
from cnnc_util import IRConv2d
def get_state_dict(path):
    new_params=OrderedDict()
    for key,value in torch.load(path).items():
      new_params[key[7:]]=value
    return new_params

def dict2str(metric_dict):
    return " ".join(list(map(lambda key:'{}:{:.6f}'.format(key,metric_dict[key]),metric_dict)))

def NamedOutput(func):
    code =inspect.getsource(func).strip()
    assert 'return' in code
    var_names=list(map(lambda x:x.strip(),re.findall(r'return (.*)',code)[0].split(',')))
    if len(var_names)>1:
        def wrapper(*args,**kw):
            outputs=func(*args,**kw)
            return OrderedDict(zip(var_names,outputs))
        return  wrapper
    else:
        def wrapper(*args,**kw):
            outputs=func(*args,**kw)
            return OrderedDict(zip(var_names,(outputs,)))
        return  wrapper


class TrainVal3:
    def __init__(self,
                model,
                model_binary,
                supervised_list,
                supervised_list_val,
                model_eval_list=[],
                data_eval_list=[],
                key_metric_name=None,
                is_acc=False,
                num_epoches=100,
                display=10,
                model_name='model',
                snapshot_path=None,
                restore_path=None,
                init_lr=0.001,
                momentum=0.9,
                betas=(0.9, 0.999),
                test_initialization=True,
                degenerated_to_sgd=True,
                lr_policy=lambda epoch:(0.1 ** (epoch // 20)),
                test_itervals=0,
                iter_size=1,
                weight_decay=0.0005,
                reporter=None,
                logger_dir="logs",
                always_snapshot=False,
                restore_strict=True,
                solver_type='adam',
                model_placing=True,merge_bn=False,dist=False,
                lowrank=False,
                binary=False,
                distillation = False,
                layer_pairs=[],
                lowrank_snapshot_path=None,
                binary_snapshot_path=None,
                cosin=False,
                fp16=False):

        self.model=model
        self.fp16 = fp16
        self.model_binary = model_binary
        self.supervised_list=supervised_list
        self.supervised_list_val=supervised_list_val

        self.model_eval_list=model_eval_list
        self.data_eval_list=data_eval_list
        self.key_metric_name=key_metric_name
        self.num_epoches=num_epoches
        self.display=display
        self.model_name=model_name
        self.always_snapshot=always_snapshot
        self.init_lr=init_lr
        self.lr_policy=lr_policy
        self.is_acc=is_acc
        self.test_itervals=test_itervals
        self.iter_size=iter_size
        self.train_batches=self.test_itervals if self.test_itervals>0 else sum(list(map(lambda data_criterion:len(data_criterion[0]),self.supervised_list)))
        self.test_batches=self.test_itervals if self.test_itervals>0 else sum(list(map(lambda data_criterion:len(data_criterion[0]),self.supervised_list_val)))

        self.test_initialization=test_initialization
        self.lowrank= lowrank
        self.binary = binary
        self.lowrank_snapshot_path = lowrank_snapshot_path
        self.binary_snapshot_path = snapshot_path
        self.layer_pairs = layer_pairs
        self.distillation = distillation
        self.cosin= cosin
        glog.info("==>>> supervised train batches:{}".format(self.train_batches))
        if self.is_acc:
            self.best_test_loss_0=-np.inf
        else:
            self.best_test_loss_0=np.inf
        logger_dir=os.path.join(logger_dir,"{}/{}".format(model_name,datetime.now().strftime("%Y%m%d-%H%M%S")))
        if os.path.isdir(logger_dir):
            os.mkdir(logger_dir)
        self.logger = SummaryWriter(logger_dir)
        
        # if snapshot_path is None:
            # snapshot_path=os.path.join('snapshot/phone_128/',model_name+'.pth')
           
        #     # snapshot_path=os.path.join('snapshot/','phone_128_lowrank'+'.pth')

        self.snapshot_path='/home/disk/yenanfei/DMS_phone/phone_model_pytorch/snapshot/phone_128/phone_128_float_continue_best.pth'

        if restore_path is not None :
            glog.info("specify snapshot {}".format(restore_path))
            self.model.load_state_dict(torch.load(restore_path),strict=restore_strict)
        if restore_path is None  and os.path.isfile(self.snapshot_path):
            # glog.info("restore snapshot {}".format(self.snapshot_path))
            
            if hasattr(self.model,'module'):
                print(1)
            else:
                print(2)
                

        assert solver_type in ["adam","radam","sgd",'AdaBelief']
        
        model_to_optimize = self.model_binary if self.binary else self.model
        # model_to_train = model_to_optimize


        if solver_type=='adam':
            self.optimizer = optim.Adam(model_to_optimize.parameters(),lr=self.init_lr, betas=betas,weight_decay=weight_decay)
            # self.optimizer.load_state_dict(torch.load('/home/yenanfei/ssd_pytorch/snapshot/phone_128/layer_by_layer/fc7_mbox/optimizer_best.pth'))
            
        elif solver_type=='radam':
            from optim import RAdam
            self.optimizer = RAdam(model_to_optimize.parameters(),lr=self.init_lr, betas=betas,weight_decay=weight_decay,degenerated_to_sgd=degenerated_to_sgd)
            # self.optimizer.load_state_dict(torch.load('/home/yenanfei/ssd_pytorch/snapshot/phone_128_addChanel/optimizer_best.pth'))
        elif solver_type=='AdaBelief':
            from adabelief_pytorch import AdaBelief
            self.optimizer = AdaBelief(model_to_optimize.parameters(),lr= self.init_lr,betas=betas,eps=1e-3,weight_decay=weight_decay,amsgrad=False,weight_decouple=True,fixed_decay=False,rectify=False,)
        else:
            self.optimizer = optim.SGD(model_to_optimize.parameters(), lr = self.init_lr, momentum = momentum,weight_decay=weight_decay)
            # self.optimizer.load_state_dict(torch.load('/home/yenanfei/ssd_pytorch/snapshot/phone_128_addChanel/optimizer_best.pth'))

        print(self.optimizer)

        self.supervised_dataloader_iterators=[]
        self.supervised_test_dataloader_iterators=[]
        
        for train_dataloader,criterion in self.supervised_list:
            assert isinstance(train_dataloader,DataLoader)
            self.supervised_dataloader_iterators.append(iter(train_dataloader))

        for test_dataloader,test_criterion in self.supervised_list_val:
            assert isinstance(test_dataloader,DataLoader)
            self.supervised_test_dataloader_iterators.append(iter(test_dataloader))

        if self.cosin:
            scheduler= optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=5, T_mult=2)
            self.optimizer = scheduler
    def _val(self,model_to_val):
        # model=self.model
        # model.eval()
        # model_binary = self.model_binary
        model_to_val.eval()

        # model_to_val = model_binary if self.binary else model

        metric_dict=OrderedDict()
        with torch.no_grad():
            for model_criterion in self.model_eval_list:
                eval_dict=model_criterion(model_to_val)
                for eval_name in eval_dict:
                    metric_dict[eval_name]=eval_dict[eval_name]
            val_loss_name=[]

            # for val_dataloader,val_criterion in self.data_eval_list:
            #     for data,gt_labels in val_dataloader:
            #         # y=model(data)
            #         y = model_to_val(data)
            #         loss_dict=val_criterion(y,gt_labels)
            #         for loss_name in loss_dict:
            #             if loss_name not in metric_dict:
            #                 metric_dict[loss_name]=loss_dict[loss_name]
            #                 val_loss_name.append(loss_name)
            #             else:
            #                 metric_dict[loss_name]=metric_dict[loss_name]+loss_dict[loss_name]
            # for loss_name in val_loss_name:
            #     metric_dict[loss_name]=metric_dict[loss_name]/len(val_dataloader)

        return metric_dict



    def next_batch(self,dataloader_iters,dataloader_criterions,i):
        try:
            return next(dataloader_iters[i])
        except StopIteration:
            dataloader_iters[i]=iter(dataloader_criterions[i][0])
            return next(dataloader_iters[i])



    def _adjust_lr(self,epoch):
        lr = self.init_lr*self.lr_policy(epoch)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr



    def _train(self,epoch):

        metric_dict=OrderedDict()
        self._adjust_lr(epoch)
        ave_loss = 0
        
        model_to_train = self.model_binary if self.binary else self.model
        model_to_train.train()
        # if self.distillation:
        #     model.eval()

        for batch_idx in range(self.train_batches):
            current_iter=epoch*self.train_batches+batch_idx

            loss=0
            for i,data_iterator in enumerate(self.supervised_dataloader_iterators):
                data,gt_labels=self.next_batch(self.supervised_dataloader_iterators,self.supervised_list,i)
                
                # if self.distillation:
                #     with torch.no_grad():
                #         y=model(data)\
                if self.fp16:
                    data = data.half()
                y_binary= model_to_train(data)
                # y_binary = list(map(lambda x: x.float(),y_binary))
                
                # loss_dict=self.supervised_list[i][1](y,gt_labels)

                loss_dict=self.supervised_list[i][1](y_binary,gt_labels)

                for loss_name in loss_dict:
                    loss_weight=1.0
                    if 'loss_seg'in loss_name:
                        loss_weight=1000.0
                    elif 'ssd_Giou'in loss_name:
                        loss_weight=1.0
                    # if 'ssd_loc' == loss_name:
                    #     continue
                    loss+=loss_dict[loss_name]*loss_weight

                    if batch_idx % self.display == 0 or batch_idx == self.train_batches:
                            if loss_name not in metric_dict:
                                metric_dict[loss_name]=loss_dict[loss_name].data.item()
                            else:
                                metric_dict[loss_name]=metric_dict[loss_name]* 0.9+loss_dict[loss_name].data.item()* 0.1
                
                # vec = torch.cat([x.view(x.size(0),-1) for x in vec],1)
                # vec_binary = torch.cat([x.view(x.size(0),-1) for x in vec_binary],1)


                # loss += F.smooth_l1_loss(vec_binary,vec.detach())
            if self.distillation:
                distil =reduce(lambda x,y:x+y,map(lambda ts:F.smooth_l1_loss(ts[0].output,ts[1].output),self.layer_pairs))/float(len(self.layer_pairs))
                loss=loss+distil*0.5
                metric_dict['distil'] = distil
            loss=loss/self.iter_size
           
            loss.backward()

            # torch.cuda.empty_cache()
            if batch_idx%self.iter_size==0:
                if self.cosin:
                    self.optimizer.step(epoch)
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()#warn
                #torch.cuda.empty_cache()

                if batch_idx==self.iter_size-1:
                    ave_loss=loss.data.item()*self.iter_size
                else:
                    ave_loss = ave_loss * 0.9 + loss.data.item()*self.iter_size * 0.1



                if batch_idx % self.display == 0 or batch_idx == self.train_batches:
                    glog.info('==>>> epoch: {},batch index: {} ,lr:{} ,train loss: {:.6f} {}'.format(epoch,batch_idx, self.optimizer.param_groups[0]['lr'],ave_loss,dict2str(metric_dict)))
                    self.logger.add_scalar('{}/train_loss'.format(self.model_name), ave_loss, current_iter)
                    for metric_name in metric_dict:
                        self.logger.add_scalar('{}/{}'.format(self.model_name,metric_name), metric_dict[metric_name], current_iter)
        return model_to_train
        
       


    def _test(self,epoch,model_to_train):
        with torch.no_grad():
            metric_dict=OrderedDict()
            ave_loss = 0
            
            model_to_test = model_to_train
            model_to_test.eval()
            # if self.distillation:
            #     model.eval()
            val_loss_name=[]
            
            for val_dataloader,val_criterion in self.data_eval_list:
                loss=0
                for data,gt_labels in val_dataloader:
                    

                    y_binary= model_to_test(data)
                
                    # loss_dict=self.supervised_list[i][1](y,gt_labels)

                    loss_dict=val_criterion(y_binary[-1],gt_labels)

                    for loss_name in loss_dict:
                        loss+=loss_dict[loss_name]
                        if loss_name not in metric_dict:
                            metric_dict[loss_name]=float(loss_dict[loss_name].data.detach().item())
                            val_loss_name.append(loss_name)
                        else:
                            metric_dict[loss_name]=metric_dict[loss_name]+float(loss_dict[loss_name].data.detach().item())
                
                for loss_name in val_loss_name:
                    metric_dict[loss_name]=metric_dict[loss_name]/len(val_dataloader)
                ave_loss=loss/len(val_dataloader)
                glog.info('==>>> epoch: {} ,lr:{} ,test loss: {:.6f} {}'.format(epoch, self.optimizer.param_groups[0]['lr'],ave_loss,dict2str(metric_dict)))
                self.logger.add_scalar('{}/test_loss'.format(self.model_name), ave_loss, epoch)


    def start(self):
        T_min, T_max = 1e-1, 1e1

        def Log_UP(K_min, K_max, epoch):
            Kmin, Kmax = math.log(K_min) / math.log(10), math.log(K_max) / math.log(10)
            return torch.tensor([math.pow(10, Kmin + (Kmax - Kmin) / 200 * epoch)]).float().cuda()


        if self.test_initialization:
            # self._test(epoch=0)
            test_metric_losses_dict=self._val(self.model)
            self.best_test_loss_0=test_metric_losses_dict[self.key_metric_name]
            metric_text=lambda metric_name:"{}:{:.6f}".format(metric_name,test_metric_losses_dict[metric_name])
            glog.info('==>>> test_initialization:current best '+" ".join(list(map(metric_text,test_metric_losses_dict))))
        
        model_to_train = self.model_binary if self.binary else self.model
        for epoch in tqdm(range(0, self.num_epoches)):
            t = Log_UP(T_min, T_max, epoch)
            if (t < 1):
                k = 1 / t
            else:
                k = torch.tensor([1]).float().cuda()
            def assign_k(m):
                if hasattr(m,'k'):
                    m.k=k
                if hasattr(m,'t'):
                    m.t=t
                    
            
            model_to_train.apply(assign_k)

            # for key in self.model_binary.module._modules.keys():
            #     if (isinstance(self.model_binary.module._modules[key],IRConv2d)):
            #         layer=self.model_binary.module._modules[key]
            #         self.model_binary.module._modules[key].k = k
            #         self.model_binary.module._modules[key].t = t
            #     else:
            #         seq=self.model_binary.module._modules[key]
            #         for key in seq._modules.keys():
            #             if (isinstance(seq._modules[key],IRConv2d)):
            #                 seq._modules[key].k = k
            #                 seq._modules[key].t = t

            model_to_train = self._train(epoch)
            
            self._test(epoch,model_to_train)
            test_metric_losses_dict=self._val(model_to_train)
            metric_text=lambda metric_name:"{}:{:.6f}".format(metric_name,test_metric_losses_dict[metric_name])
            glog.info('==>>> epoch: {} '.format(epoch)+" ".join(list(map(metric_text,test_metric_losses_dict))))
            for metric_name in test_metric_losses_dict:
                self.logger.add_scalar('{}/{}'.format(self.model_name,metric_name), test_metric_losses_dict[metric_name], epoch*self.train_batches)
            
            if self.always_snapshot:
                snapshot_path = os.path.join('/home/disk/qizhongpei/ssd_pytorch/weights/','phone_newest'+str(epoch)+'.pth')#+str(epoch)
                if hasattr(model_to_train,'module'):
                    torch.save(model_to_train.module.state_dict(),snapshot_path)
                    # torch.save(self.optimizer.state_dict(),'/home/yenanfei/ssd_pytorch/snapshot/phone_128/optimizer_newest.pth')
                else:
                    torch.save(model_to_train.state_dict(),snapshot_path)
                    # torch.save(self.optimizer.state_dict(),'/home/yenanfei/ssd_pytorch/snapshot/phone_128/optimizer_newest.pth')


            if (test_metric_losses_dict[self.key_metric_name]<=self.best_test_loss_0 and not self.is_acc) or (test_metric_losses_dict[self.key_metric_name]>=self.best_test_loss_0 and self.is_acc): #or self.always_snapshot:
                self.best_test_loss_0=test_metric_losses_dict[self.key_metric_name]
                ###
                snapshot_path = os.path.join('/home/disk/qizhongpei/ssd_pytorch/weights/','phone_best'+'.pth')
                ###
                glog.info("==>>> saving current best model state dict {} with {} {:.6f}".format(snapshot_path,self.key_metric_name,self.best_test_loss_0))
                self.logger.add_scalar('{}/{}'.format(self.model_name,self.key_metric_name), self.best_test_loss_0, epoch*self.train_batches)
                if hasattr(model_to_train,'module'):
                    torch.save(model_to_train.module.state_dict(),snapshot_path)
                    # torch.save(self.optimizer.state_dict(),'/home/yenanfei/ssd_pytorch/snapshot/phone_128/optimizer_best.pth')
                else:
                    torch.save(model_to_train.state_dict(),snapshot_path)
                    # torch.save(self.optimizer.state_dict(),'/home/yenanfei/ssd_pytorch/snapshot/phone_128/optimizer_best.pth')