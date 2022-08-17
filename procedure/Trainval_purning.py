from collections import OrderedDict
import os,sys
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

sys.path.append("/home/yenanfei/")
from pytorch_prune.pruning import create_sparse_table,pruning

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


class Trainval_purning:
    def __init__(self,
                model,
                supervised_list,
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
                lr_policy=lambda epoch:(0.1 ** (epoch // 10)),
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
                lowrank_snapshot_path=None,
                binary_snapshot_path=None,
                pruning_snapshot_path=None):

        self.model=model
        self.supervised_list=supervised_list
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
        self.test_initialization=test_initialization
        self.lowrank= lowrank
        self.binary = binary
        self.lowrank_snapshot_path = lowrank_snapshot_path
        self.binary_snapshot_path = snapshot_path
        self.pruning_snapshot_path = pruning_snapshot_path
        glog.info("==>>> supervised train batches:{}".format(self.train_batches))
        if self.is_acc:
            self.best_test_loss_0=-np.inf
        else:
            self.best_test_loss_0=np.inf
        logger_dir=os.path.join(logger_dir,"{}/{}".format(model_name,datetime.now().strftime("%Y%m%d-%H%M%S")))
        if os.path.isdir(logger_dir):
            os.mkdir(logger_dir)
        self.logger = SummaryWriter(logger_dir)
        if snapshot_path is None:
            snapshot_path=os.path.join('snapshot/',model_name+'.pth')
           
            # snapshot_path=os.path.join('snapshot/','phone_128_lowrank'+'.pth')

        self.snapshot_path=snapshot_path
        # print(self.model)
        if restore_path is not None :
            glog.info("specify snapshot {}".format(restore_path))
            self.model.load_state_dict(torch.load(restore_path),strict=restore_strict)
        if restore_path is None  and os.path.isfile(self.snapshot_path):
            glog.info("restore snapshot {}".format(self.snapshot_path))
            
            if hasattr(self.model,'module'):
                print(1)
                self.model.module.load_state_dict(torch.load(self.snapshot_path),strict=True)
                 #### lowrank
                if self.lowrank:
                    
                    torch_lowrank_layers(model.cpu(),has_bn=False)

                    print(model)
                    model = model.cuda()

                    if self.binary:
                        lowrank_snapshot_path = self.lowrank_snapshot_path
                        self.model.module.load_state_dict(torch.load(lowrank_snapshot_path),strict=True)

                        input_shape=(1,1,128,128)
                        sparse_steps = [0.65, 0.7, 0.75, 0.80,0.85, 0.88, 0.90,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99]
                        sparse_dict_list = {k:v for k,v in zip(sparse_steps,create_sparse_table(model,input_shape,DEFAULT_SPARSE_STEP=sparse_steps))}
                        model.module.load_state_dict(torch.load(self.pruning_snapshot_path),strict=False)
                        total_sparse,sparse_dict = sparse_dict_list[0.91]
                        glog.info("curr total sparse:{}".format(total_sparse))
                        pruning(model,sparse_dict)
                        
                        model = model.cuda()
            else:
                print(2)
                self.model.load_state_dict(torch.load(self.snapshot_path),strict=True)
                if self.lowrank:
                    torch_lowrank_layers(model.cpu(),has_bn=False)
                    model = model.cuda()
                    if self.binary:
                        lowrank_snapshot_path = self.lowrank_snapshot_path
                        self.model.module.load_state_dict(torch.load(lowrank_snapshot_path),strict=True)

                        torch_binarize_layers(model.cpu())
                        model = model.cuda()
                

        assert solver_type in ["adam","radam","sgd"]
        if solver_type=='adam':
            self.optimizer = optim.Adam(self.model.parameters(),lr=self.init_lr, betas=betas,weight_decay=weight_decay)
        elif solver_type=='radam':
            from optim import RAdam
            self.optimizer = RAdam(self.model.parameters(),lr=self.init_lr, betas=betas,weight_decay=weight_decay,degenerated_to_sgd=degenerated_to_sgd)
        else:
            self.optimizer = optim.SGD(self.model.parameters(), lr = self.init_lr, momentum = momentum,weight_decay=weight_decay)
        print(self.optimizer)
        self.supervised_dataloader_iterators=[]
        for train_dataloader,criterion in self.supervised_list:
            assert isinstance(train_dataloader,DataLoader)
            self.supervised_dataloader_iterators.append(iter(train_dataloader))



    def _val(self):
        model=self.model
        model.eval()


        metric_dict=OrderedDict()
        with torch.no_grad():
            for model_criterion in self.model_eval_list:
                eval_dict=model_criterion(model)
                for eval_name in eval_dict:
                    metric_dict[eval_name]=eval_dict[eval_name]
            val_loss_name=[]
            for val_dataloader,val_criterion in self.data_eval_list:
                for data,gt_labels in val_dataloader:
                    y = model(data)
                    loss_dict=val_criterion(y,gt_labels)
                    for loss_name in loss_dict:
                        if loss_name not in metric_dict:
                            metric_dict[loss_name]=loss_dict[loss_name]
                            val_loss_name.append(loss_name)
                        else:
                            metric_dict[loss_name]=metric_dict[loss_name]+loss_dict[loss_name]
            for loss_name in val_loss_name:
                metric_dict[loss_name]=metric_dict[loss_name]/len(val_dataloader)
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
        model=self.model
        

        metric_dict=OrderedDict()
        self._adjust_lr(epoch)
        ave_loss = 0
        model.train()


        for batch_idx in range(self.train_batches):
            current_iter=epoch*self.train_batches+batch_idx

            loss=0
            for i,data_iterator in enumerate(self.supervised_dataloader_iterators):
                data,gt_labels=self.next_batch(self.supervised_dataloader_iterators,self.supervised_list,i)

                y=model(data)
                loss_dict=self.supervised_list[i][1](y,gt_labels)

                for loss_name in loss_dict:
                    loss_weight=1.0
                    if 'loss_seg'in loss_name:
                        loss_weight=1000.0
                    # if 'ssd_loc' == loss_name:
                    #     continue
                    loss+=loss_dict[loss_name]*loss_weight

                    if batch_idx % self.display == 0 or batch_idx == self.train_batches:
                            if loss_name not in metric_dict:
                                metric_dict[loss_name]=loss_dict[loss_name].data.item()
                            else:
                                metric_dict[loss_name]=metric_dict[loss_name]* 0.9+loss_dict[loss_name].data.item()* 0.1
                                   
            loss=loss/self.iter_size
            loss.backward()

            # torch.cuda.empty_cache()
            if batch_idx%self.iter_size==0:
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




    def start(self):

        if self.test_initialization:
            test_metric_losses_dict=self._val()
            self.best_test_loss_0=test_metric_losses_dict[self.key_metric_name]
            metric_text=lambda metric_name:"{}:{:.6f}".format(metric_name,test_metric_losses_dict[metric_name])
            glog.info('==>>> test_initialization:current best '+" ".join(list(map(metric_text,test_metric_losses_dict))))
        for epoch in tqdm(range(0, self.num_epoches)):
            
            self._train(epoch)
            test_metric_losses_dict=self._val()
            metric_text=lambda metric_name:"{}:{:.6f}".format(metric_name,test_metric_losses_dict[metric_name])
            glog.info('==>>> epoch: {} '.format(epoch)+" ".join(list(map(metric_text,test_metric_losses_dict))))
            for metric_name in test_metric_losses_dict:
                self.logger.add_scalar('{}/{}'.format(self.model_name,metric_name), test_metric_losses_dict[metric_name], epoch*self.train_batches)
            
            if self.always_snapshot:
                snapshot_path = os.path.join('snapshot/','phone_128_lowrank_purning_0.91_newest'+'.pth')
                if hasattr(self.model,'module'):
                    torch.save(self.model.module.state_dict(),snapshot_path)
                else:
                    torch.save(self.model.state_dict(),snapshot_path)


            if (test_metric_losses_dict[self.key_metric_name]<=self.best_test_loss_0 and not self.is_acc) or (test_metric_losses_dict[self.key_metric_name]>=self.best_test_loss_0 and self.is_acc): #or self.always_snapshot:
                self.best_test_loss_0=test_metric_losses_dict[self.key_metric_name]
                ###
                snapshot_path = os.path.join('snapshot/','phone_128_lowrank_purning_0.91_best'+'.pth')
                ###
                glog.info("==>>> saving current best model state dict {} with {} {:.6f}".format(snapshot_path,self.key_metric_name,self.best_test_loss_0))
                self.logger.add_scalar('{}/{}'.format(self.model_name,self.key_metric_name), self.best_test_loss_0, epoch*self.train_batches)
                if hasattr(self.model,'module'):
                    torch.save(self.model.module.state_dict(),snapshot_path)
                else:
                    torch.save(self.model.state_dict(),snapshot_path)