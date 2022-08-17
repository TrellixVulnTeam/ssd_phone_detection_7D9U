import os
import argparse
import sys
import torch
import glog
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import cv2
from tqdm.auto import tqdm
from torchsummary import summary
from layers.modules import MultiBoxLoss

from metrics.AverageMeter import AverageMeter
from metrics.test_net import test_net
from collections import OrderedDict
from datasets.voc_data import VOCFormatDetectionDataset
from procedure.TrainVal3 import TrainVal3
# from procedure.Trainval_purning import Trainval_purning
import yaml
from cnnc_util import torch_lowrank_layers,torch_binarize_layers,torch_mergebn_layers,freeze,unfreeze
import custom_nn as custom_nn
def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('-s','--show', dest='show',default=False,
                        type=bool)
    parser.add_argument('-imgs','--image_seq', dest='image_seq',default="",
                        type=str)
    # parser.add_argument('-vid','--video', dest='video',default="/home/disk/tanjing/projects/tianmai_bsd/2020_04_29_15_31_43",
    #                     type=str)
    parser.add_argument('-vis_conf','--visiable_confidence', dest='vis_conf',default=0.3,
                        type=float)
    parser.add_argument('-svid','--save_video', dest='save_video',default="result.avi",
                        type=str)
    parser.add_argument('-disp','--display', dest='display',default=100,
                        type=int)
    parser.add_argument('-d','--devices', dest='devices',nargs='+',
                        help='devices',default=[0]#3,4,7
    )
    parser.add_argument('-is','--iter_size', dest='iter_size',
                        help='iter size',default=10,type=int)

    parser.add_argument('-i','--input_shape',dest='input_shape',nargs='+',
                        help='input shape',default=[1,128,128],
                        type=int)
    parser.add_argument('-m','--input_mean',dest='input_mean',nargs='+',
                        help='input mean',default=[128,128,128],
                        type=int)
    parser.add_argument('-opt','--optimizer',dest='optimizer',
                        help='optimizer',default='adam',
                        type=str)
    parser.add_argument('-wd','--weight_decay',dest='weight_decay',
                        help='weight decay',default=0.00036,
                        type=float)
    parser.add_argument('-lr','--learning_rate',dest='learning_rate',
                        help='learning rate',default=0.001,
                        type=float)
    parser.add_argument('-ep','--epochs',dest='epochs',
                        help='epochs',default=50,
                        type=int)
    parser.add_argument('-tbz','--train_batch_size',dest='train_batch_size',
                        help='train batch size',default=64,
                        type=int)
    parser.add_argument('-b','--backbone', dest='backbone',default='VGG16',
                        type=str)
    parser.add_argument('-iou','--iou_thresh', dest='iou_thresh',default=0.5,
                        type=float)
    parser.add_argument('-ef','--eval_first', dest='eval_first',default=False,
                        type=bool)
    parser.add_argument('-sd','--snapshot_dir', dest='snapshot_dir',default='snapshot',
                        type=str)
    parser.add_argument('-n','--name', dest='name',default='BSD_SSD_256',
                        type=str)
    parser.add_argument('-sync','--sync_bn', dest='sync_bn',default=False,
                        type=bool)
    parser.add_argument('--dist', default=False, type=bool, dest='dist',
                        help='distributed training')
    parser.add_argument('--local_rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--train_aug', default=True, type=bool,
                        help='training augmentation')
    parser.add_argument('--only_vista', default=False, type=bool,
                        help='only_vista')
    parser.add_argument('--aug_config', default='./data/hyp.scratch.yaml', type=str,
                        help='only_vista')
    parser.add_argument('--restore',  dest='restore_snapshot',default=None, type=str,
                        help='restore_snapshot')
    parser.add_argument('--use_atss',  default=False, type=bool,
                        help='use_atss')
                        
    parser.add_argument('--fp16',  default=False, type=bool,
                        help='fp16')


    parser.add_argument('--lowrank',action="store_true")
                    
    parser.add_argument('--prune',action="store_true")
                        
    parser.add_argument('--binary',action="store_true")
                        
    parser.add_argument('--distillation',action="store_true")
                        
    args = parser.parse_args()

    return args

def valid_select(preds,labels,concat=True):
    with torch.no_grad():
        targets = [label.unsqueeze(0).cuda(non_blocking=True) for label in labels if label is not None]
        select_indices = list(filter(lambda i: labels[i] is not None, range(len(labels))))
    if concat:
        return preds[select_indices],torch.cat(targets,dim=0)
    else:
        return preds[select_indices],targets


def get_sparse_table_from_pruned_model(model):
    state_dict=model.state_dict()
    sparse_dict=OrderedDict()
    for key in state_dict:
        if ('deeplab_head' not in key) and ('weight' in key):
            weight=state_dict[key].cpu().numpy()
            if 'loc' in key or 'conf' in key:
                sparse_dict[key.replace('.weight','')]=0.0
            else:
                sparse_dict[key.replace('.weight','')]=(np.sum(weight==0.0)+1)*1.0/len(weight.reshape(-1))

    return sparse_dict

import torch.distributed as dist

def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0

def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


import matplotlib.pyplot as plt
import os
def get_state_dict(path,invalid_keys=['conf','deeplab_head.seg.3']):
    new_params=OrderedDict()
    for key,value in torch.load(path).items():
        new_params[key[7:]]=value
    return new_params


def show_result(image,pred_det,pred_mask,args):
    COLORS = [(255, 255, 0), (0, 255, 0), (0, 0, 255)]
    crop_w,crop_h=image.shape[1],image.shape[0]
    mask_full=np.zeros(shape=(crop_w,crop_h),dtype='uint8')
    detections = pred_det.detach().cpu().numpy()
    if pred_mask is not None:
        pred_mask=np.argmax(pred_mask[0].detach().cpu().numpy(),axis=0)

    for i in range(detections.shape[1]):
        j = 0
        while detections[0, i, j, 0] >= args.vis_conf:
            pt = detections[0, i, j, 1:] * np.array([crop_w,crop_h,crop_w,crop_h])
            cv2.rectangle(image,(int(pt[0]), int(pt[1])),(int(pt[2]), int(pt[3])), COLORS[i % 3], 1)
            cv2.putText(image, "{:.2f}".format(detections[0, i, j, 0]*100 ,), (int(pt[0]), int(pt[1])),cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS[i % 3], 1, cv2.LINE_AA)
            j += 1
    if pred_mask is not None:
        image[pred_mask==2]=(255,0,255)
        image[pred_mask==3]=(0,255,255)


def show(model,dataset,args=None):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = 20
    out = cv2.VideoWriter(args.save_video, fourcc, fps, (args.input_shape[2],args.input_shape[1]))

    LABEL_MAP = ["car", 'bus-truck', 'person']


    model=model.cuda()
    model.eval()

    with torch.no_grad():
        cv2.namedWindow('frame',0)

        if os.path.isdir(args.image_seq):
            def get_input():
                image_files=os.listdir(args.image_seq)
                for img_id in image_files:
                    img_path=os.path.join(args.image_seq,img_id)
                    image=cv2.resize(cv2.imread(img_path,0),(args.input_shape[2],args.input_shape[1]))[:,:,np.newaxis]
                    image=torch.from_numpy(image*1.0-128.0).permute(2, 0, 1).float()
                    yield image
        else:
            def get_input():
                for item in dataset:
                    image,objects,img_path,img_info=item
                    yield image

        for image in get_input():
            net_input=image.unsqueeze(0).cuda()
            image=(image.permute(1, 2, 0).numpy()+128).astype('uint8').copy()
            pred_det,pred_mask,_=model(net_input)

            show_result(image,pred_det,pred_mask,args)
            out.write(image)  # 写入视频对象
            cv2.imshow('frame',image)
            if cv2.waitKey(0)&0xff==ord('q'):
                out.release()
                break
        cv2.destroyAllWindows()
        # cap.release()
        out.release()


best_mAp=0.0
if __name__ == "__main__":
    args = parse_args()


    if args.dist:
        torch.backends.cudnn.benchmark=True
        import torch.distributed as dist
        args.gpu = args.local_rank
        dist.init_process_group(backend='nccl',init_method="env://")
        torch.cuda.set_device(args.local_rank)
        args.world_size = torch.distributed.get_world_size()
        glog.info("World Size is :{}".format(args.world_size))

    if args.backbone=='VGG16':
        from caffe_models.vgg2.phone_128_vgg_float_3 import phone_128_vgg_float
        model=phone_128_vgg_float()
        model_binary = phone_128_vgg_float()
        
        import torch.nn.init as init
        def weights_init(m):
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            # elif isinstance(m,nn.BatchNorm2d):
            #     m.weight.data.fill_(1)
            #     m.bias.data.zero_()
        model.apply(weights_init)

    # snapshot_path=os.path.join('/home/disk/yenanfei/DMS_phone/ssd_pytorch/caffe_models/vgg2/',args.name+'.pth')
    # snapshot_path=os.path.join('/home/disk/yenanfei/DMS_phone/phone_model_pytorch/snapshot/recult_dataset/uploaded_version/vgg_float/phone_128_vgg_float_ssd_best.pth')
    snapshot_path=os.path.join('/home/disk/yenanfei/OMS_phone/weights/float/OMS_phone_128_vgg_float_best.pth')


    if args.restore_snapshot is not None and os.path.isfile(args.restore_snapshot):
        snapshot_path=args.restore_snapshot

    if os.path.isfile(snapshot_path):
        glog.info("restore {}".format(snapshot_path))
        state_dict=torch.load(snapshot_path,map_location=lambda storage, loc: storage)

        # model.load_state_dict(state_dict,strict=False)
        # model_binary.load_state_dict(state_dict,strict=False)


    vista_dataset_val=VOCFormatDetectionDataset(
                                            root='/home/disk/qizhongpei/DMS_phone/PhoneDataset_recut',
                                            # root='/home/disk/yenanfei/DMS_phone/daiyule/VOC2021',

                                            file_list='/home/disk/qizhongpei/DMS_phone/PhoneDataset_recut/ImageSets/test_modified.txt',
                                            # file_list='/home/disk/yenanfei/DMS_phone/daiyule/VOC2021/ImageSets/test.txt',

                                            img_size=args.input_shape[1],
                                            batch_size=args.train_batch_size,
                                            augment=False,
                                            hyp=None,
                                            rect=False,
                                            image_weights=False,
                                            cache_images=False,
                                            single_cls=False,
                                            stride=32,
                                            pad=0,
                                            rank=-1
                                            )
    if args.train_aug:
        # train_transform = SSDAugmentation(args.input_shape[1], args.input_mean,base_size=726, ssd_sampling=True, deeplab_sampling=False)
        glog.info("training augmentation")
    else:
        train_transform=None

    with open(args.aug_config) as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)  # load hyps

    vista_dataset_train=VOCFormatDetectionDataset(
                                            root='/home/disk/qizhongpei/DMS_phone/PhoneDataset_recut',
                                            # root='/home/disk/yenanfei/DMS_phone/daiyule/VOC2021',

                                            file_list='/home/disk/qizhongpei/DMS_phone/PhoneDataset_recut/ImageSets/trainval_modified.txt',
                                            # file_list='/home/disk/yenanfei/DMS_phone/daiyule/VOC2021/ImageSets/trainval.txt',

                                            img_size=args.input_shape[1],
                                            batch_size=args.train_batch_size,
                                            augment=True,
                                            hyp=hyp,
                                            rect=False,
                                            image_weights=False,
                                            cache_images=False,
                                            single_cls=False,
                                            stride=32,
                                            pad=0,
                                            rank=-1
                                            )

    detection_collate=VOCFormatDetectionDataset.collate_fn


    if args.sync_bn:
        from sync_batchnorm import convert_model
        if args.dist:
            model=convert_model(model).cuda(args.local_rank)
        else:
            model=convert_model(model).cuda()
    else:
        if args.dist:
            model=model.cuda(args.local_rank)
        else:
            model=model.cuda()



    summary(model, tuple(args.input_shape))



    if args.dist:

        model=nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
        train_sampler = torch.utils.data.distributed.DistributedSampler(vista_dataset_train)
        vista_data_loader_train = torch.utils.data.DataLoader(vista_dataset_train, args.train_batch_size//args.world_size,
                                                        num_workers=6,
                                                        collate_fn=detection_collate,
                                                        pin_memory=True, drop_last=True,sampler=train_sampler)

    else:
        model = nn.DataParallel(model.cuda(),device_ids=args.devices)
        model_binary = nn.DataParallel(model_binary.cuda(),device_ids=args.devices)

        vista_data_loader_train = torch.utils.data.DataLoader(vista_dataset_train, args.train_batch_size,
                                                        num_workers=6,
                                                        shuffle=True, collate_fn=detection_collate,
                                                        pin_memory=True, drop_last=True)

        vista_data_loader_val = torch.utils.data.DataLoader(vista_dataset_val, args.train_batch_size,
                                                        num_workers=6,
                                                        shuffle=True, collate_fn=detection_collate,
                                                        pin_memory=True, drop_last=True)



    if args.optimizer=="adam":
        optimizer = optim.Adam(model_binary.parameters(),lr=args.learning_rate, betas=(0.9, 0.999),weight_decay=args.weight_decay)
        # optimizer = optim.Adam(model.parameters(),lr=args.learning_rate, betas=(0.9, 0.999),weight_decay=args.weight_decay)
        
        scheduler =optim.lr_scheduler.StepLR(optimizer,step_size=40,gamma=0.1)
    multibox_loss = MultiBoxLoss(2, 0.5, True, 0, True,3, 0.4, False, True,cross_dataset=False,use_atss=args.use_atss,no_conflict_label=4)



    def seg_det_csp_criterion(pred, y_gt):
        objects,img_path,img_info= y_gt
        # glog.info("{} {}".format(pred[0].size(0),objects))
        with torch.no_grad():
            select_indices = list(filter(lambda i: objects[i] is not None, range(len(objects))))
            object_targets = [object.cuda(non_blocking=True) for object in objects if object is not None]
            # conflict_flag = [flag for flag in conflict_flag if flag is not None]

        loc_data,conf_data,priors=pred[0],pred[1],pred[2]
        loc_data = loc_data[select_indices]
        conf_data = conf_data[select_indices]
        pos_index=[]
        neg_index=[]
        object_targets_new=[]
        for i in range(len(object_targets)):
            s=object_targets[i][0,4]
            if s!=-1:
                pos_index.append(i)
                object_targets_new.append(object_targets[i])
            else:
                neg_index.append(i)

        loss_l, loss_c = multibox_loss( [loc_data[pos_index].float(), conf_data[pos_index].float(), priors.float()], object_targets_new,None)
        # loss_l_G = 3*loss_l_G
        ## 添家负样本训练
        negative_num,anchors,cls_num=conf_data[neg_index].shape

        torch.set_printoptions(profile="full")
        all_negative=conf_data[neg_index].view(-1,2)
        if all_negative.shape[0]>0:
            negative_pred=all_negative.argmax(1)
            
            hard_negative=all_negative[negative_pred==1]

            if hard_negative.shape[0]>0:
                loss = F.cross_entropy(hard_negative, torch.zeros(hard_negative.shape[0]).long().cuda())
                # glog.info('loss{}'.format(loss))
                # glog.info('loss_c{}'.format(loss_c))

                loss_c=loss_c+loss
                

        return { "ssd_loc": loss_l, "ssd_conf": loss_c}

    def test_net_criterion(model):
        mIoU, mAp=test_net('result',model,True,vista_dataset_val,["phone","cell phone"], iou_thresh=args.iou_thresh)
        return {'mAp': mAp, 'mIoU': mIoU}


    def change_layer(model):

        model.module.conv4_2=nn.Conv2d(512, 640, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), groups=1, dilation=1, bias=True)
        model.module.bn_conv4_2=nn.BatchNorm2d(num_features=640)
        model.module.relu4_2=nn.ReLU(inplace=True)
        model.module.pool4=nn.MaxPool2d(kernel_size=2, stride=2)
        model.module.conv5_1=nn.Conv2d(640, 512, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), groups=1, dilation=1, bias=True)
        model.module.bn_conv5_1=nn.BatchNorm2d(num_features=512)
        model.module.relu5_1=nn.ReLU(inplace=True)
        model.module.conv4_3_norm_mbox_loc=nn.Conv2d(640, 24, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), groups=1, dilation=1, bias=True)
        model.module.conv4_3_norm_mbox_loc_perm=custom_nn.Permute(order=[0, 2, 3, 1])
        model.module.conv4_3_norm_mbox_loc_flat=custom_nn.Flatten()
        model.module.conv4_3_norm_mbox_conf=nn.Conv2d(640, 12, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), groups=1, dilation=1, bias=True)
        model.module.conv4_3_norm_mbox_conf_perm=custom_nn.Permute(order=[0, 2, 3, 1])
        model.module.conv4_3_norm_mbox_conf_flat=custom_nn.Flatten()


        model.module.conv4_2.apply(weights_init)
        model.module.conv5_1.apply(weights_init)
        model.module.conv4_3_norm_mbox_loc.apply(weights_init)
        model.module.conv4_3_norm_mbox_conf.apply(weights_init)

    def add_lowrank_layer(layers):
        keys=layers._modules.keys()

        for i,key in enumerate(keys):
            conv_layer=layers._modules[key]
            key_v = str(key+'_v')
            conv_v =conv_layer._modules[key_v]
            print(conv_v)
            exit(0)



    if args.lowrank:

        torch_lowrank_layers(model.cpu(),percentale=0.9,has_bn=False)
        torch_lowrank_layers(model_binary.cpu(),percentale=0.9,has_bn=False)

        lowrank_snapshot_path = '/home/disk/yenanfei/DMS_phone/phone_model_pytorch/snapshot/recult_dataset/uploaded_version/lowrank/phone_128_vgg_float_lowrank_newest9.pth'
        if os.path.isfile(lowrank_snapshot_path):
            model.module.load_state_dict(torch.load(lowrank_snapshot_path),strict=True)
            model_binary.module.load_state_dict(torch.load(lowrank_snapshot_path),strict=True)
            glog.info("======>>> restore snapshot {}".format(lowrank_snapshot_path))
   

        model = model.cuda()
        model_binary=model_binary.cuda()
        print(model)
        
        
        


    
    if args.binary:
        binary_snapshot_path = '/home/disk/yenanfei/DMS_phone/phone_model_pytorch/snapshot/recult_dataset/binary/phone_128_vgg_float_binary_newest26_0.882.pth'

        exclude_layer=['fc7_mbox_loc','fc7_mbox_conf',\
                'conv6_2_mbox_loc','conv6_2_mbox_conf','conv7_2_mbox_loc','conv7_2_mbox_conf','conv8_2_mbox_loc','conv8_2_mbox_conf']
        torch_binarize_layers(model_binary.cpu())

        freeze_layer=['conv1_1','conv1_2','conv2_1','conv2_2','conv3_1','conv3_2','conv4_1','conv4_2','conv5_1','conv5_2','conv5_3',\
            'fc7','conv6_1','conv6_2', 'conv7_1','conv7_2','conv8_1','conv8_2']
        # freeze(model.cpu(),freeze_layer=freeze_layer)
        
        if os.path.isfile(binary_snapshot_path):
            model_binary.module.load_state_dict(torch.load(binary_snapshot_path),strict=True)
            glog.info("==>>>loading binary snapshot from :{}".format(binary_snapshot_path))
        
        
        

        model_binary.cuda()
        print(model_binary)

    if args.show:
        show(model,vista_dataset_val,args=args)
        exit(0)

    layer_pairs=[]
    if args.distillation:

        teacher_layers=OrderedDict()
        student_layers=OrderedDict()
        
        from procedure.distill import build_parents_graph
        build_parents_graph(model.module,teacher_layers)
        build_parents_graph(model_binary.module,student_layers)
        
        for layer_name in student_layers:
            student_layer=student_layers[layer_name]

            exclude = ['conv1_1','conv1_2','conv2_1','conv2_2','conv3_1','conv3_2','conv4_1','conv5_1','conv5_2','conv5_3',\
            'fc7','conv6_1','conv6_2','conv7_1','conv7_2','conv8_1','conv8_2','fc7_mbox_loc','fc7_mbox_conf','conv6_2_mbox_loc',\
            'conv6_2_mbox_conf','conv7_2_mbox_loc','conv7_2_mbox_conf','conv8_2_mbox_loc','conv8_2_mbox_conf']
    
            if layer_name in exclude:
                    continue

            if 'Binary' in student_layer.__class__.__name__ :#

                teacher_layer=teacher_layers[layer_name]

                def get_intermediate_output(module,input,output):
                    module.output=output
                student_layer.register_forward_hook(get_intermediate_output)
                teacher_layer.register_forward_hook(get_intermediate_output)
                layer_pairs.append((teacher_layer,student_layer))

        model =model.cuda()
        model_binary = model_binary.cuda()
        

    if args.prune:
        pruning_snapshot_path = '/home/disk/yenanfei/OMS_phone/weights/purning/OMS_phone_128_vgg_purne_0.96_best_1.pth'
        sys.path.append("/home/disk/yenanfei/DMS_phone/")

        state_dict = torch.load(pruning_snapshot_path)
        print(state_dict)
        for name in state_dict.keys():
            if 'weight' in name:
                
                mask = name.replace('weight','mask')
                if mask in state_dict.keys():
                    state_dict[name]=state_dict[name]*state_dict[mask]
        
        model.module.load_state_dict(state_dict,strict=False)
        glog.info("==>>>loading purning snapshot from :{}".format(pruning_snapshot_path))    
        from pytorch_prune.pruning import create_sparse_table,pruning
        input_shape=(1,1,128,128)
        model.eval()
        sparse_steps = [0.96]
        sparse_dict_list = {k:v for k,v in zip(sparse_steps,create_sparse_table(model,input_shape,DEFAULT_SPARSE_STEP=sparse_steps))}
        # model.module.load_state_dict(torch.load(pruning_snapshot_path),strict=False)
        # glog.info("==>>>loading purning snapshot from :{}".format(pruning_snapshot_path))

        total_sparse,sparse_dict = sparse_dict_list[0.96]
        glog.info("==>>>curr total sparse:{}".format(total_sparse))
        pruning(model,sparse_dict)
        
        model = model.cuda()
       

    if args.fp16:
        model = model.half()

    train_val = TrainVal3(model,
                        model_binary,
                        supervised_list=[(vista_data_loader_train, seg_det_csp_criterion)],
                        supervised_list_val=[(vista_data_loader_val, seg_det_csp_criterion)],
                        model_eval_list=[test_net_criterion],
                        data_eval_list=[(vista_data_loader_val,seg_det_csp_criterion)],
                        key_metric_name="mAp",
                        is_acc=True,
                        test_initialization=args.eval_first,
                        num_epoches=args.epochs,
                        display=args.display,
                        model_name=args.name,
                        init_lr=args.learning_rate,
                        iter_size=args.iter_size,
                        lr_policy=lambda epoch: (0.1 ** (epoch // 20)),
                        weight_decay=args.weight_decay,#
                        reporter=None,
                        solver_type=args.optimizer,  # 注意Adam
                        betas=(0.95, 0.999),
                        degenerated_to_sgd=True,
                        momentum=0.843,
                        always_snapshot=True,
                        lowrank = args.lowrank,
                        binary = args.binary,
                        distillation = args.distillation,
                        layer_pairs = layer_pairs,
                        lowrank_snapshot_path = './snapshot/phone_128/phone_128_lowrank_continue.pth',
                        binary_snapshot_path='./snapshot/phone_128/phone_128_lowrank_binary_best_5.pth',
                        cosin=False,
                        fp16=args.fp16,
                        )
    train_val.start()