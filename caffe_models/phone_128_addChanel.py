import torch
import torch.nn as nn
import torch.nn.functional as F
import custom_nn as custom_nn
from layers import *
import os

def local_path(fname):
    return os.path.join(os.path.dirname(__file__),fname)


class phone_128_addChanel(nn.Module):
    def __init__(self):
        super(phone_128_addChanel, self).__init__()
        self.conv1_1=nn.Conv2d(1, 64, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), groups=1, dilation=1, bias=True)
        self.bn_conv1_1=nn.BatchNorm2d(num_features=64)
        self.relu1_1=nn.ReLU(inplace=True)
        self.conv1_2=nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), groups=1, dilation=1, bias=True)
        self.bn_conv1_2=nn.BatchNorm2d(num_features=64)
        self.relu1_2=nn.ReLU(inplace=True)
        self.pool1=nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2_1=nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), groups=1, dilation=1, bias=True)
        self.bn_conv2_1=nn.BatchNorm2d(num_features=128)
        self.relu2_1=nn.ReLU(inplace=True)
        self.conv2_2=nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), groups=1, dilation=1, bias=True)
        self.bn_conv2_2=nn.BatchNorm2d(num_features=128)
        self.relu2_2=nn.ReLU(inplace=True)
        self.pool2=nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3_1=nn.Conv2d(128, 256, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), groups=1, dilation=1, bias=True)
        self.bn_conv3_1=nn.BatchNorm2d(num_features=256)
        self.relu3_1=nn.ReLU(inplace=True)
        self.conv3_2=nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), groups=1, dilation=1, bias=True)
        self.bn_conv3_2=nn.BatchNorm2d(num_features=256)
        self.relu3_2=nn.ReLU(inplace=True)
        self.pool3=nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4_1=nn.Conv2d(256, 512, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), groups=1, dilation=1, bias=True)
        self.bn_conv4_1=nn.BatchNorm2d(num_features=512)
        self.relu4_1=nn.ReLU(inplace=True)
        self.conv4_2=nn.Conv2d(512, 512, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), groups=1, dilation=1, bias=True)
        self.bn_conv4_2=nn.BatchNorm2d(num_features=512)
        self.relu4_2=nn.ReLU(inplace=True)

        self.conv4_3=nn.Conv2d(512, 128, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), groups=1, dilation=1, bias=True)
        self.bn_conv4_3=nn.BatchNorm2d(num_features=128)
        self.relu4_3=nn.ReLU(inplace=True)
        self.pool4=nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5_1=nn.Conv2d(512, 512, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), groups=1, dilation=1, bias=True)
        self.bn_conv5_1=nn.BatchNorm2d(num_features=512)
        self.relu5_1=nn.ReLU(inplace=True)
        self.conv5_2=nn.Conv2d(512, 512, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), groups=1, dilation=1, bias=True)
        self.bn_conv5_2=nn.BatchNorm2d(num_features=512)
        self.relu5_2=nn.ReLU(inplace=True)
        self.conv5_3=nn.Conv2d(512, 512, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), groups=1, dilation=1, bias=True)
        self.bn_conv5_3=nn.BatchNorm2d(num_features=512)
        self.relu5_3=nn.ReLU(inplace=True)
        self.fc7=nn.Conv2d(512, 512, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), groups=1, dilation=1, bias=True)
        self.bn_fc7=nn.BatchNorm2d(num_features=512)
        self.relu7=nn.ReLU(inplace=True)

        self.fc7_2=nn.Conv2d(512, 128, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), groups=1, dilation=1, bias=True)
        self.bn_fc7_2=nn.BatchNorm2d(num_features=128)
        self.relu7_2=nn.ReLU(inplace=True)

        self.conv6_1=nn.Conv2d(512, 256, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), groups=1, dilation=1, bias=True)
        self.bn_conv6_1=nn.BatchNorm2d(num_features=256)
        self.conv6_1_relu=nn.ReLU(inplace=True)
        self.conv6_2=nn.Conv2d(256, 512, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2), groups=1, dilation=1, bias=True)
        self.bn_conv6_2=nn.BatchNorm2d(num_features=512)
        self.conv6_2_relu=nn.ReLU(inplace=True)

        self.conv6_3=nn.Conv2d(512, 128, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), groups=1, dilation=1, bias=True)
        self.bn_conv6_3=nn.BatchNorm2d(num_features=128)
        self.conv6_3_relu=nn.ReLU(inplace=True)

        self.conv7_1=nn.Conv2d(512, 128, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), groups=1, dilation=1, bias=True)
        self.bn_conv7_1=nn.BatchNorm2d(num_features=128)
        self.conv7_1_relu=nn.ReLU(inplace=True)
        self.conv7_2=nn.Conv2d(128, 256, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2), groups=1, dilation=1, bias=True)
        self.bn_conv7_2=nn.BatchNorm2d(num_features=256)
        self.conv7_2_relu=nn.ReLU(inplace=True)

        self.conv7_3=nn.Conv2d(256, 128, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), groups=1, dilation=1, bias=True)
        self.bn_conv7_3=nn.BatchNorm2d(num_features=128)
        self.conv7_3_relu=nn.ReLU(inplace=True)


        self.conv8_1=nn.Conv2d(256, 128, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), groups=1, dilation=1, bias=True)
        self.bn_conv8_1=nn.BatchNorm2d(num_features=128)
        self.conv8_1_relu=nn.ReLU(inplace=True)
        self.conv8_2=nn.Conv2d(128, 256, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2), groups=1, dilation=1, bias=True)
        self.bn_conv8_2=nn.BatchNorm2d(num_features=256)
        self.conv8_2_relu=nn.ReLU(inplace=True)

        self.conv8_3=nn.Conv2d(256, 128, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), groups=1, dilation=1, bias=True)
        self.bn_conv8_3=nn.BatchNorm2d(num_features=128)
        self.conv8_3_relu=nn.ReLU(inplace=True)


        self.conv4_3_norm_mbox_loc=nn.Conv2d(128, 24, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), groups=1, dilation=1, bias=True)
        self.conv4_3_norm_mbox_loc_perm=custom_nn.Permute(order=[0, 2, 3, 1])
        self.conv4_3_norm_mbox_loc_flat=custom_nn.Flatten()
        self.conv4_3_norm_mbox_conf=nn.Conv2d(128, 12, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), groups=1, dilation=1, bias=True)
        self.conv4_3_norm_mbox_conf_perm=custom_nn.Permute(order=[0, 2, 3, 1])
        self.conv4_3_norm_mbox_conf_flat=custom_nn.Flatten()
        self.fc7_mbox_loc=nn.Conv2d(128, 24, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), groups=1, dilation=1, bias=True)
        self.fc7_mbox_loc_perm=custom_nn.Permute(order=[0, 2, 3, 1])
        self.fc7_mbox_loc_flat=custom_nn.Flatten()
        self.fc7_mbox_conf=nn.Conv2d(128, 12, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), groups=1, dilation=1, bias=True)
        self.fc7_mbox_conf_perm=custom_nn.Permute(order=[0, 2, 3, 1])
        self.fc7_mbox_conf_flat=custom_nn.Flatten()
        self.conv6_2_mbox_loc=nn.Conv2d(128, 24, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), groups=1, dilation=1, bias=True)
        self.conv6_2_mbox_loc_perm=custom_nn.Permute(order=[0, 2, 3, 1])
        self.conv6_2_mbox_loc_flat=custom_nn.Flatten()
        self.conv6_2_mbox_conf=nn.Conv2d(128, 12, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), groups=1, dilation=1, bias=True)
        self.conv6_2_mbox_conf_perm=custom_nn.Permute(order=[0, 2, 3, 1])
        self.conv6_2_mbox_conf_flat=custom_nn.Flatten()
        self.conv7_2_mbox_loc=nn.Conv2d(128, 24, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), groups=1, dilation=1, bias=True)
        self.conv7_2_mbox_loc_perm=custom_nn.Permute(order=[0, 2, 3, 1])
        self.conv7_2_mbox_loc_flat=custom_nn.Flatten()
        self.conv7_2_mbox_conf=nn.Conv2d(128, 12, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), groups=1, dilation=1, bias=True)
        self.conv7_2_mbox_conf_perm=custom_nn.Permute(order=[0, 2, 3, 1])
        self.conv7_2_mbox_conf_flat=custom_nn.Flatten()
        self.conv8_2_mbox_loc=nn.Conv2d(128, 24, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), groups=1, dilation=1, bias=True)
        self.conv8_2_mbox_loc_perm=custom_nn.Permute(order=[0, 2, 3, 1])
        self.conv8_2_mbox_loc_flat=custom_nn.Flatten()
        self.conv8_2_mbox_conf=nn.Conv2d(128, 12, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), groups=1, dilation=1, bias=True)
        self.conv8_2_mbox_conf_perm=custom_nn.Permute(order=[0, 2, 3, 1])
        self.conv8_2_mbox_conf_flat=custom_nn.Flatten()
        self.mbox_loc=custom_nn.Concat(axis=1, n_inputs=5)
        self.mbox_conf=custom_nn.Concat(axis=1, n_inputs=5)
        self.softmax = nn.Softmax(dim=-1)
        self.num_classes=2
        # self.load_state_dict(torch.load(local_path('phone_128_float.pth')))
        self.detect = Detect(self.num_classes, 0, 200, 0.01, 0.45)
        with torch.no_grad():
            self.priors = custom_nn.PriorBox(local_path('phone_128_float_xywh_anchors2046x4.npy'))()
        
    def forward(self,data):
        conv1_1=self.conv1_1(data)
        conv1_1=self.bn_conv1_1(conv1_1)
        conv1_1=self.relu1_1(conv1_1)
        conv1_2=self.conv1_2(conv1_1)
        conv1_2=self.bn_conv1_2(conv1_2)
        conv1_2=self.relu1_2(conv1_2)
        pool1=self.pool1(conv1_2)
        conv2_1=self.conv2_1(pool1)
        conv2_1=self.bn_conv2_1(conv2_1)
        conv2_1=self.relu2_1(conv2_1)
        conv2_2=self.conv2_2(conv2_1)
        conv2_2=self.bn_conv2_2(conv2_2)
        conv2_2=self.relu2_2(conv2_2)
        pool2=self.pool2(conv2_2)
        conv3_1=self.conv3_1(pool2)
        conv3_1=self.bn_conv3_1(conv3_1)
        conv3_1=self.relu3_1(conv3_1)
        conv3_2=self.conv3_2(conv3_1)
        conv3_2=self.bn_conv3_2(conv3_2)
        conv3_2=self.relu3_2(conv3_2)
        pool3=self.pool3(conv3_2)
        conv4_1=self.conv4_1(pool3)
        conv4_1=self.bn_conv4_1(conv4_1)
        conv4_1=self.relu4_1(conv4_1)
        conv4_2=self.conv4_2(conv4_1)
        conv4_2=self.bn_conv4_2(conv4_2)
        conv4_2=self.relu4_2(conv4_2)
        pool4=self.pool4(conv4_2)

        conv4_3=self.conv4_3(conv4_2)
        conv4_3=self.bn_conv4_3(conv4_3)
        conv4_3=self.relu4_3(conv4_3)
               
        
        conv5_1=self.conv5_1(pool4)
        conv5_1=self.bn_conv5_1(conv5_1)
        conv5_1=self.relu5_1(conv5_1)
        conv5_2=self.conv5_2(conv5_1)
        conv5_2=self.bn_conv5_2(conv5_2)
        conv5_2=self.relu5_2(conv5_2)
        conv5_3=self.conv5_3(conv5_2)
        conv5_3=self.bn_conv5_3(conv5_3)
        conv5_3=self.relu5_3(conv5_3)
        fc7=self.fc7(conv5_3)
        fc7=self.bn_fc7(fc7)
        fc7=self.relu7(fc7)

        fc7_2=self.fc7_2(fc7)
        fc7_2=self.bn_fc7_2(fc7_2)
        fc7_2=self.relu7_2(fc7_2)      
        

        conv6_1=self.conv6_1(fc7)
        conv6_1=self.bn_conv6_1(conv6_1)
        conv6_1=self.conv6_1_relu(conv6_1)
        conv6_2=self.conv6_2(conv6_1)
        conv6_2=self.bn_conv6_2(conv6_2)
        conv6_2=self.conv6_2_relu(conv6_2)

        conv6_3=self.conv6_3(conv6_2)
        conv6_3=self.bn_conv6_3(conv6_3)
        conv6_3=self.conv6_3_relu(conv6_3)

        conv7_1=self.conv7_1(conv6_2)
        conv7_1=self.bn_conv7_1(conv7_1)
        conv7_1=self.conv7_1_relu(conv7_1)
        conv7_2=self.conv7_2(conv7_1)
        conv7_2=self.bn_conv7_2(conv7_2)
        conv7_2=self.conv7_2_relu(conv7_2)
    
        conv7_3=self.conv7_3(conv7_2)
        conv7_3=self.bn_conv7_3(conv7_3)
        conv7_3=self.conv7_3_relu(conv7_3)

        conv8_1=self.conv8_1(conv7_2)
        conv8_1=self.bn_conv8_1(conv8_1)
        conv8_1=self.conv8_1_relu(conv8_1)
        conv8_2=self.conv8_2(conv8_1)
        conv8_2=self.bn_conv8_2(conv8_2)
        conv8_2=self.conv8_2_relu(conv8_2)

        conv8_3=self.conv8_3(conv8_2)
        conv8_3=self.bn_conv8_3(conv8_3)
        conv8_3=self.conv8_3_relu(conv8_3)

        conv4_3_norm_mbox_loc=self.conv4_3_norm_mbox_loc(conv4_3)
        conv4_3_norm_mbox_loc_perm=self.conv4_3_norm_mbox_loc_perm(conv4_3_norm_mbox_loc)
        conv4_3_norm_mbox_loc_flat=self.conv4_3_norm_mbox_loc_flat(conv4_3_norm_mbox_loc_perm)
        conv4_3_norm_mbox_conf=self.conv4_3_norm_mbox_conf(conv4_3)
        conv4_3_norm_mbox_conf_perm=self.conv4_3_norm_mbox_conf_perm(conv4_3_norm_mbox_conf)
        conv4_3_norm_mbox_conf_flat=self.conv4_3_norm_mbox_conf_flat(conv4_3_norm_mbox_conf_perm)
        fc7_mbox_loc=self.fc7_mbox_loc(fc7_2)
        fc7_mbox_loc_perm=self.fc7_mbox_loc_perm(fc7_mbox_loc)
        fc7_mbox_loc_flat=self.fc7_mbox_loc_flat(fc7_mbox_loc_perm)
        fc7_mbox_conf=self.fc7_mbox_conf(fc7_2)
        fc7_mbox_conf_perm=self.fc7_mbox_conf_perm(fc7_mbox_conf)
        fc7_mbox_conf_flat=self.fc7_mbox_conf_flat(fc7_mbox_conf_perm)
        conv6_2_mbox_loc=self.conv6_2_mbox_loc(conv6_3)
        conv6_2_mbox_loc_perm=self.conv6_2_mbox_loc_perm(conv6_2_mbox_loc)
        conv6_2_mbox_loc_flat=self.conv6_2_mbox_loc_flat(conv6_2_mbox_loc_perm)
        conv6_2_mbox_conf=self.conv6_2_mbox_conf(conv6_3)
        conv6_2_mbox_conf_perm=self.conv6_2_mbox_conf_perm(conv6_2_mbox_conf)
        conv6_2_mbox_conf_flat=self.conv6_2_mbox_conf_flat(conv6_2_mbox_conf_perm)
        conv7_2_mbox_loc=self.conv7_2_mbox_loc(conv7_3)
        conv7_2_mbox_loc_perm=self.conv7_2_mbox_loc_perm(conv7_2_mbox_loc)
        conv7_2_mbox_loc_flat=self.conv7_2_mbox_loc_flat(conv7_2_mbox_loc_perm)
        conv7_2_mbox_conf=self.conv7_2_mbox_conf(conv7_3)
        conv7_2_mbox_conf_perm=self.conv7_2_mbox_conf_perm(conv7_2_mbox_conf)
        conv7_2_mbox_conf_flat=self.conv7_2_mbox_conf_flat(conv7_2_mbox_conf_perm)
        conv8_2_mbox_loc=self.conv8_2_mbox_loc(conv8_3)
        conv8_2_mbox_loc_perm=self.conv8_2_mbox_loc_perm(conv8_2_mbox_loc)
        conv8_2_mbox_loc_flat=self.conv8_2_mbox_loc_flat(conv8_2_mbox_loc_perm)
        conv8_2_mbox_conf=self.conv8_2_mbox_conf(conv8_3)
        conv8_2_mbox_conf_perm=self.conv8_2_mbox_conf_perm(conv8_2_mbox_conf)
        conv8_2_mbox_conf_flat=self.conv8_2_mbox_conf_flat(conv8_2_mbox_conf_perm)
        mbox_loc=self.mbox_loc(conv4_3_norm_mbox_loc_flat,fc7_mbox_loc_flat,conv6_2_mbox_loc_flat,conv7_2_mbox_loc_flat,conv8_2_mbox_loc_flat)
        mbox_conf=self.mbox_conf(conv4_3_norm_mbox_conf_flat,fc7_mbox_conf_flat,conv6_2_mbox_conf_flat,conv7_2_mbox_conf_flat,conv8_2_mbox_conf_flat)
        if self.training:
            output = (
                mbox_loc.view(mbox_loc.size(0), -1, 4),
                mbox_conf.view(mbox_conf.size(0), -1, self.num_classes),
                self.priors.cuda()
                ,None)#,[conv4_3_norm_mbox_loc_flat,fc7_mbox_loc_flat,conv6_2_mbox_loc_flat,conv7_2_mbox_loc_flat,conv8_2_mbox_loc_flat,conv4_3_norm_mbox_conf_flat,fc7_mbox_conf_flat,conv6_2_mbox_conf_flat,conv7_2_mbox_conf_flat,conv8_2_mbox_conf_flat]
        else:
            output = (self.detect(
                    mbox_loc.view(mbox_loc.size(0), -1, 4),                   # loc preds
                    self.softmax(mbox_conf.view(mbox_conf.size(0), -1,
                    self.num_classes)),                # conf preds
                    self.priors.cuda(),
                    ),None)

        return output

# import torch.nn.init as init
# def weights_init(m):
#     if isinstance(m, nn.Conv2d):
#         init.xavier_uniform(m.weight.data)
#         if m.bias is not None:
#             m.bias.data.zero_()

# if __name__=="__main__":

#     from torchsummary import summary
#     model=phone_128_float().cuda()
#     model.eval()
#     model.apply(weights_init)
#     summary(model,(1,128,128),device='cuda')
#     from torchvision.models import vgg16_bn
#     features=vgg16_bn(pretrained=True).features.cuda()
#     print(features)
#     print(model)
#     state_dict=model.state_dict()
#     vgg_w_list=list(features.state_dict().items())
#     i=0
#     for key,item in state_dict.items():
#         try:
#             vgg_key,vgg_param=vgg_w_list[i]
#             item.copy_(vgg_param)
#             print(key,vgg_key,item.shape,vgg_param.shape)

#         except Exception as e:
#             print("fail {}".format(e))
#         i=i+1
#     #print(state_dict['conv1_1.weight'])
#     torch.save(state_dict,'caffe_models/phone_128_float.pth')