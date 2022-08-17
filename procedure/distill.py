import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from collections import OrderedDict
from functools import reduce
from custom_nn import BinaryConv2d


def build_parents_graph(layers,layer_dict):
    if "model" not in layer_dict:
        layer_dict["model"]=layers
        layers.layer_name="model"
    keys=layers._modules.keys()
    for i,key in enumerate(keys):
        layer=layers._modules[key]
        if hasattr(layers,'layer_name'):
            layer.parent_name=layers.layer_name
        else:
            layer.parent_name=""
        layer.layer_name=(layer.parent_name+'.'+key) if layer.parent_name!='' else ''+key
        layer_dict[layer.layer_name]=layer
        if hasattr(layer,'_modules'):
            build_parents_graph(layer,layer_dict)
import glog
class ModelDistill(nn.Module):
    def __init__(self,teacher,student,shared_model=None,shared_train=False):
        super(ModelDistill,self).__init__()
        self.teacher=teacher
        self.shared_train=shared_train
        self.student=student
        self.shared_model=shared_model
        self.layer_pairs=[]
        if self.teacher is not None:
            self.mse= nn.MSELoss(size_average=True)
            teacher_layers=OrderedDict()
            build_parents_graph(self.teacher,teacher_layers)
            student_layers=OrderedDict()
            build_parents_graph(self.student,student_layers)

            for layer_name in student_layers:
                student_layer=student_layers[layer_name]

                # if type(student_layer) is nn.BatchNorm2d:
                #     continue


                teacher_layer=teacher_layers[layer_name]

                if  'mbox_loc' in layer_name or 'mbox_conf' in layer_name:#type(student_layer) is nn.BatchNorm2d or

                    def get_intermediate_output(module,input,output):
                        module.output=output
                    student_layer.register_forward_hook(get_intermediate_output)
                    teacher_layer.register_forward_hook(get_intermediate_output)
                    self.layer_pairs.append((teacher_layer,student_layer))

            assert len(self.layer_pairs)>0,self.layer_pairs
    def forward(self,x):


        if self.teacher is not None:
            self.teacher.eval()


        # for module in self.student.modules():

        #     if isinstance(module, nn.BatchNorm2d):
        #         # if hasattr(module, 'weight'):
        #         #     module.weight.requires_grad_(False)
        #         # if hasattr(module, 'bias'):
        #         #     module.bias.requires_grad_(False)
        #         module.eval()

        if self.teacher is not None:
            with torch.no_grad():
                teacher_output=self.teacher(x)

        student_output=self.student(x)
        if len(self.layer_pairs)==0:
            loss=0.0
        else:
            loss=reduce(lambda x,y:x+y,map(lambda ts:self.mse(ts[0].output,ts[1].output),self.layer_pairs))


        return (student_output,loss)


if __name__=="__main__":
    from torchvision.models import resnet18
    teacher=resnet18(pretrained=True)
    student=resnet18(pretrained=True)
    torch_binarize_layers(student)
    print(teacher)
    print(student)
    model_distill=ModelDistill(teacher,student)
    output=model_distill(torch.rand(1,3,224,224))
    print(output)
