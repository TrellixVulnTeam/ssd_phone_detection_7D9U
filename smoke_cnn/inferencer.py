import cv2
import os
import numpy as np
class TRTInferencer():
    def __init__(self,model_file,output_names,height,width,channels,mean=128.,scale=128.):
        import pytrt
        self.trt = pytrt.Trt()
        self.trt.LoadEngine(model_file)
        self.mean=mean
        self.scale=scale
        self.channels=channels
        self.width=width
        self.height=height
        self.output_names=output_names
        assert self.channels in [1,3]
        
    def infer(self,image):
        image=cv2.resize(image,(self.width,self.height))
        if self.channels==1 and image.ndim==3 and image.shape[-1]==3:
            image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        if image.ndim==2:
            image=image[:,:,np.newaxis]
        net_input=np.transpose(image,(2,0,1))
        self.trt.DoInference((net_input-self.mean)/self.scale)
        outputs=tuple(map(lambda name:self.trt.GetOutput(name),self.output_names))
        return outputs
        
class caffeInferencer():
    def __init__(self,model_file,weight_file,output_names,height,width,channels,mean=128.,scale=128.):
        import caffe
        caffe.set_mode_gpu()
        self.mean=mean
        self.scale=scale
        self.channels=channels
        self.width=width
        self.height=height
        self.output_names=output_names
        self.net=caffe.Net(model_file,weight_file,caffe.TEST)
        assert self.channels in [1,3]
        
    def infer(self,image):
        image=cv2.resize(image,(self.width,self.height))
        if self.channels==1 and image.ndim==3 and image.shape[-1]==3:
            image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        if image.ndim==2:
            image=image[:,:,np.newaxis]
        net_input=np.transpose(image,(2,0,1))
        self.net.blobs['data'].data[:]=((net_input-self.mean)/self.scale)[:]
        output_dict=self.net.forward()
        outputs=tuple(map(lambda name:output_dict[name][0],self.output_names))
        return outputs


class ONNXInferencer():
    def __init__(self,model_file,output_names,height,width,channels,mean=128.,scale=128.):
        import onnxruntime
        self.mean=mean
        self.scale=scale
        self.channels=channels
        self.width=width
        self.height=height
        self.output_names=output_names
        self.ort_session=onnxruntime.InferenceSession(model_file)
        assert self.channels in [1,3]
        
    def infer(self,image):
        image=cv2.resize(image,(self.width,self.height))
        if self.channels==1 and image.ndim==3 and image.shape[-1]==3:
            image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        if image.ndim==2:
            image=image[:,:,np.newaxis]
        net_input=np.transpose(image,(2,0,1))
        ort_inputs = {self.ort_session.get_inputs()[0].name: (net_input-self.mean)/self.scale}
        outputs = self.ort_session.run(None, ort_inputs)
        return outputs

try: 
    import torch
except Exception as e:
    pass
class PytorchInferencer():
    
    def __init__(self,model_file,output_names,height,width,channels,mean=128.,scale=128.):
        
        self.model = torch.load(model_file)['model'].cuda()  # load FP32 model
        self.mean=mean
        self.scale=scale
        self.channels=channels
        self.width=width
        self.height=height
        self.output_names=output_names
        print(self.model)
        assert self.channels in [1,3]
        
    def infer(self,image):
        image=cv2.resize(image,(self.width,self.height))
        if self.channels==1 and image.ndim==3 and image.shape[-1]==3:
            image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        if image.ndim==2:
            image=image[:,:,np.newaxis]
        with torch.no_grad():
            net_input=(torch.from_numpy(np.transpose(image,(2,0,1))).unsqueeze(0).cuda()-self.mean)/self.scale
            outputs =self.model(net_input)
            if type(outputs) is tuple or type(outputs) is list:
                return tuple(map(lambda x:x.cpu().numpy(),outputs))
            else:
                return (outputs.cpu().numpy())
         
class Inferencer:
    def __init__(self,config_dict):
        model_file=config_dict['model_file']
        output_names=config_dict['output_names'].split(',')
        width=config_dict['width']
        height=config_dict['height']
        channel=config_dict['channel']
        mean=config_dict['mean']
        scale=config_dict['scale']
        weight_file=config_dict['weight_file']
        python_path=config_dict['python_path']
        assert model_file is not None
        
        if python_path is not None:
            import sys
            sys.path.insert(0,python_path) 
        if model_file.endswith('.prototxt') and weight_file is not None and weight_file.endswith('.caffemodel'):

            self.inference=caffeInferencer(model_file,weight_file,output_names,height,width,channel,mean=mean,scale=scale)
        elif model_file.endswith('.engine'):
            self.inference=TRTInferencer(model_file,output_names,height,width,channel,mean=mean,scale=scale)
        elif model_file.endswith('.onnx'):
            self.inference=ONNXInferencer(model_file,output_names,height,width,channel,mean=mean,scale=scale)
        elif model_file.endswith('.pt') and python_path is not None and os.path.isdir(python_path):
            import sys
            sys.path.insert(0,python_path)
            self.inference=PytorchInferencer(model_file,output_names,height,width,channel,mean=mean,scale=scale)
        else:
            raise NotImplementedError
            
    def infer(self,image):
        return self.inference.infer(image)