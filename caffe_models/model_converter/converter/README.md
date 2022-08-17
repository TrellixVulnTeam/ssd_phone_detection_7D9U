# Convert Pytorch Model into Caffe Model

## Usage

### step1 Genrate A Caffemodel Definition and Weights
```
from pytorch2caffe import create_model,create_model_weight
...
model=PytorchModel()
model.eval()#don't forget it
numpy_input=np.load("image_input.npy")#CxHxW
net_input=torch.from_numpy(numpy_input).unsqueeze(0).float()
create_model_weight(model,net_input,model_path="model.prototxt",weight_path="model.caffemodel",feature_dir="feature_dir")

```
### step2 Modify The Caffemodel Definition Generated

1.remove some useless layer from the prototxt file.

2.add some caffe layer in the prototxt file for operation in pytorch modelwhich is not as a nn.Module such as F.relu, tensor.permute and etc.

### step3 Compare the Feature Map of Pytorch Model and Caffe Model
```
from pytorch2caffe import compare_caffe_model
compare_caffe_model("model_modified.prototxt","model.caffemodel",numpy_input,"feature_dir")
```
**Tip:
Comparison only focus on some layers near to the final layers. 
After call caffe forward,you can only access the final featuremap of some inplace layers, 
it is quite different from pytorch in which you can access any layer outputs no matter inplace or not during forward,
so you can see the cosine similarity of the featuremaps of  conv2d+relu layer is small, please don't care.**

comparison result:
```
....
I0731 21:53:53.899582 28370 pytorch2caffe.py:185] relu_sim:1.0 sim:0.5403797626495361 l2_dist:0.0002801486989483237 (16384,) (16384,) down7_mpconv_1_conv_0_convx_conv
I0731 21:53:53.901759 28370 pytorch2caffe.py:185] relu_sim:0.9999998807907104 sim:0.9999998807907104 l2_dist:0.00016875340952537954 (4096,) (4096,) down8_mpconv_0
I0731 21:53:53.903952 28370 pytorch2caffe.py:185] relu_sim:1.0 sim:0.7736454010009766 l2_dist:9.413668885827065e-05 (4096,) (4096,) down8_mpconv_1_conv_0_convx_conv
I0731 21:53:53.905725 28370 pytorch2caffe.py:185] relu_sim:1.0 sim:1.0 l2_dist:6.489406951004639e-05 (1024,) (1024,) down9_mpconv_0
I0731 21:53:53.907405 28370 pytorch2caffe.py:185] relu_sim:1.0 sim:0.7368014454841614 l2_dist:2.914430842793081e-05 (2048,) (2048,) down9_mpconv_1_conv_0_convx_conv
I0731 21:53:53.908901 28370 pytorch2caffe.py:185] relu_sim:1.0 sim:1.0 l2_dist:2.111184221575968e-05 (512,) (512,) down10
I0731 21:53:53.910264 28370 pytorch2caffe.py:185] relu_sim:1.0 sim:1.0 l2_dist:1.9312450604047626e-05 (134,) (134,) lm67_reg_head_conv_0_convx_conv
I0731 21:53:53.911365 28370 pytorch2caffe.py:185] relu_sim:1.0 sim:1.0 l2_dist:8.637533710498246e-07 (3,) (3,) headpose_head_conv_0_convx_conv


```
### step4 Debugging According the Comparison Analysis

# Convert Caffe Model into Pytorch Model
## Usage
### step1 Genrate Pytorch Model and Weight
```
from pytorch2caffe import caffe2pytorch
...
prototxt_path='model.prototxt'
weights_path='model.caffemodel'
model=caffe2pytorch(prototxt_path,
                    weights_path,
                    pytorch_model_path='PytorchModel.py',
                    outputs=["caffe_output_top_name1","caffe_output_top_name2"]
                    )

```
### step2 Compare the Feature Map of Caffe Model and Pytorch Model
```
from pytorch2caffe import compare_caffe_pytorch_model
from PytorchModel import PytorchModel
...
model=PytorchModel()
model.load_state_dict(torch.load("PytorchModel.pth"))
model.eval()#don't forget it
numpy_input=np.load("image_input.npy")#CxHxW
compare_caffe_pytorch_model(numpy_input,prototxt_path,weights_path,model,feature_dir="feature_dir")

```

comparison result:
```
...
I0731 21:37:24.410944 16063 caffe2pytorch.py:262] relu_sim:1.0 sim:1.0000001192092896 l2_dist:6.988649693084881e-05 (4096,) (4096,) conv12_2_v
I0731 21:37:24.412403 16063 caffe2pytorch.py:262] relu_sim:1.0 sim:1.0 l2_dist:3.3708845876390114e-05 (1024,) (1024,) pool10
I0731 21:37:24.413871 16063 caffe2pytorch.py:262] relu_sim:1.0000001192092896 sim:0.9999999403953552 l2_dist:1.593397610122338e-05 (2048,) (2048,) conv12_3_v
I0731 21:37:24.415183 16063 caffe2pytorch.py:262] relu_sim:1.0 sim:1.0 l2_dist:3.5486809792928398e-06 (512,) (512,) pool11
I0731 21:37:24.416522 16063 caffe2pytorch.py:262] relu_sim:1.0 sim:0.9137374758720398 l2_dist:1.0452118885950767e-06 (512,) (512,) conv12_4
I0731 21:37:24.417885 16063 caffe2pytorch.py:262] relu_sim:1.0 sim:0.9237357378005981 l2_dist:5.969965855001647e-07 (512,) (512,) conv12_5
I0731 21:37:24.419638 16063 caffe2pytorch.py:262] relu_sim:1.0 sim:1.0 l2_dist:4.65406969851756e-07 (134,) (134,) conv13_1
I0731 21:37:24.421286 16063 caffe2pytorch.py:262] relu_sim:1.0 sim:1.0 l2_dist:8.429369557916289e-08 (3,) (3,) conv13_2

```
### step3 Debugging According the Comparison Analysis
