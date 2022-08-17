import caffe
import numpy as np
from utils.general import xyxy2xywh, xywh2xyxy
net=caffe.Net('/home/disk/yenanfei/DMS_phone/PhoneDataset_recut/deploy.prototxt','/home/disk/yenanfei/DMS_phone/PhoneDataset_recut/weight/v1_iter_7500.caffemodel',caffe.TEST)
output=net.forward(['mbox_priorbox'])
xyxy_anchors=output['mbox_priorbox'][0][0].reshape(-1,4)
xywh_anchors=xyxy2xywh(xyxy_anchors)
print(xywh_anchors.shape)
np.save('caffe_models/vgg2/xywh_anchors{}x{}.npy'.format(xywh_anchors.shape[0],xywh_anchors.shape[1]),xywh_anchors)