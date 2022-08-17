import caffe
import numpy as np
caffe_phone=caffe.Net('./phone_128_vgg_float_binary_0825.prototxt','./phone_128_vgg_float_binary_0825.caffemodel',caffe.TRAIN)

for param_name in caffe_phone.params.keys():
    weight = caffe_phone.params[param_name][0].data
    bias = caffe_phone.params[param_name][1].data

    # print(weight)
    # weight = np.array(weight)
    print(param_name,'+++++++++++++++++')

    weight[np.where(np.abs(weight)<=1e-15)] = 0
    bias[np.where(np.abs(bias)<=1e-15)] = 0

    caffe_phone.params[param_name][0].data[:] = weight
    caffe_phone.params[param_name][1].data[:] = bias
    # caffe_phone.layers(param_name).params(1).set_data(weight)
    # caffe_phone.layers(param_name).params(1).set_data(bias)

    
res  = caffe_phone.forward()
caffe_phone.save('./phone_128_vgg_float_binary_0825_2zero.caffemodel')

    # exit(0)