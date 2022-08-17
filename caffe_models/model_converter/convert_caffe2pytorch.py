
import sys
import numpy as np
import os
import argparse

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='convert caffe model into pytorch model')
    parser.add_argument('-m','--model', dest='caffe_prototxt',default='caffe_model/train_prun_sparse9_head_p0.prototxt',
                        help='prototxt file',
                        type=str)
    parser.add_argument('-w','--weights', dest='caffe_model',
                        help='weights file',default='caffe_model/prun_adas_iterative_sparse9_iter_120000.caffemodel',
                        type=str)

    parser.add_argument('-o','--output_tops',dest='output_tops',nargs='+',
                        help='caffe output tops',default=["mbox_conf_flatten","mbox_loc"],
                        type=str)

    parser.add_argument('-i','--input_shape',dest='input_shape',nargs='+',
                        help='caffe input shape',default=[3,512,512],
                        type=int)

    parser.add_argument('-n','--pytorch_name',dest='pytorch_model_name',
                        help='pytorch model name',
                        default="SSD512",
                        type=str)

    parser.add_argument('-fd','--feature_dir',dest='feature_dir',
                        help='feature map storage directory path',
                        default="feature_dir",
                        type=str)



    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    prototxt_path=args.caffe_prototxt
    weights_path=args.caffe_model
    from converter import caffe2pytorch,compare_caffe_pytorch_model
    model=caffe2pytorch(prototxt_path,
                        weights_path,
                        pytorch_model_path='{}.py'.format(args.pytorch_model_name),outputs=args.output_tops
                        )
    model.eval()

    if  not os.path.exists(args.feature_dir):
        os.mkdir(args.feature_dir)
    input_numpy=np.random.rand(*args.input_shape)
    compare_caffe_pytorch_model(input_numpy,prototxt_path,weights_path,model,feature_dir=args.feature_dir)



