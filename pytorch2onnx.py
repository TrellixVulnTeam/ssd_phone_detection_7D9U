import io
import torch
import torch.onnx
from caffe_models.vgg2.phone_128_vgg_float import phone_128_vgg_float
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test():
    model = phone_128_vgg_float()

    pruning_snapshot_path = '/home/disk/yenanfei/DMS_phone/phone_model_pytorch/snapshot/recult_dataset/purning/phone_128_vgg_float_purning_0.95_newest5.pth'
    state_dict = torch.load(pruning_snapshot_path)
    for name in state_dict.keys():
        if 'weight' in name:
            mask = name.replace('weight','mask')
            if mask in state_dict.keys():
                state_dict[name]=state_dict[name]*state_dict[mask]
    model.load_state_dict(state_dict,strict=False)

    model = model.cuda()
    dummy_input1 = torch.randn(1, 1, 128, 128)
    input_names = [ "actual_input_1"]
    output_names = [ "output1" ]
    # torch.onnx.export(model, (dummy_input1, dummy_input2, dummy_input3), "C3AE.onnx", verbose=True, input_names=input_names, output_names=output_names)
    torch.onnx.export(model, dummy_input1, "phone_128_vgg_puring_95%.onnx", verbose=True, input_names=input_names, output_names=output_names)

if __name__ == "__main__":
    test()