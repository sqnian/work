import torch 
import time 
import os
# import sys

# CUR_PATH = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.dirname(os.path.dirname(CUR_PATH)))

from pytorch_quantization import tensor_quant
from pytorch_quantization.nn import QuantLinear, TensorQuantizer,QuantConv1d


os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = "cuda:0"

"""
test QuantConv1d and how to use it

"""

def get_Conv1d():
    conv1d = torch.nn.Conv1d(16,33,3,2)
    weight = conv1d.weight.data.clone()
    bias = conv1d.bias.data.clone()

    return conv1d,weight,bias

def conv1d_Imp(inputs,conv1d):
    outputs = conv1d(inputs)

    return outputs

def test():

    inputs =torch.randn([20, 16, 50],dtype=torch.float32)

    conv1d,weight,bias = get_Conv1d()

    outputs = conv1d_Imp(inputs,conv1d)
    # print(f"weight : {weight}\n")
    # print(f"bias : {bias}\n")
    print(f"outputs flatten()[:50]: {outputs.flatten()[:50]}\n")


    print("test quant conv1d \n")

    inputs_quantizer = TensorQuantizer(QuantConv1d.default_quant_desc_input)
    weight_quantizer = TensorQuantizer(QuantConv1d.default_quant_desc_weight)
    bias_quantizer = TensorQuantizer(QuantConv1d.default_quant_desc_input)

    # print(QuantConv1d.default_quant_desc_input)
    # print(QuantConv1d.default_quant_desc_weight)
    # print(QuantConv1d.default_quant_desc_input)
    # print(inputs_quantizer)
    # exit(0)


    q_inputs = inputs_quantizer(inputs)
    q_weight = weight_quantizer(weight)

    q_bias = bias_quantizer(bias)

    # print(f"q_inputs:{q_inputs}\n")
    # print(f"q_weight:{q_weight}\n")


    q_outputs = torch.nn.functional.conv1d(q_inputs,q_weight,q_bias,2)
    print(f"q_outputs flatten()[:50]:{q_outputs.flatten()[:50]}\n")


    diff = outputs.flatten() - q_outputs.flatten()
    print(f"diff max:{diff.max()}\n")


    print("time analyse\n")
    iter_num = 100
    torch.cuda.synchronize()
    start_time = time.time()
    for i in range(iter_num):
        outputs_ =  conv1d_Imp(inputs,conv1d)
        # q_outputs_ = torch.nn.functional.linear(q_inputs,q_weight)
    torch.cuda.synchronize()
    pt_time = time.time() - start_time

    torch.cuda.synchronize()
    start_time = time.time()
    for i in range(iter_num):
        # outputs_ = linear(inputs)
        q_outputs_ = torch.nn.functional.conv1d(q_inputs,q_weight,bias,2).half()
    torch.cuda.synchronize()
    q_pt_time = time.time() - start_time

    print("pt_time:{}, q_pt_time:{}, pt_time/q_pt_time:{}\n".format(pt_time,q_pt_time,pt_time/q_pt_time))



if __name__ == "__main__":
    torch.manual_seed(12345)
    torch.cuda.manual_seed_all(123)

    test()



"""
GPU : 
  fp16: pt_time:0.004925966262817383, q_pt_time:0.004044055938720703, pt_time/q_pt_time:1.2180756986204457
  fp32: pt_time:0.005042076110839844, q_pt_time:0.0037260055541992188, pt_time/q_pt_time:1.353212183260814

CPU : fp16 和fp32 耗时占比
  fp16: cpu不支持半精度
  fp32: pt_time:0.013942718505859375, q_pt_time:0.012614727020263672, pt_time/q_pt_time:1.1052731052731053

"""