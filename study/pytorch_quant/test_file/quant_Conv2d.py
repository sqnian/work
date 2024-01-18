import torch 
import time 
import os
from pytorch_quantization import tensor_quant
from pytorch_quantization.nn import QuantLinear, TensorQuantizer,QuantConv2d


os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = "cuda:0"

"""
test QuantConv1d and how to use it

"""

def get_Conv2d():
    conv2d = torch.nn.Conv2d(64,128,3,2)
    weight = conv2d.weight.data.clone()
    # bias = conv2d.bias.data.clone()
    bias = torch.zeros_like(conv2d.bias.data.clone())
    print("bias:",bias)

    return conv2d,weight,bias

def conv2d_Imp(inputs,conv2d):
    outputs = conv2d(inputs)

    return outputs


def test():

    inputs =torch.randn([16, 64, 128,128],dtype=torch.float32)

    conv2d,weight,bias = get_Conv2d()

    outputs = conv2d_Imp(inputs,conv2d)
    # print(f"weight : {weight}\n")
    # print(f"bias : {bias}\n")
    print(f"outputs flatten()[:50]: {outputs.flatten()[:50]}\n")


    print("test quant conv2d \n")

    inputs_quantizer = TensorQuantizer(QuantConv2d.default_quant_desc_input)
    weight_quantizer = TensorQuantizer(QuantConv2d.default_quant_desc_weight)
    bias_quantizer = TensorQuantizer(QuantConv2d.default_quant_desc_input)

    # print(QuantConv1d.default_quant_desc_input)
    # print(QuantConv1d.default_quant_desc_weight)
    # print(QuantConv1d.default_quant_desc_input)
    # print(inputs_quantizer)
    # exit(0)


    q_inputs = inputs_quantizer(inputs)
    q_weight = weight_quantizer(weight)

    q_bias = bias_quantizer(bias)
    print("q_bias",q_bias)

    # print(f"q_inputs:{q_inputs}\n")
    # print(f"q_weight:{q_weight}\n")


    q_outputs = torch.nn.functional.conv2d(q_inputs,q_weight,q_bias,2)
    print(f"q_outputs flatten()[:50]:{q_outputs.flatten()[:50]}\n")


    diff = outputs.flatten() - q_outputs.flatten()
    print(f"diff max:{diff.max()}\n")


    print("time analyse\n")
    iter_num = 100
    torch.cuda.synchronize()
    start_time = time.time()
    for i in range(iter_num):
        outputs_ =  conv2d_Imp(inputs,conv2d)
        # q_outputs_ = torch.nn.functional.linear(q_inputs,q_weight)
    torch.cuda.synchronize()
    pt_time = time.time() - start_time

    torch.cuda.synchronize()
    start_time = time.time()
    for i in range(iter_num):
        # outputs_ = linear(inputs)
        q_outputs_ = torch.nn.functional.conv2d(q_inputs,q_weight,q_bias,2)
    torch.cuda.synchronize()
    q_pt_time = time.time() - start_time

    print("pt_time:{}, q_pt_time:{}, pt_time/q_pt_time:{}\n".format(pt_time,q_pt_time,pt_time/q_pt_time))



if __name__ == "__main__":
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)

    test()


"""
GPU : 
  fp16: pt_time:0.0023584365844726562, q_pt_time:0.002114534378051758, pt_time/q_pt_time:1.1153455857481114
  fp32: pt_time:0.14644742012023926, q_pt_time:0.14627385139465332, pt_time/q_pt_time:1.0011866011862751

CPU : fp16 和fp32 耗时占比
  fp16: cpu不支持半精度
  fp32: pt_time:1.0687623023986816, q_pt_time:1.1298277378082275, pt_time/q_pt_time:0.9459515522888402

"""