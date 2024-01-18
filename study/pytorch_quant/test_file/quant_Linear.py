import torch 
import os
from pytorch_quantization.nn import QuantLinear, TensorQuantizer
import  time

os.environ["CUDA_VISIBLE_DEVICES"]="0"
device="cuda:0"


def test():
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(123)

    inputs = torch.randn([2,3],dtype=torch.float16) # *10000
    print(f"inputs:{inputs}\n")

    linear = torch.nn.Linear(3,3,bias=True).half()
    weight = linear.weight.data.clone()
    bias = linear.bias.data.clone()

    outputs = linear(inputs)
    print(f"weight:{weight}\n")
    print(f"outputs:{outputs}\n")
    print(f"bias:{bias}\n")


    print("test quantization quant linear\n")
    inputs_quantizer = TensorQuantizer(QuantLinear.default_quant_desc_input)
    weight_quantizer = TensorQuantizer(QuantLinear.default_quant_desc_weight)
    # weight_quantizer = TensorQuantizer(QuantLinear.default_quant_desc_input)
    # bias_quantizer = TensorQuantizer(QuantLinear.default_quant_desc_weight)

    q_inputs = inputs_quantizer(inputs)
    q_weight = weight_quantizer(weight)

    # q_bias = bias_quantizer(bias)

    print(f"q_inputs:{q_inputs}\n")
    print(f"q_weight:{q_weight}\n")


    q_outputs = torch.nn.functional.linear(q_inputs,q_weight,bias).half()
    # q_outputs = torch.nn.functional.linear(q_inputs,q_weight,q_bias)


    print(f"q_outputs:{q_outputs}\n")


    diff = outputs.flatten() - q_outputs.flatten()
    print(f"diff max:{diff.max()}\n")


    print("time analyse\n")
    iter_num = 100
    torch.cuda.synchronize()
    start_time = time.time()
    for i in range(iter_num):
        outputs_ = linear(inputs)
        # q_outputs_ = torch.nn.functional.linear(q_inputs,q_weight)
    torch.cuda.synchronize()
    pt_time = time.time() - start_time

    torch.cuda.synchronize()
    start_time = time.time()
    for i in range(iter_num):
        # outputs_ = linear(inputs)
        q_outputs_ = torch.nn.functional.linear(q_inputs,q_weight,bias).half()
    torch.cuda.synchronize()
    q_pt_time = time.time() - start_time

    print("pt_time:{}, q_pt_time:{}, pt_time/q_pt_time:{}\n".format(pt_time,q_pt_time,pt_time/q_pt_time))


"""
GPU : 
  fp16: pt_time:0.0023584365844726562, q_pt_time:0.002114534378051758, pt_time/q_pt_time:1.1153455857481114
  fp32: pt_time:0.00220489501953125, q_pt_time:0.0015811920166015625, pt_time/q_pt_time:1.3944511459589868

CPU : fp16 和fp32 耗时占比
  fp16: pt_time:0.0014576911926269531, q_pt_time:0.0007116794586181641, pt_time/q_pt_time:2.0482412060301507
  fp32:pt_time:0.0014755725860595703, q_pt_time:0.0006556510925292969, pt_time/q_pt_time:2.2505454545454544

"""








    









if __name__ == "__main__":
    test()