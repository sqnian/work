import torch
import conformer_infer_opt
import os
import time
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = "cuda:0"

"""
conformer中subsampling模块 input and weight shape padding
"""
def gen_conv1():
    
    conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, 256, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, 3, 2),
            torch.nn.ReLU()
        ).to(device).float()
   
    weight1 = conv[0].weight.data 
    bias1 = conv[0].bias.data
    
    weight2 = conv[2].weight.data  
    bias2 =  conv[2].bias.data 

 
    return conv,weight1, bias1, weight2, bias2

def gen_conv64(weight1,bias1, weight2, bias2):

    conv = torch.nn.Sequential(
            torch.nn.Conv2d(64, 256, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, 3, 2),
            torch.nn.ReLU()
        ).to(device).float()
    
    # print("weight1 shape:", weight1.shape)
    # w1 = torch.zeros([weight1.shape[0], 63,weight1.shape[2],weight1.shape[3]], device=device, dtype=torch.float32)
    # w1 = torch.ones([weight1.shape[0], 63,weight1.shape[2],weight1.shape[3]], device=device, dtype=torch.float32)
    w1 = torch.randn([weight1.shape[0], 63,weight1.shape[2],weight1.shape[3]], device=device, dtype=torch.float32)
    weight1 = torch.cat((weight1,w1),dim=1)

    conv[0].weight.data = weight1
    conv[0].bias.data = bias1 
    
    conv[2].weight.data =  weight2
    conv[2].bias.data = bias2 

 
    return conv

def torch_imply(inputs, conv):
    y = conv(inputs)
    # b, c, t, f = y.size()
    # # print(y.size())
    # y = y.transpose(1, 2).contiguous().view(b, t, c * f)
    return y

def test():

    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    # inputs = torch.randn([24, 64, 710, 80], device=device, dtype=torch.float32) 
    inputs_1 = torch.randn([24, 1, 710, 80], device=device, dtype=torch.float32)

    # inputs_1 = torch.randint(-127, 127,(24, 1, 710, 80), device=device, dtype=torch.float32)
    # inputs_1 = torch.randint(-64, 64,(24, 1, 710, 80), device=device, dtype=torch.float32)
    print("inputs_1 shape:\n",inputs_1.shape)
    print("inputs_1 :\n",inputs_1.flatten()[:100])

    inputs_ = torch.zeros([24, 63, 710, 80], device=device, dtype=torch.float32)
    # inputs_ = torch.ones([24, 63, 710, 80], device=device, dtype=torch.float32)
    # inputs_ = torch.randn([24, 63, 710, 80], device=device, dtype=torch.float32)
    
    inputs_2 = torch.cat((inputs_1,inputs_),dim=1)

    print("inputs_2 shape:\n",inputs_2.shape)
    print("inputs_2 :\n",inputs_2.flatten()[:100])


    conv1,weight1, bias1, weight2, bias2 = gen_conv1()

    conv64 = gen_conv64(weight1,bias1, weight2, bias2)

    res1 = torch_imply(inputs_1, conv1)
    res2 = torch_imply(inputs_2, conv64)

    # diff 
    print("测试res2输出和res1输出之间的分析,填充")
    diff = torch.abs(res1.flatten() - res2.flatten())
    print("res1:\n",res1.flatten()[:200])
    print("res2:\n",res2.flatten()[:200])
    print("res1 shape:\n",res1.shape)
    print("res2 shape:\n",res2.shape)

    print("diff max:",diff.max())

  


if __name__ == "__main__":
    test()
