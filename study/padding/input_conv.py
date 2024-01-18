import torch
import conformer_infer_opt
import os
import time
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = "cuda:0"


"""
conformer中convolution 模块 input 三维-->四维; conv1d--->conv2d

"""


def pointwise_1d():

    conv =  torch.nn.Conv1d(256, 2*256, kernel_size=1,
            stride=1,
            padding=0,bias=True).to(device)
    weight = conv.weight.data.clone()
    print("weight pointwise_1d :")

    print(f"weight.shape:{weight.shape}") # 
    
    bias = conv.bias.data.clone()
    
    return conv, weight, bias

def pointwise_2d(weight2, bias2):
    
    conv =  torch.nn.Conv2d(256, 2*256, kernel_size=1,
            stride=1,
            padding=0,bias=True).to(device)

    weight2 = weight2.reshape(weight2.shape[0], weight2.shape[1], weight2.shape[2], 1)
    conv.weight.data =  weight2 
    print("weight pointwise_2d :")
    print(f"weight.shape:{weight2.shape}") # 
    
    conv.bias.data = bias2
    
    return conv, weight2, bias2


def torch_impl_1(inputs,conv):
    output = conv(inputs) # 
    output = torch.nn.functional.glu(output, dim=1)

    return output

def pointwise_1_test(inputs):
    print("===================================pointwise_1_test ====================================================\n")

    pointwise_conv1d, weight, bias = pointwise_1d()

    output = torch_impl_1(inputs,pointwise_conv1d)
    print(f"output shape :{output.shape}\n")

    inputs_ = inputs.reshape(inputs.shape[0], inputs.shape[1],inputs.shape[2],1)
    pointwise_conv2d, weigh2t, bias2 = pointwise_2d( weight, bias )

    output_ = torch_impl_1(inputs_,pointwise_conv2d)
    print(f"output_ shape :{output_.shape}\n")
    print("===================================input data========================")
    print(f"inputs.flatten() :{inputs.flatten()[:100]}\n")
    print(f"inputs_.flatten() :{inputs_.flatten()[:100]} \n")
    diff_in = torch.abs(inputs.flatten() - inputs_.flatten())
    print(f"diff_in max:{diff_in.max()}\n")

    output_ = output_.reshape(inputs.shape[0], inputs.shape[1],inputs.shape[2])
    

    diff = torch.abs(output.flatten() - output_.flatten())
    print("===================================output data========================")
    print(f"output_ shape :{output_.shape}\n")
    print(f"diff max:{diff.max()}\n")
    print(f"output.flatten() :{output.flatten()[:100]}\n")
    print(f"output_.flatten() :{output_.flatten()[:100]} \n")
    # return 


def depthwise_1d():
    
    conv = torch.nn.Sequential(
            torch.nn.Conv1d(256, 256, kernel_size=15,    # # kernel_size 15 padding 7
            stride=1,
            padding=7,
            groups=256,
            bias=True),
            # torch.nn.SiLU() 
            ).to(device)

    weight = conv[0].weight.data.clone()
    print("weight depthwise_1d :")

    print(f"weight.shape:{weight.shape}") # 
    
    bias = conv[0].bias.data.clone()
    
    return conv, weight, bias

def depthwise_2d(weight2, bias2):
  
    conv = torch.nn.Sequential(
            torch.nn.Conv2d(256, 256, kernel_size=15,   # # kernel_size 15 padding 7
            stride=1,
            padding=(7,0),
            groups=256,
            bias=True),
            # torch.nn.SiLU() 
            ).to(device)

    # weight = conv[0].weight.data.clone()
    # print(f"weight.shape:{weight.shape}") # 


    weight2 = weight2.reshape(weight2.shape[0], weight2.shape[1], 15,1)
    # w = torch.zeros([weight2.shape[0], weight2.shape[1], 14,weight2.shape[2]])

    conv[0].weight.data =  weight2 
    print("weight depthwise_2d :")
    print(f"weight.shape:{weight2.shape}") # 
    
    conv[0].bias.data = bias2
    
    return conv

def torch_impl_2(inputs,conv):
    output = conv(inputs) # 
    return output

def depthwise_test(inputs):
    print("===================================depthwise_test ===============================================\n")

    pointwise_conv1d, weight, bias = depthwise_1d()

    output = torch_impl_2(inputs,pointwise_conv1d)
    print(f"output shape :{output.shape}\n")

    inputs_ = inputs.reshape(inputs.shape[0], inputs.shape[1],inputs.shape[2],1)
    pointwise_conv2d = depthwise_2d( weight, bias )

    output_ = torch_impl_2(inputs_,pointwise_conv2d)
    print(f"output_ shape :{output_.shape}\n")
    print("===================================input data========================")
    print(f"inputs.flatten() :{inputs.flatten()[:100]}\n")
    print(f"inputs_.flatten() :{inputs_.flatten()[:100]} \n")
    diff_in = torch.abs(inputs.flatten() - inputs_.flatten())
    print(f"diff_in max:{diff_in.max()}\n")

    # output_ = output_.reshape(inputs.shape[0], inputs.shape[1],inputs.shape[2])
    # print(f"output_ shape :{output_.shape}\n")

    diff = torch.abs(output.flatten() - output_.flatten()[:1228800])
    print("===================================output data========================")
    print(f"output_ shape :{output_.shape}\n")
    print(f"diff max:{diff.max()}\n")
    print(f"output.flatten() :{output.flatten()[:100]}\n")
    print(f"output_.flatten() :{output_.flatten()[:100]} \n")


def pointwise2_1d():
    
    conv =  torch.nn.Conv1d(256, 256, kernel_size=1,
            stride=1,
            padding=0,bias=True).to(device)
    weight = conv.weight.data.clone()
    print("weight pointwise_1d :")

    print(f"weight.shape:{weight.shape}") # 
    
    bias = conv.bias.data.clone()
    
    return conv, weight, bias

def pointwise2_2d(weight2, bias2):
    
    conv =  torch.nn.Conv2d(256, 256, kernel_size=1,
            stride=1,
            padding=0,bias=True).to(device)

    weight2 = weight2.reshape(weight2.shape[0], weight2.shape[1], weight2.shape[2], 1)
    conv.weight.data =  weight2 
    print("weight pointwise_2d :")
    print(f"weight.shape:{weight2.shape}") # 
    
    conv.bias.data = bias2
    
    return conv, weight2, bias2



def pointwise_2_test(inputs):
    print("===========================================pointwise_2_test ====================================\n")

    pointwise_conv1d, weight, bias = pointwise2_1d()

    output = torch_impl_2(inputs,pointwise_conv1d)
    print(f"output shape :{output.shape}\n")

    inputs_ = inputs.reshape(inputs.shape[0], inputs.shape[1],inputs.shape[2],1)
    pointwise_conv2d, weigh2t, bias2 = pointwise2_2d( weight, bias )

    output_ = torch_impl_2(inputs_,pointwise_conv2d)
    print(f"output_ shape :{output_.shape}\n")
    print("===================================input data========================")
    print(f"inputs.flatten() :{inputs.flatten()[:100]}\n")
    print(f"inputs_.flatten() :{inputs_.flatten()[:100]} \n")
    diff_in = torch.abs(inputs.flatten() - inputs_.flatten())
    print(f"diff_in max:{diff_in.max()}\n")

    output_ = output_.reshape(inputs.shape[0], inputs.shape[1],inputs.shape[2])
    print(f"output_ shape :{output_.shape}\n")

    diff = torch.abs(output.flatten() - output_.flatten())
    print("===================================output data========================")
    print(f"diff max:{diff.max()}\n")
    print(f"output.flatten() :{output.flatten()[:100]}\n")
    print(f"output_.flatten() :{output_.flatten()[:100]} \n")
    # return 


def run():
    inputs_1 = torch.randn([24, 256,200],
                         device=device, dtype=torch.float32)


    
    pointwise_1_test(inputs_1)

    depthwise_test(inputs_1)

    pointwise_2_test(inputs_1)
    
    




if __name__ =="__main__":
    torch.manual_seed(1233)
    torch.cuda.manual_seed_all(1234)
    run()
