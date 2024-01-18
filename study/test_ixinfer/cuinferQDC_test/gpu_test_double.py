import torch
import conformer_infer_opt
import os
import time
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
device = "cuda:0"



def gen_data(batch, seq_len, idim, odim):

    inputs = torch.randn([batch, 32, 56, 56], 
                         device=device, dtype=torch.float32) 

    conv = torch.nn.Sequential(
            torch.nn.Conv2d(32, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU()
        ).to(device)
   
    weight1 = conv[0].weight.data.clone()
    bias1 = conv[0].bias.data.clone()
    
    weight2 = conv[2].weight.data.clone()
    bias2 = conv[2].bias.data.clone()

 
    return inputs, conv, weight1, bias1, weight2, bias2

def gen_conv1_data(inputs, odim, weight_, bias_):

    conv = torch.nn.Sequential(
            torch.nn.Conv2d(32, odim, 3, 2),
            torch.nn.ReLU()
        ).to(device)
    conv[0].weight.data = weight_
    conv[0].bias.data = bias_

    out = conv(inputs)
    return out 


def torch_imply(inputs, conv):
    y = conv(inputs)
    # b, c, t, f = y.size()
    # # print(y.size())
    # y = y.transpose(1, 2).contiguous().view(b, t, c * f)
    return y


def quantize_to_int8(tensor, quant_range, clip_max):
    scale = quant_range / clip_max
    min_bound = - quant_range
    max_bound = quant_range
    outputs = torch.clamp(
        (tensor.float() * scale).round_(), min_bound, max_bound)
    quant_tensor = outputs.char()
    return quant_tensor

def test():
    batch = 16
    time_len = 269  # 269
    idim = 80  # 80
    odim = 128

    # test two 
    inputs, conv, weight1,bias1, weight2, bias2 = gen_data(
        batch, time_len, idim, odim)

    # print(f"inputs shape:{inputs.shape}")
    # print(f"weight1 shape:{weight1.shape}")
    # print(f"bias1 shape:{bias1.shape}") 
    # print(f"weight12 shape:{weight2.shape}")
    # print(f"bias2 shape:{bias2.shape}") 
 
    # normal 
    outputs_py = torch_imply(inputs,conv)
    # print(f"outputs_py shape:{outputs_py.shape}\n")

    outputs_conv1 = gen_conv1_data(inputs,odim,weight1,bias1)

    # quant
    amax_in = torch.abs(inputs).max().item()
    # weight1
    amax_wei1_arr = []
    for i in range(weight1.shape[0]):
        amax_wei1_arr.append(torch.abs(weight1[i]).max().item())
    # print(f"amax_wei1_arr :{amax_wei1_arr}")
    # print(f"torch.abs(weight1[0]).max().item():{torch.abs(weight1[i]).max().item()}")
    amax_wei1_arr = torch.Tensor(amax_wei1_arr)

    # weight2
    amax_wei2_arr = []
    for i in range(weight2.shape[0]):
        amax_wei2_arr.append(torch.abs(weight2[i]).max().item())
    # print(f"amax_wei1_arr :{amax_wei1_arr}")
    # print(f"torch.abs(weight1[0]).max().item():{torch.abs(weight1[i]).max().item()}")
    amax_wei2_arr = torch.Tensor(amax_wei2_arr)
  
    amax_out_conv1 = torch.abs(outputs_conv1).max().item()
    amax_out = torch.abs(outputs_py).max().item()

    i_int8 = quantize_to_int8(inputs, 127, amax_in)
    # print(f"i_int8:{i_int8}")
    # w_int8 = quantize_to_int8(weight1, 127, amax_wei1)

    # weight1 int8
    w_int8_arr = torch.zeros_like(weight1,device=device, dtype=torch.int8)
    for i in range(weight1.shape[0]):
        # w_int8_arr.append(quantize_to_int8(weight1[i], 127, amax_wei1_arr[i]))
        w_int8_arr[i] = quantize_to_int8(weight1[i], 127, amax_wei1_arr[i])
        # print(f"w_int8_arr:{w_int8_arr[0]}")
   
    ## weight2 int8
    w2_int8_arr = torch.zeros_like(weight2,device=device, dtype=torch.int8)
    for i in range(weight2.shape[0]):
        # w_int8_arr.append(quantize_to_int8(weight1[i], 127, amax_wei1_arr[i]))
        w2_int8_arr[i] = quantize_to_int8(weight2[i], 127, amax_wei2_arr[i])

    alpha_arr = []
    for i in range(weight1.shape[0]):
        # alpha_arr.append( 1 / ((amax_in * amax_wei1_arr[i]) / (127 * amax_out)))
        alpha_arr.append((amax_in * amax_wei1_arr[i] ) / (127 * amax_out_conv1))
        #  alpha_arr.append(0)

    alpha_arr = torch.Tensor(alpha_arr)
    alpha_arr = alpha_arr.to(device)
    # print(f"alpha_arr:{alpha_arr}")
    # exit(0)

    alpha_arr2 = []
    for i in range(weight2.shape[0]):
        # alpha_arr.append( 1 / ((amax_in * amax_wei1_arr[i]) / (127 * amax_out)))
        alpha_arr2.append((amax_out_conv1 * amax_wei2_arr[i] ) / (127 * amax_out))
        #  alpha_arr.append(0)

    alpha_arr2 = torch.Tensor(alpha_arr2)
    alpha_arr2 = alpha_arr2.to(device)

    out_py_int8 = quantize_to_int8(outputs_py, 127, amax_out)
    
    """
    调用底层op
    输入为:input,weight,bias,alpha
    输出为:int8量化的结果
    """
    print("==================  start  qd   ================")

    # i_int8 = i_int8.transpose(2,1).transpose(3,2)
    i_int8 = i_int8.permute(0,2,3,1)
    # print(f"i_int8 [:100] :{i_int8.flatten()[:100]}\n")
    # w_int8_arr = w_int8_arr.transpose(2,1).transpose(3,2)
    w_int8_arr = w_int8_arr.permute(0,2,3,1)
    w2_int8_arr = w2_int8_arr.permute(0,2,3,1)

    # print(f"w_int8_arr  :{w_int8_arr.size()}")
    # exit(0)
    # print("amax_out:",amax_out)
    bias1 = (bias1 *127) / amax_out_conv1
    bias2 = (bias2 *127) / amax_out
    # print("bias1:",bias1)
    # # print(f"w_int8_arr:{w_int8_arr}")

    
    # out_cu = conformer_infer_opt.test_conv_qd(i_int8,i_int8.flatten(),w_int8_arr, w_int8_arr.flatten(),alpha_arr,bias1)
    # out_cu = conformer_infer_opt.test_conv2d(i_int8,i_int8.flatten(),w_int8_arr, w_int8_arr.flatten(),alpha_arr,bias1)

    out_cu = conformer_infer_opt.test_basic(i_int8,i_int8.flatten(),w_int8_arr, w_int8_arr.flatten(),alpha_arr,bias1,
                                            w2_int8_arr, w2_int8_arr.flatten(),alpha_arr2,bias2)

    # out_cu = out_cu.transpose(3,2).transpose(2,1)
    # print(f"out_cu :{out_cu }")

    print("==================  end  qd   ================")
    # out_py_int8 = out_py_int8.transpose(2,1).transpose(3,2)
    out_py_int8 = out_py_int8.permute(0,2,3,1)

    print(f"out_py_int8 shape:{out_py_int8.shape}")
    print(f"out_py_int8:{out_py_int8.flatten()[:100]}\n")
    print(f"out_cu:{out_cu.flatten()[:100]}\n")
    print(f"out_cu :{out_cu.flatten().shape}\n")

    print(f"out_cu shape:{out_cu.shape}\n")

    diff = torch.abs(out_py_int8.flatten()-out_cu.flatten())

    print(f"diff max:{diff.max()}\n")
    print("==================diff analyse==========================")
    diff_count = diff.numel()
    diff_zero = (diff == 0).sum()
    diff_two = (diff > 1).sum()
    diff_one = diff_count - diff_zero - diff_two
    diff_check = diff_one + diff_zero + diff_two
    print(f"diff_check:{diff_check}\n")

    print(f"diff_count :{diff_count}\n") 
    print(f"diff_zero :{diff_zero}\n") 
    print(f"diff_one :{diff_one}\n") 
    print(f"diff_two :{diff_two}\n") 
    print(f"zero count: {diff_zero}\n") 
    print(f"zero count percent: {(diff_zero) / diff_count }\n")
    print(f"not zero count: {(diff_two + diff_one)}\n") 
    print(f"not zero count percent:{(diff_two + diff_one) / diff_count}\n") 


    print("=======================time analyse====================================\n")
    num_iter = 100
    torch.cuda.synchronize()
    start_time = time.time()
    for i in range(num_iter):
        pt_conv = torch_imply(inputs,conv)
    torch.cuda.synchronize()
    pt_time = time.time() - start_time

    torch.cuda.synchronize()
    start_time = time.time()
    for i in range(num_iter):
        cuda_conv = conformer_infer_opt.test_basic(i_int8,i_int8.flatten(),w_int8_arr, w_int8_arr.flatten(),alpha_arr,bias1,
                                            w2_int8_arr, w2_int8_arr.flatten(),alpha_arr2,bias2)
    torch.cuda.synchronize()
    cuda_time = time.time() - start_time
    print(f'time pt_time {pt_time/num_iter} cuda_time {cuda_time/num_iter} pt_time/cuda_time {pt_time/cuda_time}')
    





if __name__ == "__main__":
    test()


