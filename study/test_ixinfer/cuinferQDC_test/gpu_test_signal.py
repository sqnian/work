import torch
import conformer_infer_opt
import os
import time
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
device = "cuda:0"



def gen_data(batch, seq_len, idim, odim):

   
    inputs = torch.randn([batch, 32, 56, 56], 
                         device=device, dtype=torch.float32) 

    conv = torch.nn.Sequential(
            torch.nn.Conv2d(32, odim, 3, 2),
            torch.nn.ReLU(),
            # torch.nn.Conv2d(odim, odim, 3, 2),
            # torch.nn.ReLU()
        ).to(device)
    
    weight1 = conv[0].weight.data.clone()
    bias1 = conv[0].bias.data.clone()

    return inputs, conv, weight1, bias1


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
    odim = 256

    # test two 
    inputs, conv, weight1,bias1 = gen_data(
        batch, time_len, idim, odim)

    # normal 
    outputs_py = torch_imply(inputs,conv)
    print(f"outputs_py shape:{outputs_py.shape}\n")


    # quant
    amax_in = torch.abs(inputs).max().item()
    # amax_wei1 = torch.abs(weight1).max().item() # perchannels array !!!!
    # weight1
    amax_wei1_arr = []
    for i in range(weight1.shape[0]):
        amax_wei1_arr.append(torch.abs(weight1[i]).max().item())
    # print(f"amax_wei1_arr :{amax_wei1_arr}")
    # print(f"torch.abs(weight1[0]).max().item():{torch.abs(weight1[i]).max().item()}")
    amax_wei1_arr = torch.Tensor(amax_wei1_arr)
    # amax_bias1 = torch.abs(bias1).max().item()

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
        # exit(4)
    # bias1_int8 = quantize_to_int8(bias1, 127, amax_bias1).to(device)

    alpha_arr = []
    for i in range(weight1.shape[0]):
        # alpha_arr.append( 1 / ((amax_in * amax_wei1_arr[i]) / (127 * amax_out)))
        alpha_arr.append((amax_in * amax_wei1_arr[i] ) / (127 * amax_out))
        #  alpha_arr.append(0)

    alpha_arr = torch.Tensor(alpha_arr)
    alpha_arr = alpha_arr.to(device)
    # print(f"alpha_arr:{alpha_arr}")

    out_py_int8 = quantize_to_int8(outputs_py, 127, amax_out)
    
    # conv[0].weight.data = w_int8
    # out = torch_imply(i_int8,conv)
    # print(f"out shape: {out.shape}")

    # // quant
    # out = quantize_to_int8(out, 127, amax_out)
    # out = out * alpha


    """
    调用底层op
    输入为:input,weight,bias,alpha
    输出为:int8量化的结果
    """
    print("==================  start  qd   ================")

    i_int8 = i_int8.permute(0,2,3,1)
    w_int8_arr = w_int8_arr.permute(0,2,3,1)

    bias1 = (bias1 *127) / amax_out
 
    
    out_cu = conformer_infer_opt.test_conv2d_relu(i_int8,i_int8.flatten(),w_int8_arr, w_int8_arr.flatten(),alpha_arr,bias1)



    out_py_int8 = out_py_int8.permute(0,2,3,1)

    print(f"out_py_int8 shape:{out_py_int8.shape}")
    print(f"out_py_int8:{out_py_int8.flatten()[:100]}\n")
    print(f"out_cu:{out_cu.flatten()[:100]}\n")
    print(f"out_cu :{out_cu.flatten().shape}\n")

    print(f"out_cu shape:{out_cu.shape}\n")

    
    print("==================diff analyse==========================")
    diff = torch.abs(out_py_int8.flatten()-out_cu.flatten())

    print(f"diff max:{diff.max()}\n")

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
        cuda_conv = conformer_infer_opt.test_conv2d_relu(i_int8,i_int8.flatten(),w_int8_arr, w_int8_arr.flatten(),alpha_arr,bias1)
    torch.cuda.synchronize()
    cuda_time = time.time() - start_time
    print(f'time pt_time {pt_time/num_iter} cuda_time {cuda_time/num_iter} pt_time/cuda_time {pt_time/cuda_time}')
    





if __name__ == "__main__":
    test()



