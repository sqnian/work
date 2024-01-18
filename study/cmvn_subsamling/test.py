import torch
import conformer_infer_opt

import os

os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = "cuda:0"

def cmvn(input,mean,istd):
    output = input - mean
    output = output * istd
    return output

def quantize_to_int8(tensor, quant_range, clip_max):
        scale = quant_range / clip_max
        min_bound = - quant_range
        max_bound = quant_range
        # outputs = torch.clamp(
        #     (tensor.float() * scale).round_(), min_bound, max_bound)
        outputs = torch.clamp(
        (torch.floor(tensor.float() * scale+0.5)), min_bound, max_bound)
        quant_tensor = outputs.char()
        return quant_tensor


def gen_data(input,mean,istd):

    inputs = cmvn(input, mean, istd)  # 24,710,80
    inputs = inputs.unsqueeze(1) # 24,1,710,80 
    inputs_= torch.zeros([24,63,710,80],dtype=torch.float16).cuda()

    inputs = torch.cat((inputs,inputs_),dim=1).contiguous() # 24,64,710,80


    conv = torch.nn.Sequential(
            torch.nn.Conv2d(64, 256, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, 3, 2),
            torch.nn.ReLU()
        ).to(device).half()
   
    weight1 = conv[0].weight.data.clone()
    bias1 = conv[0].bias.data.clone()
    
    weight2 = conv[2].weight.data.clone()
    bias2 = conv[2].bias.data.clone()

 
    return inputs, conv, weight1, bias1, weight2, bias2


def gen_conv1_data(inputs, weight_, bias_):

    conv = torch.nn.Sequential(
            torch.nn.Conv2d(64, 256, 3, 2),
            torch.nn.ReLU()
        ).to(device).half()
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


def test(input,mean,istd):
    # test two 
    inputs, conv, weight1,bias1, weight2, bias2 = gen_data(input,mean,istd) # inputs shape: 24,64,710,80

    # normal 
    outputs_py = torch_imply(inputs,conv)
    # first conv2d+relu
    outputs_conv1 = gen_conv1_data(inputs,weight1,bias1)

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
    # print(" alpha_arr2", alpha_arr2.shape)
    # exit(0)

    out_py_int8 = quantize_to_int8(outputs_py, 127, amax_out)

    print("==================  start  qd   ================")
    # i_int8 = i_int8.permute(0,2,3,1).contiguous()
    # # test cmvn 
    input_cu = conformer_infer_opt.test_cmvn(input,i_int8,mean,istd,amax_in,True) 
    # diff = i_int8.flatten() - input_cu.flatten()
    # print(f"i_int8 flatten:{i_int8.flatten()[:50]}")
    # print(f"i_int8 shape:{i_int8.shape}")

    # print(f"input_cu flatten():{input_cu.flatten()[:50]}")
    # print(f"input_cu shape:{input_cu.shape}")
    # print("diff max:\n",diff.max())

    # exit(0)
  
    w_int8_arr = w_int8_arr.permute(0,2,3,1).contiguous()
    w2_int8_arr = w2_int8_arr.permute(0,2,3,1).contiguous()

    bias1 = (bias1.float() *127) / amax_out_conv1
    bias2 = (bias2.float() *127) / amax_out

    out_cu = conformer_infer_opt.test_func(input_cu,w_int8_arr,alpha_arr,bias1,
                                            w2_int8_arr,alpha_arr2,bias2)

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




   
if __name__ == "__main__":
    # torch.manual_seed(123)
    # torch.cuda.manual_seed_all(123)
    input = torch.load("./cmvn_input.pt")
    mean = torch.load("./cmvn_mean.pt")
    istd = torch.load("./cmvn_istd.pt")

    # test_data(input,mean,istd)
    # gen_data(input,mean,istd)

    test(input,mean,istd)
