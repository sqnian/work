from pickle import TRUE
import torch
import conformer_infer_opt
import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = "cuda:0"


def gen_data(batch_size, seq_len, feat_dim, odim):

    # inputs = torch.randn([batch_size, odim,seq_len,1],
    #                      device=device, dtype=torch.float32)
    
    inputs = torch.randn([20, 64,269,80],
                         device=device, dtype=torch.float32)
    print(f"inputs.shape:{inputs.shape}")  # torch.Size([24, 80, 200])

    conv =  torch.nn.Sequential(
            torch.nn.Conv2d(64, odim, kernel_size=1,stride=1,padding=0),  # kernel_size 15 padding 7
            torch.nn.SiLU()   # SiLU
    ).to(device)
   
    weight = conv[0].weight.data.clone()
    print(f"weight.shape:{weight.shape}") # torch.Size([256, 80, 1])
    
    bias = conv[0].bias.data.clone()
  
    return inputs, conv, weight, bias


def torch_imply(inputs, conv):
    res1 = conv(inputs)    
    return res1

def quantize_to_int8(tensor, quant_range, clip_max):
    scale = quant_range / clip_max
    min_bound = - quant_range
    max_bound = quant_range
    outputs = torch.clamp(
        (tensor.float() * scale).round_(), min_bound, max_bound)
    # outputs = torch.clamp(
    #     (torch.floor(tensor.float() * scale+0.5)), min_bound, max_bound)
    quant_tensor = outputs.char()
    return quant_tensor

if __name__ == "__main__":
    batch_size = 24
    seq_len = 200
    feat_dim = 80
    # batch_size = 1
    # seq_len = 3
    # feat_dim = 1
    # torch.cuda.benchmark = True
    odim = 256
    # torch.manual_seed(1234)
    # torch.cuda.manual_seed_all(1234)

    inputs, conv, weight,bias = gen_data(
        batch_size, seq_len, feat_dim, odim)
    
    #step 2 point wise conv
    pt_conv1 = torch_imply(inputs, conv)
    print(f"pt_conv1 shape:{pt_conv1.shape}") # [24, 256, 200]

    amax_out = torch.abs(pt_conv1).max().item()


    # print("amax_out:",amax_out)
    # exit(2)

    # step3 quant data
    # input data quant
    amax_in = torch.abs(inputs).max().item()
    i_int8 = quantize_to_int8(inputs, 127, amax_in)
    # print(f"i_int8:{i_int8}")

    # weight data quant 
    amax_wei_arr = []
    for i in range(weight.shape[0]):
        amax_wei_arr.append(torch.abs(weight[i]).max().item())
    # print(f"amax_wei1_arr :{amax_wei1_arr}")
    # print(f"torch.abs(weight1[0]).max().item():{torch.abs(weight1[i]).max().item()}")
    amax_wei_arr = torch.Tensor(amax_wei_arr)
    w_int8_arr = torch.zeros_like(weight,device=device, dtype=torch.int8)
    for i in range(weight.shape[0]):
        # w_int8_arr.append(quantize_to_int8(weight1[i], 127, amax_wei1_arr[i]))
        w_int8_arr[i] = quantize_to_int8(weight[i], 127, amax_wei_arr[i])

    
    # alpha

    print("==================  start  qd   ================")
        
    pt_conv1_int8 = quantize_to_int8(pt_conv1, 127, amax_out)
    # pt_conv1_int8 = pt_conv1_int8.permute(0,2,3,1)
    print(f"pt_conv1_int8 shape:{pt_conv1_int8.shape}")


    i_int8 = i_int8.permute(0,2,3,1)
    w_int8_arr = w_int8_arr.permute(0,2,3,1)

    # activation type num
    """
    typedef enum {
        CUINFER_ACTIVATION_SIGMOID = 0,
        CUINFER_ACTIVATION_RELU = 1,
        CUINFER_ACTIVATION_TANH = 2,
        CUINFER_ACTIVATION_CLIPPED_RELU = 3,
        CUINFER_ACTIVATION_ELU = 4,
        CUINFER_ACTIVATION_IDENTITY = 5,
        CUINFER_ACTIVATION_LEAKY_RELU = 6,
        CUINFER_ACTIVATION_SILU = 7,
        CUINFER_ACTIVATION_HARD_SWISH = 8,
        CUINFER_ACTIVATION_HARD_SIGMOID = 9,
        } cuinferActivationMode_t;
    """
    act_num = 7

    # # bias and alpha
    # bias = (bias * 127 ) / amax_out

    alpha_arr = []

    if act_num not in [0,7]:
        # alpha
        for i in range(weight.shape[0]):
            alpha_arr.append((amax_in * amax_wei_arr[i] ) / (127 * amax_out ))
        bias = (bias * 127 ) / amax_out
        print("run in here\n")
    else:
         for i in range(weight.shape[0]):
                alpha_arr.append((amax_in * amax_wei_arr[i] ) / (127 * 127 ))

    alpha_arr = torch.Tensor(alpha_arr).to(device)
    

    cuda_conv1 = conformer_infer_opt.test_conv(i_int8,i_int8.flatten(),w_int8_arr, w_int8_arr.flatten(),alpha_arr,bias,act_num,amax_out)   
    # NHWC
    cuda_conv1 = cuda_conv1.permute(0,3,1,2)
    print(f"cuda_conv1 shape:{cuda_conv1.shape}") 
    
   
    diff = torch.abs(pt_conv1_int8.flatten() - cuda_conv1.flatten())
    print(f"cuda_conv diff max: {diff.max()}")

    print(f"pt_conv1_int8 shape:{pt_conv1_int8.flatten().shape}")
    print(f"pt_conv1_int8:{pt_conv1_int8.flatten()[:100]}\n")
    print(f"cuda_conv1:{cuda_conv1.flatten()[:100]}\n")
    print(f"cuda_conv1 :{cuda_conv1.flatten().shape}\n")

    zero = (diff==0 ).sum()
    diff_count = diff.numel()
    print("zero total:\n",zero)
    print("zero %:\n",zero / diff_count)


    

    num_iter = 100
    torch.cuda.synchronize()
    start_time = time.time()
    for i in range(num_iter):
        pt_conv1 = torch_imply(inputs, conv)
    torch.cuda.synchronize()
    pt_time = time.time() - start_time

    torch.cuda.synchronize()
    start_time = time.time()
    for i in range(num_iter):
        cuda_conv1 = conformer_infer_opt.test_conv(i_int8,i_int8.flatten(),w_int8_arr, w_int8_arr.flatten(),alpha_arr,bias,act_num,amax_out)   
    torch.cuda.synchronize()
    cuda_time = time.time() - start_time
    print(f'time pt_time {pt_time/num_iter} cuda_time {cuda_time/num_iter} pt_time/cuda_time {pt_time/cuda_time}')




   