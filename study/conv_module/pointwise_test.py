from pickle import TRUE
import torch
import conformer_infer_opt
import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = "cuda:0"


def gen_data(batch_size, seq_len, feat_dim, odim):

    inputs = torch.randn([batch_size, feat_dim,seq_len,1],
                         device=device, dtype=torch.float32)
    print(f"inputs.shape:{inputs.shape}")  # torch.Size([24, 80, 200])

    conv =  torch.nn.Conv2d(feat_dim, odim, kernel_size=1,
            stride=1,
            padding=0,bias=True).to(device)
    weight = conv.weight.data.clone()
    print(f"weight.shape:{weight.shape}") # torch.Size([256, 80, 1])
    
    bias = conv.bias.data.clone()
    # print(f"bias.shape:{bias.shape}") 
    # bias=torch.randn(odim).to(device)
    # bias=torch.zeros(odim).to(device)
    
    return inputs, conv, weight, bias


def torch_imply(inputs, conv):
    res1 = conv(inputs)
    # res1 = torch.nn.functional.glu(res1, dim=1)
    return res1

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

if __name__ == "__main__":
    batch_size = 30 # 24
    seq_len = 200 
    feat_dim = 80
    # batch_size = 1
    # seq_len = 3
    # feat_dim = 1
    # torch.cuda.benchmark = True
    odim = 256
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    inputs, conv, weight,bias = gen_data(
        batch_size, seq_len, feat_dim, odim)
    
    #step 2 point wise conv
    pt_conv1 = torch_imply(inputs, conv)
    print(f"pt_conv1 shape:{pt_conv1.shape}") # [24, 256, 200]
    # exit(2)
    pt_glu = torch.nn.functional.glu(pt_conv1, dim=1)

    amax_out = torch.abs(pt_conv1).max().item()

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
    alpha_arr = []
    for i in range(weight.shape[0]):
        # alpha_arr.append( 1 / ((amax_in * amax_wei1_arr[i]) / (127 * amax_out)))
        alpha_arr.append((amax_in * amax_wei_arr[i] ) / (127 * amax_out))
        #  alpha_arr.append(0)

    alpha_arr = torch.Tensor(alpha_arr).to(device)
    
    # bias
    bias = (bias *127) / amax_out

    print("==================  start  conv test   ================")
        
    pt_conv1_int8 = quantize_to_int8(pt_conv1, 127, amax_out)
    # pt_conv1_int8 = pt_conv1_int8.permute(0,2,3,1)
    print(f"pt_conv1_int8 shape:{pt_conv1_int8.shape}")

    amax_out_glu = torch.abs(pt_glu).max().item()
    pt_glu_int8 = quantize_to_int8(pt_glu, 127, amax_out_glu)

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
    act_num = 5

    cuda_conv1 = conformer_infer_opt.test_conv(i_int8,i_int8.flatten(),w_int8_arr, w_int8_arr.flatten(),alpha_arr,bias,act_num,0)   
    # NHWC
    cuda_conv1 = cuda_conv1.permute(0,3,1,2).contiguous()
    print(f"cuda_conv1 shape:{cuda_conv1.shape}") 
    print(f"cuda_conv1 shape:{cuda_conv1.flatten()[:50]}") 

    print(f"pt_conv1_int8 :{pt_conv1_int8.flatten()[:50]}\n")
    print(f"pt_conv1_int8  shape:{pt_conv1_int8.shape}\n")

    diff = torch.abs(pt_conv1_int8.flatten() - cuda_conv1.flatten())
    print(f"cuda_conv1 diff max: {diff.max()}")
    # exit(0)

    print("==========start glu test ==================\n")
    # glu 
    # a = pt_conv1.shape(0),pt_conv1.shape(1),pt_conv1.shape(2),pt_conv1.shape(3)
    # pt_conv1_int8 =  torch.nn.functional.glu(pt_conv1_int8, dim=1)
    

    cu_out = torch.zeros_like(pt_glu_int8)
    # pt_conv1_int8 ：原始glu中输入数据的int8 格式
    amax_cuda_conv1 = torch.abs(pt_conv1).max().item()
    # cuda_conv1 = pt_conv1_int8
    cuda_glu = conformer_infer_opt.test_glu(cu_out,cuda_conv1,amax_cuda_conv1,amax_out_glu)
    # amax_cuda_conv1 = torch.abs(cuda_conv1).max().item()   # cuda_conv1 最大值为127.量化后的数据
    print("amax_cuda_conv1:\n",amax_cuda_conv1)
    print("amax_out_glu:\n",amax_out_glu)
    
    # cuda_glu = conformer_infer_opt.test_glu(cu_out,cuda_conv1,amax_cuda_conv1,amax_out_glu)


    print(f"cuda_glu:{cuda_glu.flatten()[:50]}\n")
    print(f"cuda_glu shape:{cuda_glu.shape}\n")

    print(f"pt_glu_int8 glu:{pt_glu_int8.flatten()[:50]}\n")
    print(f"pt_glu_int8 glu shape:{pt_glu_int8.shape}\n")


    diff = torch.abs(pt_glu_int8.flatten() - cuda_glu.flatten())
    print(f"cuda_conv1 diff max: {diff.max()}")
    

    # num_iter = 100
    # torch.cuda.synchronize()
    # start_time = time.time()
    # for i in range(num_iter):
    #     pt_conv1 = torch_imply(inputs, conv)
    #     pt_glu = torch.nn.functional.glu(pt_conv1, dim=1)

    # torch.cuda.synchronize()
    # pt_time = time.time() - start_time

    # torch.cuda.synchronize()
    # start_time = time.time()
    # for i in range(num_iter):
    #     cuda_conv1 = conformer_infer_opt.test_conv(i_int8,i_int8.flatten(),w_int8_arr, w_int8_arr.flatten(),alpha_arr,bias,act_num,0)   
    #     cuda_glu = conformer_infer_opt.test_glu(cu_out,pt_conv1_int8,amax_cuda_conv1,amax_out_glu)

    # torch.cuda.synchronize()
    # cuda_time = time.time() - start_time
    # print(f'time pt_time {pt_time/num_iter} cuda_time {cuda_time/num_iter} pt_time/cuda_time {pt_time/cuda_time}')




   