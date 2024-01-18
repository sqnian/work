import torch
import os
import conformer_infer_opt
import time

os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = "cuda:0"

"""
convolution module:
    pointwise_conv1
    glu
    depthwise_conv
    silu
    pointwise_conv2
"""

# pointwise_conv1
def pointwise_conv1():
    point_conv1 = torch.nn.Conv2d(64,128,kernel_size=1,
            stride=1,
            padding=0,bias=True).to(device)

    weight_point1  = point_conv1.weight.data.clone()
    bias1_point1  = point_conv1.bias.data.clone()
    return point_conv1, weight_point1, bias1_point1
    
# depthwise_conv + silu
def depthwise_conv():
    depth_conv = torch.nn.Sequential(
                    torch.nn.Conv2d(64, 128, kernel_size=1,stride=1,padding=0),  # kernel_size 15 padding 7
                    torch.nn.SiLU()   # SiLU
    ).to(device)
    weight_depth = depth_conv[0].weight.data.clone()
    bias_depth = depth_conv[0].bias.data.clone()

    return depth_conv,weight_depth, bias_depth
    
# pointwise_conv2
def pointwise_conv2():
    point_conv2 = torch.nn.Conv2d(128, 256, kernel_size=1,
                                    stride=1,
                                    padding=0,bias=True).to(device)
    weight_point2  = point_conv2.weight.data.clone()
    bias1_point2  = point_conv2.bias.data.clone()

    return point_conv2, weight_point2, bias1_point2

# quant
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


def conv_Module(inputs):
    # get conv ,weight,bias
    point_conv1, weight_point1, bias1_point1 = pointwise_conv1()
    depth_conv,weight_depth, bias_depth = depthwise_conv()
    point_conv2, weight_point2, bias1_point2  = pointwise_conv2()

    # pointwise_conv1 + glu
    out_pc1 = point_conv1(inputs)
    out_pc1_glu = torch.nn.functional.glu(out_pc1,dim=1)

    # depthwise_conv + silu
    out_dc = depth_conv(out_pc1_glu)

    # pointwise_conv2
    out_pc2 = point_conv2(out_dc)

    # time iteration 100
    num_iter = 100
    torch.cuda.synchronize()
    start_time = time.time()
    for i in range(num_iter):
        out_pc1_ = point_conv1(inputs)
        out_pc1_glu_ = torch.nn.functional.glu(out_pc1_,dim=1)
        # depthwise_conv + silu
        out_dc_ = depth_conv(out_pc1_glu_)
        # pointwise_conv2
        out_pc2_ = point_conv2(out_dc_)
    torch.cuda.synchronize()
    pt_time = time.time() - start_time


    return point_conv1, weight_point1, bias1_point1, out_pc1, out_pc1_glu, depth_conv,weight_depth, bias_depth, out_dc, point_conv2, weight_point2, bias1_point2,out_pc2,pt_time


# quant pointwise_conv1 + glu
def quant_test_pc1_glu(inputs,out_pc1,out_pc1_glu, weight_point1, bias1_point1):
    """
    1、pointwise_conv1 quant
    2、glu quant
    """
    print("==================  start  pointwise_conv1 test   ================")

    # pointwise_conv1 quant
    # input2 quant
    amax_in = torch.abs(inputs).max().item()
    i_int8 = quantize_to_int8(inputs, 127, amax_in)

    # weight data quant 
    amax_wei_arr = []
    for i in range(weight_point1.shape[0]):
        amax_wei_arr.append(torch.abs(weight_point1[i]).max().item())
    # print(f"amax_wei1_arr :{amax_wei1_arr}")
    # print(f"torch.abs(weight1[0]).max().item():{torch.abs(weight1[i]).max().item()}")
    amax_wei_arr = torch.Tensor(amax_wei_arr)
    w_int8_arr = torch.zeros_like(weight_point1,device=device, dtype=torch.int8)
    for i in range(weight_point1.shape[0]):
        # w_int8_arr.append(quantize_to_int8(weight1[i], 127, amax_wei1_arr[i]))
        w_int8_arr[i] = quantize_to_int8(weight_point1[i], 127, amax_wei_arr[i])

    amax_out = torch.abs(out_pc1).max().item()
    out_pc1_int8 = quantize_to_int8(out_pc1, 127, amax_out)

    # alpha
    alpha_arr = []
    for i in range(weight_point1.shape[0]):
        # alpha_arr.append( 1 / ((amax_in * amax_wei1_arr[i]) / (127 * amax_out)))
        alpha_arr.append((amax_in * amax_wei_arr[i] ) / (127 * amax_out))
        #  alpha_arr.append(0)

    alpha_arr = torch.Tensor(alpha_arr).to(device)
    
    # bias
    bias1_point1 = (bias1_point1 *127) / amax_out
    i_int8 = i_int8.permute(0,2,3,1)
    w_int8_arr = w_int8_arr.permute(0,2,3,1)

    act_num = 5
    q_out_pc1 = conformer_infer_opt.test_conv(i_int8,i_int8.flatten(),w_int8_arr, w_int8_arr.flatten(),alpha_arr,bias1_point1,act_num,0)
    # shape nchw
    q_out_pc1 = q_out_pc1.permute(0,3,1,2).contiguous()
    diff = torch.abs(q_out_pc1.flatten() - out_pc1_int8.flatten())
    print(f"pointwise_conv1 diff max: {diff.max()}")

    print("==========start glu test ==================\n")
    amax_glu = torch.abs(out_pc1_glu).max().item()
    out_pc1_glu_int8 = quantize_to_int8(out_pc1_glu, 127, amax_glu)

    cu_out = torch.zeros_like(out_pc1_glu_int8)

    q_out_glu = conformer_infer_opt.test_glu(cu_out,q_out_pc1,amax_out,amax_glu)

    diff = torch.abs(q_out_glu.flatten() - out_pc1_glu_int8.flatten())
    print(f"  glu diff max: {diff.max()}\n")

    # time iteration 100
    num_iter = 100
    torch.cuda.synchronize()
    start_time = time.time()
    for i in range(num_iter):
        q_out_pc1_ = conformer_infer_opt.test_conv(i_int8,i_int8.flatten(),w_int8_arr, w_int8_arr.flatten(),alpha_arr,bias1_point1,act_num,0)
        q_out_glu_ = conformer_infer_opt.test_glu(cu_out,out_pc1_int8,amax_out,amax_glu)
    torch.cuda.synchronize()
    step1_time = time.time() - start_time

    return  q_out_glu,out_pc1_glu_int8,step1_time


# depthwise_conv + silu
def quant_test_dc_silu(out_pc1_glu, q_out_glu,out_dc,weight_depth, bias_depth):
    

    amax_in = torch.abs(out_pc1_glu).max().item()
    # i_int8 = quantize_to_int8(out_pc1_glu,127,amax_in)
    i_int8 = q_out_glu

     # weight data quant 
    amax_wei_arr = []
    for i in range(weight_depth.shape[0]):
        amax_wei_arr.append(torch.abs(weight_depth[i]).max().item())
    # print(f"amax_wei1_arr :{amax_wei1_arr}")
    # print(f"torch.abs(weight1[0]).max().item():{torch.abs(weight1[i]).max().item()}")
    amax_wei_arr = torch.Tensor(amax_wei_arr)
    w_int8_arr = torch.zeros_like(weight_depth,device=device, dtype=torch.int8)
    for i in range(weight_depth.shape[0]):
        # w_int8_arr.append(quantize_to_int8(weight1[i], 127, amax_wei1_arr[i]))
        w_int8_arr[i] = quantize_to_int8(weight_depth[i], 127, amax_wei_arr[i])

    amax_out = torch.abs(out_dc).max().item()
    out_dc_int8 = quantize_to_int8(out_dc, 127, amax_out)

    # alpha
    alpha_arr = []
    for i in range(weight_depth.shape[0]):
        # alpha_arr.append( 1 / ((amax_in * amax_wei1_arr[i]) / (127 * amax_out)))
        alpha_arr.append((amax_in * amax_wei_arr[i] ) / (127 * 127))
        #  alpha_arr.append(0)

    alpha_arr = torch.Tensor(alpha_arr).to(device)
    
    # bias

    i_int8 = i_int8.permute(0,2,3,1).contiguous()
    w_int8_arr = w_int8_arr.permute(0,2,3,1).contiguous()

    act_num = 7
    q_out_dc = conformer_infer_opt.test_conv(i_int8,i_int8.flatten(),w_int8_arr, w_int8_arr.flatten(),alpha_arr,bias_depth,act_num,amax_out)   
    # shape nchw
    q_out_dc = q_out_dc.permute(0,3,1,2).contiguous()
    diff = torch.abs(q_out_dc.flatten() - out_dc_int8.flatten())
    print("==========start depthwise_conv + silu test ==================\n")
    print(f"depthwise_conv + silu diff max: {diff.max()}")


     # time iteration 100
    num_iter = 100
    torch.cuda.synchronize()
    start_time = time.time()
    for i in range(num_iter):
        q_out_dc_ = conformer_infer_opt.test_conv(i_int8,i_int8.flatten(),w_int8_arr, w_int8_arr.flatten(),alpha_arr,bias_depth,act_num,amax_out)   

    torch.cuda.synchronize()
    step2_time = time.time() - start_time


    return q_out_dc,out_dc_int8,step2_time


# pointwise_conv2
def quant_test_pc2(out_dc,q_out_dc,out_pc2,weight_point2, bias1_point2):
    amax_in = torch.abs(out_dc).max().item()
    # i_int8 = quantize_to_int8(out_dc, 127, amax_in)
    i_int8 = q_out_dc

    # weight data quant 
    amax_wei_arr = []
    for i in range(weight_point2.shape[0]):
        amax_wei_arr.append(torch.abs(weight_point2[i]).max().item())
    # print(f"amax_wei1_arr :{amax_wei1_arr}")
    # print(f"torch.abs(weight1[0]).max().item():{torch.abs(weight1[i]).max().item()}")
    amax_wei_arr = torch.Tensor(amax_wei_arr)
    w_int8_arr = torch.zeros_like(weight_point2,device=device, dtype=torch.int8)
    for i in range(weight_point2.shape[0]):
        # w_int8_arr.append(quantize_to_int8(weight1[i], 127, amax_wei1_arr[i]))
        w_int8_arr[i] = quantize_to_int8(weight_point2[i], 127, amax_wei_arr[i])

    amax_out = torch.abs(out_pc2).max().item()
    out_pc2_int8 = quantize_to_int8(out_pc2, 127, amax_out)

    # alpha
    alpha_arr = []
    for i in range(weight_point2.shape[0]):
        # alpha_arr.append( 1 / ((amax_in * amax_wei1_arr[i]) / (127 * amax_out)))
        alpha_arr.append((amax_in * amax_wei_arr[i] ) / (127 * amax_out))
        #  alpha_arr.append(0)

    alpha_arr = torch.Tensor(alpha_arr).to(device)
    
    # bias
    bias1_point2 = (bias1_point2 *127) / amax_out
    i_int8 = i_int8.permute(0,2,3,1)
    w_int8_arr = w_int8_arr.permute(0,2,3,1)

    act_num = 5
    q_out_pc2 = conformer_infer_opt.test_conv(i_int8,i_int8.flatten(),w_int8_arr, w_int8_arr.flatten(),alpha_arr,bias1_point2,act_num,0)
    # shape nchw
    q_out_pc2 = q_out_pc2.permute(0,3,1,2).contiguous()
    diff = torch.abs(q_out_pc2.flatten() - out_pc2_int8.flatten())
    print(f"pointwise_conv2 diff max: {diff.max()}")


    # time iteration 100
    num_iter = 100
    torch.cuda.synchronize()
    start_time = time.time()
    for i in range(num_iter):
        q_out_pc2_ = conformer_infer_opt.test_conv(i_int8,i_int8.flatten(),w_int8_arr, w_int8_arr.flatten(),alpha_arr,bias1_point2,act_num,0)

    torch.cuda.synchronize()
    step3_time = time.time() - start_time

    
    return q_out_pc2,out_pc2_int8,step3_time


def test():
    batch_size = 24 # 24
    seq_len = 200 
    feat_dim = 80
    odim = 256
    torch.manual_seed(12)
    torch.cuda.manual_seed_all(12)
    
    inputs = torch.randn([batch_size,64,seq_len,feat_dim],dtype=torch.float32,device=device)

    point_conv1, weight_point1, bias1_point1, out_pc1, out_pc1_glu, depth_conv,weight_depth, bias_depth, out_dc, point_conv2, weight_point2, bias1_point2,out_pc2, pt_time = conv_Module(inputs)

    # pointwise_conv1 + glu and quant diff
    print("================================Step One: pointwise_conv1 + glu and quant diffs=============================     \n")
    # bug 入参信息
    q_out_glu,out_pc1_glu_int8,step1_time = quant_test_pc1_glu(inputs,out_pc1,out_pc1_glu, weight_point1, bias1_point1)

    # diff1 = torch.abs(q_out_glu.flatten() - out_pc1_glu_int8.flatten())
    # print(f" pointwise_conv1 + glu diff max: {diff1.max()}\n")

    #  depthwise_conv + silu and quant diff
    print("================================Step Two: depthwise_conv + silu and quant diffs=============================     \n")
    q_out_dc,out_dc_int8,step2_time = quant_test_dc_silu(out_pc1_glu, q_out_glu,out_dc,weight_depth, bias_depth)
    # diff2 = torch.abs(q_out_dc.flatten() - out_dc_int8.flatten())
    # print(f" depthwise_conv + silu diff max: {diff2.max()}\n")

    #  depthwise_conv2 and quant diff
    print("================================Step Three: pointwise_conv2 and quant diffs=============================     \n")
    q_out_pc2,out_pc2_int8,step3_time = quant_test_pc2(out_dc,q_out_dc,out_pc2,weight_point2, bias1_point2)
    # diff3 = torch.abs(q_out_pc2.flatten() - out_pc2_int8.flatten())
    # print(f"depthwise_conv2 diff max: {diff3.max()}\n")


    # time analyse
    num_iter = 100
    print("==========================================Time analyse =====================================================     \n")
    cuda_time = step1_time + step2_time + step3_time
    print(f'time pt_time {pt_time/num_iter} cuda_time {cuda_time/num_iter} pt_time/cuda_time {pt_time/cuda_time}')







if __name__ == "__main__":
    batch_size = 30 # 24
    seq_len = 200 
    feat_dim = 80
    odim = 256
    test()

