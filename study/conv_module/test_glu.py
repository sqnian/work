import torch 
import os
import conformer_infer_opt
import time

os.environ["CUDA_VISIBLE_DEVICES"]= "0"
device = "cuda:0"

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




def test():
    # input = torch.randn(5,6,7,8,dtype=torch.float32,device = device)
    # input = torch.ones(5,6,7,8,dtype=torch.float32,device = device)
    input = torch.randn(5,6,7,8,dtype=torch.float32,device = device)

    # conv = torch.nn.Conv2d()


    output_pt = torch.nn.functional.glu(input,dim=1)

    amax_in = torch.abs(input).max().item()
    amax_out = torch.abs(output_pt).max().item()

    alpha = amax_in  / (127 * amax_out)

    print(f"alpha :{alpha}")

    i_int8 = quantize_to_int8(input,127,amax_in)
    out_int8 = quantize_to_int8(output_pt,127,amax_out)
    # print(f"i_int8 :{i_int8.flatten()[:50]}\n")
    # print(f"i_int8 : {i_int8.shape}")
    # print(f"out_int8 :{out_int8.flatten()[:50]}\n")
    # print(f"out_int8 : {out_int8.shape}")

    # exit(0)



    output_cu = torch.zeros_like(out_int8)

    output_cu_int8 = conformer_infer_opt.test_glu(output_cu,i_int8,amax_in,amax_out)

    print(f"out_int8 :{out_int8.flatten()[:50]}\n")
    print(f"out_int8 : {out_int8.shape}")
    print(f"output_cu_int8 : {output_cu_int8.flatten()[:50]}\n")
    print(f"output_cu_int8 : {output_cu_int8.shape}")


    diff = output_cu_int8.flatten() - out_int8.flatten()

    print(f"diff max : {diff.max()}\n")


    num_iter = 100
    torch.cuda.synchronize()
    start_time = time.time()
    for i in range(num_iter):
        output_pt = torch.nn.functional.glu(input,dim=1)

    torch.cuda.synchronize()
    pt_time = time.time() - start_time

    torch.cuda.synchronize()
    start_time = time.time()
    for i in range(num_iter):
         output_cu_int8 = conformer_infer_opt.test_glu(output_cu,i_int8,amax_in,amax_out)

    torch.cuda.synchronize()
    cuda_time = time.time() - start_time
    print(f'time pt_time {pt_time/num_iter} cuda_time {cuda_time/num_iter} pt_time/cuda_time {pt_time/cuda_time}')




if __name__ == "__main__":
    test()