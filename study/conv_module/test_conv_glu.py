import torch 
import os
import conformer_infer_opt


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
    

    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    input = torch.randn(5,64,7,8,dtype=torch.float32,device = device)

    conv =  torch.nn.Conv2d(64, 64, kernel_size=1,
            stride=2,
            padding=0,bias=True).to(device)

    output_ = conv(input)


    output_pt = torch.nn.functional.glu(output_,dim=1)

    amax_in = torch.abs(output_).max().item()
    amax_out = torch.abs(output_pt).max().item()
    print("amax_in:\n",amax_in)
    print("amax_out:\n",amax_out)
    alpha = amax_in  / (127 * amax_out)

    print(f"alpha :{alpha}")

    i_int8 = quantize_to_int8(output_,127,amax_in)
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
 

if __name__ == "__main__":
    test()