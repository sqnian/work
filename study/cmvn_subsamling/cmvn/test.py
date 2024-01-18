import torch
import conformer_infer_opt

import os

os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = "cuda:0"

def cmvn(input,mean,istd):
    output = input - mean
    output = output * istd
    return output


def test_data(input,mean,istd):
    print("  validation data ok  ")
    output = torch.load("./cmvn_output.pt")
    print("input shape:\n",input.shape)
    print("mean shape:\n",mean.shape)
    print("istd shape:\n",istd.shape)

    output_pt = cmvn(input,mean,istd)
    diff = output_pt.flatten() - output.flatten()
    print("diff max:\n",diff.max())


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

def test(input,mean,istd):
    print("run test op")
    # input = torch.randn([24,710,80],device=device,dtype=torch.float16)
    fp16 = False # True
    if not fp16:
        input = input.float()
        mean = mean.float()
        istd = istd.float()
    output_pt = cmvn(input,mean,istd)
    amax_out = torch.abs(output_pt).max().item()
    amax_in =  torch.abs(input).max().item()
    # print(f"amax_out :{type(amax_out)}")  # float
    # print(f"amax_out :{(amax_out)}") 
    # print(f"amax_out :{ 127 / (amax_out)}") 
    # exit(0)
    
    output_pt_int8  = quantize_to_int8(output_pt,127,amax_out)
    input_int8  = quantize_to_int8(input,127,amax_in)

    output_cu = conformer_infer_opt.test_cmvn(input,input_int8,mean,istd,amax_out,fp16)  # False

    diff = output_pt_int8.flatten() - output_cu.flatten()
    # diff = output_pt.flatten() - output_cu.flatten()
    print(f"output_pt_int8 flatten:{output_pt_int8.flatten()[:50]}")
    print(f"output_pt_int8 shape:{output_pt_int8.shape}")

    print(f"output_cu flatten():{output_cu.flatten()[:50]}")
    print(f"output_cu shape:{output_cu.shape}")

    # print(f"output_pt flatten:{output_pt.flatten()[:50]}")
    # print(f"output_pt shape:{output_pt.shape}")


    print("diff max:\n",diff.max())
    
if __name__ == "__main__":
    # torch.manual_seed(123)
    # torch.cuda.manual_seed_all(123)
    input = torch.load("./cmvn_input.pt")
    mean = torch.load("./cmvn_mean.pt")
    istd = torch.load("./cmvn_istd.pt")

    # test_data(input,mean,istd)

    test(input,mean,istd)
