import torch
import conformer_infer_opt
import copy

import os 
os.environ["CUDA_VISIBLE_DEVICES"]= "1"
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
    print("transpose\n")

    # input shape :NHWC 
    # input = torch.randn([24,176,19,256],dtype=torch.float16,device=device)
    input = torch.randn([24,17,19,25],dtype=torch.float16,device=device)

    # NHWC 0123---> NHCW 0132
    output = input.permute(0,1,3,2).contiguous()
   
    amax_out = torch.abs(output).max().item()
    output_int8  = quantize_to_int8(output,127,amax_out)
    output_int8_d = copy.deepcopy(output_int8)
    print(f"output_int8 shape:{output_int8.shape}\n")
    print(f"output_int8_d:{output_int8_d.flatten()[:100]}\n")
    # print(f"output_int8:{output_int8.flatten()[-100:]}\n")

    # input int 8 
    amax_in = torch.abs(input).max().item()
    input_int8  = quantize_to_int8(input,127,amax_in)

    # print(f"input_int8 shape:{input_int8.shape}\n")  #  shape N H W C
    # print(f"input_int8:{input_int8.flatten()[-100:]}\n")
    # exit(0)

    # test transpose kernel
    # check int8 
    out_cu = conformer_infer_opt.test_transpose(input_int8,input_int8)

    print(f"output_int8 shape:{output_int8.shape}\n")
    print(f"output_int8_d:{output_int8_d.flatten()[:100]}\n")
    # # print(f"output_int8:{output_int8.flatten()[-100:]}\n")
    # print(f"input_int8: {input_int8.flatten()[:100]}\n")

    # print(f"out_cu:{out_cu.flatten()[-100:]}\n")
    print(f"out_cu :{out_cu.shape}\n")
    print(f"out_cu:{out_cu.flatten()[:100]}\n")


    print("last 100 \n")

    print(f"output_int8:{output_int8.flatten()[-100:]}\n")

    print(f"out_cu:{out_cu.flatten()[-100:]}\n")

    # print(f"out_cu shape:{out_cu.shape}\n")

    diff = torch.abs(output_int8.flatten()-out_cu.flatten())

    print(f"diff max:{diff.max()}\n")

def test_two():
    print("test index is ok \n")
    input = torch.randn([24,17,19,25],dtype=torch.float16,device=device)

    output = input.permute(0,1,3,2).contiguous()
   
    amax_out = torch.abs(output).max().item()
    output_int8  = quantize_to_int8(output,127,amax_out)


    out_cu = conformer_infer_opt.test_transpose(output_int8,output_int8)

    print(f"output_int8 shape:{output_int8.shape}\n")
    print(f"output_int8:{output_int8.flatten()[:100]}\n")
    # print(f"output_int8:{output_int8.flatten()[-100:]}\n")

    # print(f"out_cu:{out_cu.flatten()[-100:]}\n")
    print(f"out_cu :{out_cu.shape}\n")
    print(f"out_cu:{out_cu.flatten()[:100]}\n")

    diff = torch.abs(output_int8.flatten()-out_cu.flatten())
    print(f"diff max:{diff.max()}\n")


def test_3():
    print("transpose\n")

    # input shape :NHWC 
    # input = torch.randn([24,176,19,256],dtype=torch.float16,device=device)*100
    # input = torch.randn([24,17,19,25],dtype=torch.float16,device=device)*100
    input = torch.randint(-127,127,[24,17,19,25],dtype=torch.float16,device=device)


    # NHWC 0123---> NHCW 0132
    input_transpose = input.permute(0,1,3,2).contiguous()

    # input int 8 
    amax_in = torch.abs(input).max().item()
    input_int8  = quantize_to_int8(input,127,amax_in)


    # test transpose kernel
    # check int8 - > half 
    out_cu = conformer_infer_opt.test_transpose(input_int8,input,amax_in)

    print(f"input_transpose shape:{input_transpose.shape}\n")
    print(f"input_transpose:{input_transpose.flatten()[:200]}\n")

    # print(f"out_cu:{out_cu.flatten()[-100:]}\n")
    print(f"out_cu :{out_cu.shape}\n")
    print(f"out_cu:{out_cu.flatten()[:200]}\n")


    print("last 100 \n")
    print(f"input_transpose:{input_transpose.flatten()[-100:]}\n")
    print(f"out_cu:{out_cu.flatten()[-100:]}\n")

    diff = torch.abs(input_transpose.flatten()-out_cu.flatten())
    print(f"diff max:{diff.max()}\n")

    print(f":{sum(diff > 10)}")


if __name__ ==  "__main__":
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    # test()
    # test_two()
    test_3()