import torch
import conformer_infer_opt

import os

os.environ["CUDA_VISIBLE_DEVICES"]="1"
device = "cuda:0"


    
def run_two():
    print("run here\n")
    input = torch.randn([24,1487,80,256],device=device,dtype=torch.float16) # shape :NHWC
    input_ = input.permute(0,1,3,2).contiguous()  # shape :NHCW

    output_cu = conformer_infer_opt.test_transpose(input)

    diff_cu = output_cu.flatten() - input_.flatten()


    print(f"diff_cu max:{diff_cu.max()}")

    print(f"output_cu :{output_cu.shape}\n")
    print(f"output_cu:{output_cu.flatten()[:100]}\n")

    print(f"input_ :{input_.shape}\n")
    print(f"input_:{input_.flatten()[:100]}\n")



if __name__ == "__main__":
    
    run_two()