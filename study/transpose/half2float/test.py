import torch
import conformer_infer_opt

import os

os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = "cuda:0"



def run():
    print("ok ")
    input_16 = torch.randn([256],device=device,dtype=torch.float16)

    input_32 = input_16.float()
    
    output_cu = conformer_infer_opt.test_trans(input_16,input_32)

    diff_cu = output_cu.flatten() - input_32.flatten()
    print(f"diff_cu max:{diff_cu.max()}")

    print(f"output_cu :{output_cu.shape}\n")
    print(f"output_cu:{output_cu.flatten()[:200]}\n")

    print(f"input_32 :{input_32.shape}\n")
    print(f"input_32:{input_32.flatten()[:200]}\n")
    

if __name__ == "__main__":
    
    run()