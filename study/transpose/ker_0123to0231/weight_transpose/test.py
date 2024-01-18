import torch
import conformer_infer_opt

import os

os.environ["CUDA_VISIBLE_DEVICES"]="1"
device = "cuda:0"


    
def run_two():
    print("run here\n")
    weight = torch.randn([256,256,3,3],device=device,dtype=torch.float16)
    weight_ = weight.permute(0,2,3,1).contiguous()

    output_cu = conformer_infer_opt.test_transpose(weight)

    diff_cu = output_cu.flatten() - weight_.flatten()


    print(f"diff_cu max:{diff_cu.max()}")

    print(f"output_cu :{output_cu.shape}\n")
    print(f"output_cu:{output_cu.flatten()[:100]}\n")

    print(f"weight_ :{weight_.shape}\n")
    print(f"weight_:{weight_.flatten()[:100]}\n")



if __name__ == "__main__":
    
    run_two()