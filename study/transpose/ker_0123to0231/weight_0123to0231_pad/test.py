import torch
import conformer_infer_opt

import os

os.environ["CUDA_VISIBLE_DEVICES"]="1"
device = "cuda:0"



def run():
    print("ok ")
    weight1 = torch.randn([256,1,3,3],device=device,dtype=torch.float16)
    # weight_1 = weight1.permute(0,2,3,1).contiguous()

    weight1_ = torch.zeros([256,3,3,3],device=device,dtype=torch.float16)

    weight = torch.cat([weight1, weight1_], dim=1)

    weight = weight.permute(0,2,3,1).contiguous()

    
    
    output_cu = conformer_infer_opt.test_trans_pad(weight1)

    diff_cu = output_cu.flatten() - weight.flatten()
    print(f"diff_cu max:{diff_cu.max()}")

    print(f"output_cu :{output_cu.shape}\n")
    print(f"output_cu:{output_cu.flatten()[:100]}\n")

    print(f"weight :{weight.shape}\n")
    print(f"weight:{weight.flatten()[:100]}\n")
    




if __name__ == "__main__":
    
    run()
    # run_two()