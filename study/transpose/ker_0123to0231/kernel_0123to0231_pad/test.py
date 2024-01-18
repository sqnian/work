import torch
import conformer_infer_opt

import os

os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = "cuda:0"




def test():
    print("run test op")
    input = torch.randn([24,1,710,80],device=device,dtype=torch.float16)
   
    
    input1= torch.zeros([24,3,710,80],dtype=torch.float16).cuda()
  

    output_pt = torch.cat((input,input1),dim=1)
    output_pt = output_pt.permute(0,2,3,1).contiguous()


    output_cu = conformer_infer_opt.test_func(input)  

    diff = output_pt.flatten() - output_cu.flatten()
    print(f"output_pt output_pt:{output_pt.flatten()[:50]}")
    print(f"output_pt shape:{output_pt.shape}")

    print(f"output_cu flatten():{output_cu.flatten()[:50]}")
    print(f"output_cu shape:{output_cu.shape}")



    print("diff max:\n",diff.max())
    
if __name__ == "__main__":
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    test()

