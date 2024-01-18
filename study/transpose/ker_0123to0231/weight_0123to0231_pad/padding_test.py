import torch 

import os 
os.environ["CUDA_DEVICE_VISIBLES"] = "0"
device = "cuda:0"


def run_two():
    print("ok ")
    input = torch.randn([256,1,3,3],dtype=torch.float16)

    output = input.permute(0,2,3,1).contiguous() # 256,3,3,1
    # print(f"input data permute :{output}")
    # print(f"input data permute shape:{output.shape}")

    

    output_p = torch.zeros([256,3,3,1],dtype=torch.float16,).flatten()

    # input 256,1,3,3  -> output_p 256,3,3,1
    for i in range(256):
        for j in range(3):
            for k in range(3): 
                for n in range(1):
                    output_p[3*3*i +3* j + k+n] = input.flatten()[ 3*3*i +3* j  + k+n] 

    # print(f"input data handle  :{output_p.reshape(256,3,3,1)}")

    diff = output_p.flatten() - output.flatten()
    print(f"diff max:{diff.max()}")

    print("================================ padding test ============================\n")
    output_ = torch.zeros([256,3,3,63],dtype=torch.float16)

    output_cat = torch.cat([output,output_],dim=3)
    # print(f"output_cat data :{output_cat}")
    print(f"output_cat data shape :{output_cat.shape}")

    output_pa = torch.zeros([256,3,3,64],dtype=torch.float16).flatten()
    for i in range(256):
        for j in range(3):
            for k in range(3): 
                for n in range(1):
                    output_pa[3*64*3*i + 64*3*j+ 64*k + n] = output.flatten()[ 3*3*i +3* j + k+n] 

    # print(f"input data handle padding :{output_pa.reshape(256,3,3,64)}")
    diff_ = output_cat.flatten() - output_pa.flatten()
    print(f"diff_ max:{diff_.max()}")



if __name__ == "__main__":
    run_two()