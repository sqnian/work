import torch 

import os 
os.environ["CUDA_DEVICE_VISIBLES"] = "0"
device = "cuda:0"


def run_two():
    print("ok ")
    input = torch.randn([256,256,3,3],dtype=torch.float16)

    output = input.permute(0,2,3,1).contiguous() # 256,3,3,256
    # print(f"input data permute :{output}")
    print(f"input data permute shape:{output.shape}")

    

    # output_p = torch.zeros([256,3,3,256],dtype=torch.float16,).flatten()

    # # input 256,1,3,3  -> output_p 256,3,3,1
    # for i in range(256):
    #     for j in range(256):
    #         for k in range(3): 
    #             for n in range(3):
    #                 output_p[3*3*256*i + 3* 3*j  + 3 * k + n] = input.flatten()[ 3*3*256*i + 3* 3*j  + 3 * k + n] 


    # diff = output_p.flatten() - input.flatten()
    # print(f"diff max:{diff.max()}")

    print("================================ transpose test ============================\n")

    output_pa = torch.zeros([256,3,3,256],dtype=torch.float16).flatten()
    for i in range(256):
        for j in range(256):
            for k in range(3): 
                for n in range(3): 
                    output_pa[3*256*3*i + j + 3*256*k + 256*n] = input.flatten()[ 3*3*256*i + 3* 3*j  + 3 * k + n] 

    # print(f"input data handle padding :{output_pa.reshape(256,3,3,64)}")
    diff_ = output_pa.flatten() - output.flatten()
    print(f"diff_ max:{diff_.max()}")



if __name__ == "__main__":
    run_two()

    # input1 = torch.arange(36).reshape(2,2,3,3)
    # input1_ = input1.permute(0,2,3,1).contiguous()

    # print(f"input1:{input1.flatten()}\n")
    # print(f"input1_:{input1_.flatten()}\n")