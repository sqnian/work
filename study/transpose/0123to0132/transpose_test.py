import torch

def run_two():
    print("ok ")
    input = torch.randn([24,17,19,56],dtype=torch.float16)  # shape: N H W C 

    print(f"input data  shape:{input.shape}")

    output_p = torch.zeros([24,17,19,56],dtype=torch.float16).flatten()
  
    for i in range(24):  # N 
        for j in range(17):  # H 
            for k in range(19):   # W 
                for n in range(56):  # C 
                    output_p[19*56*17*i + 19*56* j + 56*k + n] = input.flatten()[19*56*17*i + 19*56 *j + 56*k + n] 

    # print(f"input data handle  :{output_p.reshape(24,17,19,56)}")

    diff = output_p.flatten() - input.flatten()
    print(f"diff max:{diff.max()}")

    print("transpose nhwc to nhcw\n")
    input_t = input.permute(0,1,3,2).contiguous()

    output_t = torch.zeros([24,17,56,19],dtype=torch.float16).flatten()
    for i in range(24):  # N 
        for j in range(17):  # H 
            for k in range(19):   # W 
                for n in range(56):  # C 
                    output_t[19*56*17*i + 19*56* j + k + 19*n] = input.flatten()[19*56*17*i + 19*56 *j + 56*k + n] 

    diff_ = output_t.flatten() - input_t.flatten()
    print(f"diff_ max:{diff_.max()}")

if __name__ == "__main__":
    run_two()