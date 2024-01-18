"""
test
cuinferHalfConvolution2dForward fp16 api function
eg: conv2d + relu 
comformer subsampling module
"""

import torch
import conformer_infer_opt 
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
device = "cuda:0"


def get_data():

    input = torch.randn([24,2, 710, 80],device=device,dtype=torch.float16)

    conv = torch.nn.Sequential(
            torch.nn.Conv2d(2,256,3,2), 
            torch.nn.ReLU(),
            torch.nn.Conv2d(256,256,3,2), 
            torch.nn.ReLU()
    ).half().to(device)

    weight1 = conv[0].weight.data.clone()
    bias1 = conv[0].bias.data.clone()

    weight2 = conv[2].weight.data.clone()
    bias2 = conv[2].bias.data.clone()

    return input, conv, weight1, bias1,weight2, bias2



def torch_impl(input, conv):
    output = conv(input)
    return output

def test():

    input, conv, weight1, bias1,weight2, bias2 = get_data()

    output_pt = torch_impl(input, conv)
    output_pt = output_pt.permute(0,2,3,1).contiguous()

    print("test api function\n")  # float test
    input = input.permute(0,2,3,1).contiguous()
    weight1 = weight1.permute(0,2,3,1).contiguous()
    weight2 = weight2.permute(0,2,3,1).contiguous()


    # output_cu = conformer_infer_opt.test_conv(input, weight1, bias1.float(), weight2, bias2.float())
    output_cu = conformer_infer_opt.test_func(input, weight1, bias1.float(), weight2, bias2.float())
    # output_cu = output_cu.permute(0,3,1,2).contiguous()
    

    print(f"output_pt:{output_pt.flatten()[:100]}\n")
    print(f"output_pt shape:{output_pt.shape}")

    print(f"output_cu:{output_cu.flatten()[:100]}\n")
    print(f"output_cu :{output_cu.shape}\n")

    diff = torch.abs(output_pt.flatten()- output_cu.flatten())

    print(f"diff max:{diff.max()}\n")


    

if __name__ == "__main__":

    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)

    test() 
