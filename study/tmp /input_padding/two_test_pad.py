#_*_ coding: UTF-8 _*_
import torch
import conformer_infer_opt 
import os
import  time

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
device = "cuda:0"

def gen_conv1():
    conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, 256, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, 3, 2),
            torch.nn.ReLU()
        ).to(device).half()
   
    weight1 = conv[0].weight.data 
    bias1 = conv[0].bias.data
    
    weight2 = conv[2].weight.data  
    bias2 =  conv[2].bias.data 

 
    return conv,weight1, bias1, weight2, bias2

def torch_imply(inputs, conv):
    y = conv(inputs)
    # b, c, t, f = y.size()
    # # print(y.size())
    # y = y.transpose(1, 2).contiguous().view(b, t, c * f)
    return y
 
def run():
    """
    test op 
    """
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)

    inputs = torch.randn([24, 1, 710, 80], device=device, dtype=torch.float16)
    conv1,weight1, bias1, weight2, bias2 = gen_conv1()
    output_pt = torch_imply(inputs, conv1)
    output_pt = output_pt.permute(0,2,3,1).contiguous()

    print("test op \n")
    weight2 = weight2.permute(0,2,3,1).contiguous()

    output_cu = conformer_infer_opt.test_func_padding(inputs, weight1, bias1.float(), weight2, bias2.float()) # 先全部输出，验证精度，再调整
    # output_cu = output_cu.permute(0,3,1,2).contiguous()
    # output_cu = output_cu.reshape(32,710,80,32)

    print(f"output_pt:{output_pt.flatten()[-100:]}\n")
    print(f"output_pt shape:{output_pt.shape}")
    print(f"output_cu:{output_cu.flatten()[-100:]}\n")
    print(f"output_cu :{output_cu.shape}\n")

    diff = torch.abs(output_pt.flatten()- output_cu.flatten()[:])
    print(f"diff max:{diff.max()}\n")



if __name__ == "__main__":
    # test() 
    run()
