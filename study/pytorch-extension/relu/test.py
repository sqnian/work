import torch
import conv_relu
import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = "cuda:0"


def conv2d_relu_single():

    batch_size = 1
    seq_len = 710
    feat_dim = 80

    odim = 256
    conv = torch.nn.Conv2d(1, odim, 3, 2).half()
    conv.to(device)

    print(conv.weight.shape)
    print(conv.bias.shape)

    # conv.bias.data = torch.zeros_like(conv.bias.data)
    # print(f"conv.bias.data:{conv.bias.data}\n")

    weight = conv.weight.data.clone()
    bias = conv.bias.data.clone()

    inputs = torch.randn([batch_size, 1, seq_len, feat_dim],
                         device=device, dtype=torch.float16)
    print("inputs:{}".format(inputs))

    res_pt = conv(inputs)
    res_pt = torch.nn.functional.relu(res_pt)
    # bias
    # print(f"bias:{bias}")

    res_cu = conv_relu.conv2d_relu(inputs, weight,bias)

    # print(res_pt.shape)
    # print(res_cu.shape)
    diff = torch.abs(res_pt-res_cu)
    print(f"diff max: {diff.max()}")

    # print(res_pt.flatten()[:100])
    # print(res_cu.flatten()[:100])


    #  test time
    torch.cuda.synchronize()
    t1 = time.time()
    for i in range(20):
        res_pt = conv(inputs)
        res_pt = torch.nn.functional.relu(res_pt)
    torch.cuda.synchronize()
    pt_time = time.time() - t1

    t1 = time.time()
    for i in range(20):
        cuda_res = conv_relu.conv2d_relu(inputs, weight,bias)
    torch.cuda.synchronize()
    cuda_time = time.time() - t1

    print(f"pt time: {pt_time}, cuda time: {cuda_time}, rate: {pt_time/cuda_time}")


def generate_data(batch_size, seq_len, feat_dim, odim):
    inputs = torch.randn([batch_size,1,seq_len,feat_dim],device=device,dtype=torch.float16)
    conv = torch.nn.Sequential(
                torch.nn.Conv2d(1,odim,3,2),
                torch.nn.ReLU(),
                torch.nn.Conv2d(odim,odim,5,3),
                torch.nn.ReLU()
            ).half().to(device)

    weight1 = conv[0].weight.data.clone()
    bias1 = conv[0].bias.data.clone()
    weight2 = conv[2].weight.data.clone()
    bias2 =conv[2].bias.data.clone()

    return inputs, conv, weight1, bias1, weight2, bias2

def torch_test(inputs,conv):
    res = conv(inputs)
    res = res.transpose(1,2)
    return res

def conv2d_relu_two():
    batch_size = 24
    seq_len = 710
    feat_dim = 80

    odim = 256

    inputs, conv, weight1, bias1, weight2,bias2 = generate_data(batch_size, seq_len, feat_dim, odim)

    res_pt = torch_test(inputs,conv)

    res_cu = conv_relu.conv2d(inputs, weight1, bias1, weight2,bias2)
    total_length = res_pt.flatten().size(0)
    res_cu = res_cu.flatten()[:total_length].view(*res_pt.shape)

    print("parameters shape: \n")
    print(f"weight1 shape :{weight1.shape}")
    print(f"bias1 shape :{bias1.shape}")
    print(f"weight2 shape :{weight2.shape}")
    print(f"bias2 shape :{bias2.shape}")
    print(f"res_pt:{res_pt.shape}")
    print(f"res_cu:{res_cu.shape}")

    diff = torch.abs(res_pt - res_cu)

    print(f"diff.max : {diff.max()}")


    #  test time
    torch.cuda.synchronize()
    t1 = time.time()
    for i in range(20):
        res_pt = torch_test(inputs,conv)
    torch.cuda.synchronize()
    pt_time = time.time() - t1

    t1 = time.time()
    for i in range(20):
        res_cu = conv_relu.conv2d(inputs, weight1, bias1, weight2,bias2)
        total_length = res_pt.flatten().size(0)
        res_cu = res_cu.flatten()[:total_length].view(*res_pt.shape)
    torch.cuda.synchronize()
    cuda_time = time.time() - t1

    print(f"pt time: {pt_time}, cuda time: {cuda_time}, rate: {pt_time/cuda_time}")





if __name__ == "__main__":

    # 单个 conv + relu 激活函数
    # conv2d_relu_single()

    # 两层 conv +  relu
    conv2d_relu_two()
