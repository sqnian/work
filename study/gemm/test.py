import torch 
import os
os.environ['CUDA_VISIBLE_DEVICES']="5"
import gemm_test

def test():
    print('run here\n')
    input =torch.randn([64,196,576],dtype=torch.float16).cuda()
    linear = torch.nn.Linear(576,384).half().cuda()
    weight = linear.weight.data.clone()
    bias = linear.bias.data.clone()
    gelu = torch.nn.GELU()
    output = gelu(linear(input))
    # output = linear(input)

    out_test = gemm_test.test_gemm(input,weight,bias)
    diff_test = torch.abs(output.flatten() - out_test.flatten())
    print('diff test max: ',diff_test.max())
    print('output shape:',output.shape)
    print('out_test shape:',out_test.shape)
    print('output flatten [-100:]:',output.flatten()[-100:])
    print('out_test flatten [-100:]:',out_test.flatten()[-100:])


if __name__ == '__main__':
    torch.manual_seed(133)
    torch.cuda.manual_seed_all(112)
    test()