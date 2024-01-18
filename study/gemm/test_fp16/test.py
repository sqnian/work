import torch 
import os
os.environ['CUDA_VISIBLE_DEVICES']="5"
import gemm_test

def test():
    print('run here\n')
    input =torch.randn([1,3136,148],dtype=torch.float16).cuda()
    weight = torch.randn([148,192],dtype=torch.float16).cuda()
    # input =torch.randn([1,3136,147],dtype=torch.float16).cuda()
    # weight = torch.randn([147,192],dtype=torch.float16).cuda()
    output = torch.matmul(input ,weight)
    print('output shape:',output.shape)
    # print('output: ', output)
 
    out_test = gemm_test.test_gemm(input,weight.T.contiguous())
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