import torch
import transpose_test
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = "5"

def test():
    input = torch.randn([1,3,7,56,7,56],dtype=torch.float16).cuda()
    output = input.permute(0,1,2,4,3,5).contiguous()

    out_test = transpose_test.test_transpose(input)
    
    diff_test = torch.abs(output.flatten() - out_test.flatten())
    print('diff test max: ',diff_test.max())
    print('output shape:',output.shape)
    print('out_test shape:',out_test.shape)

if __name__ == '__main__':
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    test()