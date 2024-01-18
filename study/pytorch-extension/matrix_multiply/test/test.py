import torch
import cublas_gemm_test
import os
import time
import copy

torch.cuda.manual_seed(1234)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = "cuda:0"

# batch_size = 1
# batch_seq_len = 9 # 126
# hidden_size = 1024

# def one_test(batch_size, batch_seq_len, hidden_size):  # 1 , 9 ,1024 第一次循环
    # m = hidden_size * 3  # 1024 X 3 = 3072
    # n = batch_size * batch_seq_len  # 1 X 9 = 9
    # k = hidden_size # 1024

    # alpha = 1
    # beta = 0

    # # batch_size * batch_seq_len, hidden_size
    # q = torch.randn([n, k], dtype=torch.float32, device=device)  # 9 X 1024 矩阵大小
    # # hidden_size, hidden_size * 3
    # weights = torch.randn([k, m], dtype=torch.float32, device=device) # 1024 X 3072
    # # batch_size * batch_seq_len, hidden_size * 3
    # outputs = torch.randn([n, m], dtype=torch.float32, device=device) # 9 X 3072

    # A = weights.clone() # 1024 X 3072
    # B = q.clone()       # 9 X 1024
    # C = outputs.clone()   # 9 X 3072

    # print(f"A:{A}")
    # print(f"A.shape:{A.shape}")
    # print(f"B:{B}")
    # print(f"B.shape:{B.shape}")

    # # 利用 AB = (BT*AT)T
    # res1 = cublas_gemm_test.test_fp32(
    #     A.clone(), B.clone(), C.clone())

    # res2 = torch.addmm(outputs, q, weights, beta=beta, alpha=alpha)

    # diff = torch.abs(res1-res2)
    # nn_max_diff = diff.max()
    # print(f"check with torch.addmm max diff: {diff.max()}")
def test_exam_fp32():
    A = torch.tensor([[1,1,1],[1,1,1]],dtype=torch.float32,device=device)
    print(f"A:{A}")
    print(f"A.shape:{A.shape}")
    print(f"A.type:{A.dtype}")

    # B = torch.tensor([[1,2,3,4],[5,6,7,8],[9,10,11,12]],dtype=torch.float32,device=device)
    # B = torch.tensor([[ 0.0157, -0.1623,  0.1781,  0.5442],
    #     [ 0.1082, -0.2898,  0.3945, -0.3555],
    #     [ 0.1955,  0.3740,  0.2431, -0.3191]],dtype=torch.float32,device=device)

    B = torch.tensor( [[0.0157,  0.1082,  0.1955],
        [-0.1623, -0.2898,  0.3740],
        [ 0.1781,  0.3945,  0.2431],
        [ 0.5442, -0.3555, -0.3191]],dtype=torch.float32,device=device)

    # print(f"B:{B}")
    # print(f"B.shape:{B.shape}")
    # print(f"B.type:{B.dtype}")

    # C = A.mm(B)
    # print(f"C:{C}")
    # print(f"C.shape:{C.shape}")
    # print(f"C.type:{C.dtype}")

    print("======================================================\n")
    torch.cuda.synchronize()
    start_time = time.time()

    # print(f"B.T:{B.T}")
    # print(f"B.T.shape:{B.T.shape}")
    # print(f"B.T.type:{B.T.dtype}")

    B_b = B.T
    print(f"B_b:{B_b}")
    print(f"B_b.type:{B_b.dtype}")


    C_c = A.mm(B_b)
    print(f"C_c:{C_c}")
    # D = cublas_gemm_test.test_fp32(A,B_b)
    D = cublas_gemm_test.test_fp32(A,B)


    torch.cuda.synchronize()
    end_time = time.time()
    print("time:",end_time - start_time)

    print(f"D:{D}")

def linear_compare_cpp_fp32():
    m = 24
    n =256
    k = 4034

    input = torch.ones(m,n,device=device,dtype=torch.float32)
    print(f"input:{input}")

    layer = torch.nn.Linear(n,k,bias=False)
    layer.to(device)
    print("layer.weight.data:",layer.weight.data)
    
    res_lin = layer(input)
    print(f"res_lin:{res_lin}")

    res_cpp = cublas_gemm_test.test_fp32(input,layer.weight.data)
    print(f"res_cpp:{res_cpp}")
    
    diff = torch.abs(res_lin - res_cpp)
    print(f"diff:{diff}")

    torch.cuda.synchronize()
    start_time = time.time()
    for i in range(20):
        res_lin = layer(input)
    torch.cuda.synchronize()
    end_time = time.time()
    print(f"Linear time:{end_time}")

    torch.cuda.synchronize()
    start_time = time.time()
    for i in range(20):
        res_cpp = cublas_gemm_test.test_fp32(input,layer.weight.data)
    torch.cuda.synchronize()
    end_time = time.time()
    print(f"cpp time:{end_time}")

def linear_compare_cpp_fp16():
    # m = 2
    # n = 4
    # k = 3

    print("running =============")

    input = torch.randn(2,3,device=device,dtype=torch.float16)
    print(f"input:{input}")

    layer = torch.nn.Linear(3,5,bias=False).half()
    layer.to(device)
    print("layer.weight.data:",layer.weight.data)

    print(f"mm():{input.mm(layer.weight.data.T)}")
    
    res_lin = layer(input)
    print(f"res_lin:{res_lin}")

    res_cpp = cublas_gemm_test.test_fp16(input,layer.weight.data)
    print(f"res_cpp:{res_cpp}")
    
    diff = torch.abs(res_lin - res_cpp)
    print(f"diff:{diff}")

    torch.cuda.synchronize()
    start_time = time.time()
    for i in range(20):
        res_lin = layer(input)
    torch.cuda.synchronize()
    end_time = time.time()
    print(f"Linear time:{end_time}")

    torch.cuda.synchronize()
    start_time = time.time()
    for i in range(20):
        res_cpp = cublas_gemm_test.test_fp16(input,layer.weight.data)
    torch.cuda.synchronize()
    end_time = time.time()
    print(f"cpp time:{end_time}")
    
def linear_compare_cpp_fp16_at():
    m = 2
    n = 4
    k = 3

    input = torch.ones(2,3,device=device,dtype=torch.float16)
    print(f"input:{input}")

    layer = torch.nn.Linear(3,2,bias=False).half()
    layer.to(device)
    print("layer.weight.data:",layer.weight.data)
    
    res_lin = layer(input)
    print(f"res_lin:{res_lin}")

    res_cpp = cublas_gemm_test.test_fp16_at(layer.weight.data,input)
    print(f"res_cpp:{res_cpp}")
    
    diff = torch.abs(res_lin - res_cpp)
    print(f"diff:{diff}")

    torch.cuda.synchronize()
    start_time = time.time()
    for i in range(20):
        res_lin = layer(input)
    torch.cuda.synchronize()
    end_time = time.time()
    print(f"Linear time:{end_time}")

    torch.cuda.synchronize()
    start_time = time.time()
    for i in range(20):
        res_cpp = cublas_gemm_test.test_fp16_at(layer.weight.data,input)
    torch.cuda.synchronize()
    end_time = time.time()
    print(f"cpp time:{end_time}")

def linear_compare_cpp_fp16_ori():
    batch_size = 24
    hidden_size = 256
    vocab_size = 4034
    layer = torch.nn.Linear(hidden_size, vocab_size, bias=False).half()
    layer.to(device)

    inputs = torch.randn([batch_size,hidden_size],device=device,dtype=torch.float16)
    print(f"input:{inputs}\n")
    print(f"layer.weight.data:{layer.weight.data}\n")

    pt_res = layer(inputs)
    print(f"pt_res:{pt_res}\n")
    
    cuda_res = cublas_gemm_test.test_fp16_at(layer.weight.data,inputs)
    print(f"cuda_res:{cuda_res}\n")

    diff = torch.abs(pt_res - cuda_res)
    print(f"diff max: {diff.max()}")

    torch.cuda.synchronize()
    t1 = time.time()
    for i in range(20):
        pt_res = layer(inputs)
    torch.cuda.synchronize()
    pt_time = time.time() - t1

    t1 = time.time()
    for i in range(20):
        cuda_res = cublas_gemm_test.test_fp16_at(layer.weight.data,inputs)
    torch.cuda.synchronize()
    cuda_time = time.time() - t1

    print(f"pt time: {pt_time}, cuda time: {cuda_time}, rate: {pt_time/cuda_time}")
if __name__ == '__main__':
    # test_exam_fp32()
    # one_test()
    linear_compare_cpp_fp32()

    linear_compare_cpp_fp16()
    # linear_compare_cpp_fp16_at()
    


    


