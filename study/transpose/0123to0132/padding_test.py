import torch 

import os 
os.environ["CUDA_DEVICE_VISIBLES"] = "0"
device = "cuda:0"


def test_two_dimension():
    input = torch.arange(15).reshape(3,5)
    print(f"input data :{input}\n") 
    
    output = input.permute(1,0)
    print(f"input permute transpose: {output}\n")

    output1 = input.reshape(5,3)
    print(f"input reshape transpose: {output1}\n")  # reshape transpose 不一样
    
    output_hand = torch.zeros([5,3]).flatten()
    input = input.flatten()
    for i in range(3):
        for j in range(5):
            output_hand[i+3*j] = input[5*i+j]
    print(f"input hand transpose: {output_hand.reshape(5,3)}\n")

def test_reshape():
    print("ok ")
    input = torch.arange(12).reshape(2,3,2)
    print("input:\n",input)

    # flatten
    input_1 = input.flatten()
    output_ = input_1.reshape(2,2,3)
    print("input reshape(2,2,3):\n",output_)

    output = torch.zeros([2,2,4]).flatten()
    print(f"output shape(2,2,4) flatten():{output}\n")

    for i in range(2):
        for j in range(2):
            for k in range(3): 
                output[8*i+4*j + k] = input_1[6*i + 3*j + k] 

    print(f"output shape(2,2,4) flatten() :{output}\n")
    output = output.reshape(2,2,4)
    print(f"output:{output}\n")

    print("reshape and padding\n")
    input = input.reshape(2,2,3)
    input_p = torch.zeros([2,2,1])
    output_padding = torch.cat([input,input_p],dim=2)
    print(f"output_padding:{output_padding}")

    diff = output.flatten() - output_padding.flatten()
    print(f"diff max:{diff.max()}")


def test_transpose():
    print("test transpose in term of one dimension\n")

    input = torch.arange(12).reshape(2,3,2)
    print(f"input data :{input}")
    print(f"input data shape:{input.shape}")

    output = input.transpose(2,1).contiguous() # 2,2,3
    print(f"input data transpose :{output}")
    print(f"input data transpose shape:{output.shape}")

    output_p = torch.zeros(2*2*3)

    for i in range(2):
        for j in range(3):
            for k in range(2):
                output_p[6*i + j + 3*k] = input.flatten()[6*i+2*j+k]
    print(f"input data handle transpose :{output_p.reshape(2,2,3)}") # for  loop test 

    print("================================ padding test ============================\n")
    """
    1、先转置
    2、再padding
    """
    output_ = torch.zeros([2,2,1])
    output_cat = torch.cat([output,output_],dim=2)
    print(f"input data padding :{output_cat}")
    print(f"input data padding shape:{output_cat.shape}")

    output_pa = torch.zeros([2,2,4]).flatten()
    # for i in range(2):
    #     for j in range(2):
    #         for k in range(3): 
    #             output_pa[8*i+4*j + k] = output.flatten()[6*i + 3*j + k] 
    for i in range(2):
        for j in range(3):
            for k in range(2): 
                output_pa[8*i+4*j + k] = output.flatten()[6*i + 3*j + k] 
    print(f"input data handle padding :{output_pa.reshape(2,2,4)}")


def test_four_dimension():
    print("ok ")
    input = torch.arange(12).reshape(2,1,3,2)
    print(f"input data :{input}")
    print(f"input data shape:{input.shape}")

    output = input.permute(0,3,2,1).contiguous()
    print(f"input data permute :{output}")
    print(f"input data permute shape:{output.shape}")

    output_p = torch.zeros([2,2,3,1]).flatten()
    for i in range(2):
        for j in range(1):
            for k in range(3): 
                for n in range(2):
                    output_p[6*i+j+ k + 3*n] = input.flatten()[6*i +j + 2*k+n] 
    print(f"input data handle padding :{output_p.reshape(2,2,3,1)}")

    print("================================ padding test ============================\n")
    """
    1、先转置
    2、再padding
    """
    output_ = torch.zeros([2,2,3,1])
    output_cat = torch.cat([output,output_],dim=3)
    print(f"input data padding :{output_cat}")
    print(f"input data padding shape:{output_cat.shape}")

    output_pa = torch.zeros([2,2,3,2]).flatten()
    for i in range(2):
        for j in range(2):
            for k in range(3): 
                    output_pa[12*i+6*j+ 2*k] = output.flatten()[6*i+3*j+k] 

    print(f"input data handle padding :{output_pa.reshape(2,2,3,2)}")

   
def test_four_dimension_():
    print("ok ")
    input = torch.arange(12).reshape(2,1,3,2)
    print(f"input data :{input}")
    print(f"input data shape:{input.shape}")

    output = input.permute(0,2,3,1).contiguous()
    print(f"input data permute :{output}")
    print(f"input data permute shape:{output.shape}")

    output_p = torch.zeros([2,3,2,1]).flatten()
    # for i in range(2):
    #     for j in range(1):
    #         for k in range(3): 
    #             for n in range(2):
    #                 output_p[6*i +j + 2*k+n] = input.flatten()[6*i +j + 2*k+n] 

    for i in range(2):
        for j in range(3):
            for k in range(2): 
                for n in range(1):
                    output_p[6*i + 2*j +k+n] = input.flatten()[ 6*i + 2*j +k+n] 
    print(f"input data handle :{output_p.reshape(2,3,2,1)}")

    print("================================ padding test ============================\n")
    """
    1、先转置
    2、再padding
    """
    output_ = torch.zeros([2,3,2,1])
    output_cat = torch.cat([output,output_],dim=3)
    print(f"input data padding :{output_cat}")
    print(f"input data padding shape:{output_cat.shape}")

    output_pa = torch.zeros([2,3,2,2]).flatten()
    for i in range(2):
        for j in range(3):
            for k in range(2): 
                for n in range(1):
                    output_pa[12*i + 4*j+ 2*k + n] = output.flatten()[6*i + 2*j+ k + n ] 

    print(f"input data handle padding :{output_pa.reshape(2,3,2,2)}")


def run():
    print("ok ")
    input = torch.randn([21,1,80,710],dtype=torch.float16)

    output = input.permute(0,2,3,1).contiguous() # 21,80,710,1
    

    output_p = torch.zeros([21,80,710,1],dtype=torch.float16,).flatten()
    # for i in range(24):
    #     for j in range(1):
    #         for k in range(80): 
    #             for n in range(710):
    #                 output_p[80*710*i + j + 710*k+n] = input.flatten()[80*710*i + j + 710*k+n] 

    for i in range(21):
        for j in range(80):
            for k in range(710): 
                for n in range(1):
                    output_p[80*710*i +710* j + k+n] = input.flatten()[80*710*i +710* j + k+n] 

    print(f"input data handle  :{output_p.reshape(21,80,710,1)}")

    diff = output_p.flatten() - output.flatten()
    print(f"diff max:{diff.max()}")

    print("================================ padding test ============================\n")
    output_ = torch.zeros([21,80,710,63],dtype=torch.float16)

    output_cat = torch.cat([output,output_],dim=3)
    # print(f"output_cat data :{output_cat}")
    print(f"output_cat data shape :{output_cat.shape}")

    output_pa = torch.zeros([21,80,710,64],dtype=torch.float16).flatten()
    for i in range(21):
        for j in range(80):
            for k in range(710): 
                for n in range(1):
                    output_pa[80*710*64*i + 64*710*j+ 64*k + n] = output.flatten()[ 80*710*i + 710* j+k+n] 

    print(f"input data handle padding :{output_pa.reshape(21,80,710,64)}")
    diff_ = output_cat.flatten() - output_pa.flatten()
    print(f"diff_ max:{diff_.max()}")

def run_two():
    print("ok ")
    input = torch.randn([24,1,710,80],dtype=torch.float16)

    output = input.permute(0,2,3,1).contiguous() # 24,710,80,1
    print(f"input data permute :{output}")
    print(f"input data permute shape:{output.shape}")

    

    output_p = torch.zeros([24,710,80,1],dtype=torch.float16,).flatten()
    # for i in range(24):
    #     for j in range(1):
    #         for k in range(80): 
    #             for n in range(710):
    #                 output_p[80*710*i + j + 710*k+n] = input.flatten()[80*710*i + j + 710*k+n] 

    for i in range(24):
        for j in range(710):
            for k in range(80): 
                for n in range(1):
                    output_p[80*710*i +80* j + k+n] = input.flatten()[80*710*i +80* j + k+n] 

    print(f"input data handle  :{output_p.reshape(24,710,80,1)}")

    diff = output_p.flatten() - output.flatten()
    print(f"diff max:{diff.max()}")

    print("================================ padding test ============================\n")
    output_ = torch.zeros([24,710,80,63],dtype=torch.float16)

    output_cat = torch.cat([output,output_],dim=3)
    # print(f"output_cat data :{output_cat}")
    print(f"output_cat data shape :{output_cat.shape}")

    output_pa = torch.zeros([24,710,80,64],dtype=torch.float16).flatten()
    for i in range(24):
        for j in range(710):
            for k in range(80): 
                for n in range(1):
                    output_pa[80*710*64*i + 64*80*j+ 64*k + n] = output.flatten()[ 80*710*i +80* j + k+n] 

    print(f"input data handle padding :{output_pa.reshape(24,710,80,64)}")
    diff_ = output_cat.flatten() - output_pa.flatten()
    print(f"diff_ max:{diff_.max()}")



def test_unsqueeze():
    input = torch.arange(12).reshape(2,3,2)
    print(f"input data :{input}")
    print(f"input data shape:{input.shape}")

    output = input.unsqueeze(1)
    print(f"output data :{output}")
    print(f"output data shape:{output.shape}")
if __name__ == "__main__":
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)

    # test()
    # test_two_dimension()
    # test_transpose()
    # test_four_dimension()
    # test_four_dimension_()
    # run()
    # run_two()
    test_unsqueeze()


    