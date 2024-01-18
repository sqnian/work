import numpy as np
import torch
import pad_test 
import os
os.environ['CUDA_VISIBLE_DEVICES']="1"

def pad_impl(data, raw_pads, mode, constant_values=0.0, axes=None):  # type: ignore
    input_rank = data.ndim
    if axes is None:
        axes = list(range(input_rank))
    else:
        axes = [axis if axis >= 0 else axis + input_rank for axis in axes]
    num_axes = len(axes)

    if num_axes * 2 != raw_pads.size:
        raise Exception("The number of elements in raw_pads should be 2 * num_axes")

    pad_width = []
    for _ in range(input_rank):
        pad_width += [[0, 0]]  # init to zero

    # re-order to np.pad accepted order ((x1_begin, x1_end), (x2_begin, x2_end), ...)
    for i in range(num_axes):
        axis = axes[i]
        if axis < 0:
            axis = input_rank + axis
        pad_width[axis] = [raw_pads[i], raw_pads[i + num_axes]]

    if mode == "constant":
        y = np.pad(
            data,
            pad_width=pad_width,
            mode=mode,
            constant_values=constant_values,
        )
        return y

    y = np.pad(
        data,
        pad_width=pad_width,
        mode=mode,
    )

    return y


def test_func():
    data = np.array([ [1.0, 1.2], [2.3, 3.4], [4.5, 5.7] ])
    pads = np.array([0, 2, 0, 0])
    y = pad_impl(data, pads,"constant")
    print('y:',y)


def test_224_228():
    # x = np.random.randn(1, 3, 224, 224).astype(np.float32)
    # pads = np.array([0, 0, 2, 2, 0, 0, 2, 2]).astype(np.int64)

    x = np.random.randn(1,3,224, 224).astype(np.float16)
    input = torch.tensor(x).cuda()
    pads = np.array([0,0,2, 2,0,0, 2, 2]).astype(np.int64)
    y = pad_impl(x, pads,"constant")
    # print('y:',y)
    output = torch.tensor(y).cuda()
    print('output:',output)
    print('output shape:',output.shape)
    print('input shape: ',input.shape)

    print('from input index to output index\n ')
    out_test = torch.zeros([1,3,228,228],dtype=torch.float16).cuda()
    for i in range(input.shape[0]):
        for j in range(input.shape[1]):
            for k in range(input.shape[2]):
                for m in range(input.shape[3]):
                    # out_index = m + 2*out_test.shape[3] + 2 + out_test.shape[2]*k + out_test.shape[3]*out_test.shape[2]*j+ out_test.shape[3]*out_test.shape[2]*out_test.shape[1]*i
                    out_index = i*(2+2+input.shape[3])*(2+2+input.shape[2])*input.shape[1] + j *(2+2+input.shape[2])*(2+2+input.shape[3]) + k *  (2+2+input.shape[2])  + m + 2*(2+2+input.shape[3]) + 2
                    # out_index = i*8*8*3+ j *8*8 + k * 8 + m + 2*8+ 2
                    in_index = i*input.shape[3]*input.shape[2]*input.shape[1] + j *input.shape[3]*input.shape[2] + k * input.shape[3] + m
                    out_test.flatten()[out_index] = input.flatten()[in_index]

    diff_test = torch.abs(output.flatten() - out_test.flatten())
    print('diff test max: ',diff_test.max())

    # print('output flatten :',output.flatten())
    # print('out_test flatten :',out_test.flatten())
        

def test_28_30():
    # x = np.random.randn(1, 3, 224, 224).astype(np.float32)
    # pads = np.array([0, 0, 2, 2, 0, 0, 2, 2]).astype(np.int64)

    x = np.random.randn(1,64,28, 28).astype(np.float16)
    input = torch.tensor(x).cuda()
    pads = np.array([0,0,1, 1,0,0, 1, 1]).astype(np.int64)
    y = pad_impl(x, pads,"constant")
    # print('y:',y)
    output = torch.tensor(y).cuda()
    print('output:',output)
    print('output shape:',output.shape)
    print('input shape: ',input.shape)

    print('from input index to output index\n ')
    out_test = torch.zeros([1,64,30,30],dtype=torch.float16).cuda()
    for i in range(input.shape[0]):
        for j in range(input.shape[1]):
            for k in range(input.shape[2]):
                for m in range(input.shape[3]):
                    out_index = i*(1+1+input.shape[3])*(1+1+input.shape[2])*input.shape[1] + j *(1+1+input.shape[2])*(1+1+input.shape[3]) + k *  (1+1+input.shape[2])  + m + 1*(1+1+input.shape[3]) + 1
                    in_index = i*input.shape[3]*input.shape[2]*input.shape[1] + j *input.shape[3]*input.shape[2] + k * input.shape[3] + m
                    out_test.flatten()[out_index] = input.flatten()[in_index]

    diff_test = torch.abs(output.flatten() - out_test.flatten())
    print('diff test max: ',diff_test.max())

def test():
    # x = np.random.randn(1, 3, 224, 224).astype(np.float32)
    # pads = np.array([0, 0, 2, 2, 0, 0, 2, 2]).astype(np.int64)

    # x = np.random.randn(1,3,224, 224).astype(np.float16)
    # pads = np.array([0,0,2, 2,0,0, 2, 2]).astype(np.int64)
    x = np.random.randn(1,64,28, 28).astype(np.float16)
    pads = np.array([0,0,1, 1,0,0, 1, 1]).astype(np.int64)
    input = torch.tensor(x).cuda()
    y = pad_impl(x, pads,"constant")
    # print('y:',y)
    output = torch.tensor(y).cuda()
    
    out_test = pad_test.test_pad(input)
    diff_test = torch.abs(output.flatten() - out_test.flatten())
    print('diff test max: ',diff_test.max())
    print('output shape:',output.shape)
    print('out_test shape:',out_test.shape)

if __name__ == '__main__':
    # test_func()
    # test_224_228()
    # test_28_30()
    test()

    



