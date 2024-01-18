#-*-coding:utf-8-*-

import torch
import os
os.environ['CUDA_VISIBLE_DEVICES']  = "7"
device = "cuda:0"
from typing import  List
import conformer_infer_opt
import time


def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """Make mask tensor containing indices of padded part.

    See description of make_non_pad_mask.

    Args:
        lengths (torch.Tensor): Batch of lengths (B,).
    Returns:
        torch.Tensor: Mask tensor containing indices of padded part.

    Examples:
        >>> lengths = [5, 3, 2]
        >>> make_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]
    """
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else lengths.max().item()
    seq_range = torch.arange(0,
                             max_len,
                             dtype=torch.int64,
                             device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return mask

def remove_duplicates_and_blank(hyp: List[int]) -> List[int]:
    new_hyp: List[int] = []
    cur = 0
    while cur < len(hyp):
        if hyp[cur] != 0:
            new_hyp.append(hyp[cur])
        prev = cur
        while cur < len(hyp) and hyp[cur] == hyp[prev]:  # 
            cur += 1
    return new_hyp

   

def test_top1():
    """
    input : tensor batch_size, max_len, vocab_size

    """
    input = torch.load('topk_before.pt')  # 24, 176, 4233
 
    # top k
    topk_prob, topk_index = input.topk(1,dim=2) #topk_index : 24, 176,1
    # print(f'topk_index type :{topk_index.dtype}\n')

    cu_out = conformer_infer_opt.test_func(input, topk_index, 1)
    cu_out = cu_out.reshape(24,176)
    # print(f"cu_out shape:{cu_out.shape}\n")
    print(f"cu_out flatten() [:176]:{cu_out.flatten()[:176]}\n")
    print(f"topk_index flatten()[:176]:{topk_index.flatten()[:176]}\n")
     
    diff = torch.abs(cu_out.flatten() - topk_index.flatten())
    print(f'diff max:{diff.max()}\n')

    print("time analyse \n")
    num_iter = 100
    torch.cuda.synchronize()
    start_time = time.time()
    for i in range(num_iter):
        topk_prob, topk_index_pt = input.topk(1,dim=2)

    torch.cuda.synchronize()
    pt_time = time.time() - start_time

    torch.cuda.synchronize()
    start_time = time.time()
    for i in range(num_iter):
         topk_index_cu = conformer_infer_opt.test_func(input, topk_index, 1)

    torch.cuda.synchronize()
    cuda_time = time.time() - start_time
    print(f'time pt_time {pt_time/num_iter} cuda_time {cuda_time/num_iter} pt_time/cuda_time {pt_time/cuda_time}\n')


def test_top1_remove_duplicate_blank():
    input = torch.load('topk_before.pt')  # 24, 176, 4233
    # top k
    topk_prob, topk_index = input.topk(1,dim=2) #topk_index : 24, 176,1
    # print(f'topk_index type :{topk_index.dtype}\n')
    topk_index = topk_index.view(24,176)

    cu_out = conformer_infer_opt.test_func(input, topk_index, 1)
    cu_out = cu_out.reshape(24,176)
    print(f"cu_out flatten() [:176]:{cu_out.flatten()[:176]}\n")

    # encoder_mask = torch.load('encoder_mask.pt') # encoder_mask.pt 24, 1,176
    # encoder_mask = encoder_mask.squeeze(1).sum(1) 
    # mask = make_pad_mask(encoder_mask,176)
    # topk_index = topk_index.masked_fill_(mask,4232) 
    # print(f"topk_index flatten()[:176]:{topk_index.flatten()[:176]}\n")
    # print(f"topk_index flatten()[176:176*2]:{topk_index.flatten()[176:176*2]}\n")

    # hyps = [hyp.tolist() for hyp in topk_index]
    # hyps = [remove_duplicates_and_blank(hyp) for hyp in hyps]
    # print(f"hyps[0]:{hyps[0]}\n")

    # print("test two\n")
    # print(f'cu_out flatten() [176:176*2]:{cu_out.flatten()[176:176*2]}\n')
    # print(f"hyps[0]:{hyps[1]}\n")


if __name__ == "__main__":
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(1234)
    # test_top1()
    test_top1_remove_duplicate_blank()