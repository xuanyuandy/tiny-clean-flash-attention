# nsys profile -t cuda,nvtx python3 test_prof.py
import torch
import math
from attention_cutlass import flash_attention_v2_cutlass
import time
import torch.cuda.nvtx as nvtx  # 导入 PyTorch 的 NVTX 模块

'''
simple attention implement without multi head
'''

torch.manual_seed(180)
def get_tensors(BS, HEAD, SEQLEN, DIM, dtype=torch.float16):
    nvtx.range_push("get_tensors")
    q = (torch.empty((BS, HEAD, SEQLEN, DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    k = (torch.empty((BS, HEAD, SEQLEN, DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    v = (torch.empty((BS, HEAD, SEQLEN, DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    nvtx.range_pop()
    return q, k, v

def self_attention(q, k, v, causal=True, sm_scale=1):
    nvtx.range_push("self_attention")
    SEQLEN = q.shape[-2]
    M = torch.tril(torch.ones((SEQLEN, SEQLEN), device="cuda"))
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    if causal:
        p[:, :, M == 0] = float("-inf")
    p = torch.softmax(p.float(), dim=-1).half()
    ref_out = torch.matmul(p, v)
    nvtx.range_pop()
    return ref_out


def run_benchmark(epoch, warmup, func, *args, **kwargs):
    nvtx.range_push(f"run_benchmark: {func.__name__}")
    
    # warmup phase
    for _ in range(warmup):
        nvtx.range_push(f"warmup_{func.__name__}")
        _ = func(*args, **kwargs)
        nvtx.range_pop()
    
    torch.cuda.synchronize()
    time_s = time.time()
    
    # benchmark phase
    for i in range(epoch):
        nvtx.range_push(f"epoch_{i}_{func.__name__}")
        _ = func(*args, **kwargs)
        torch.cuda.synchronize()
        nvtx.range_pop()
    
    time_e = time.time() - time_s
    nvtx.range_pop()
    return time_e


def main(bs=1, head=64, seq_len=4096, dim=64):
    nvtx.range_push(f"main_bs{bs}_head{head}_seq{seq_len}_dim{dim}")
    
    BS, HEAD, SEQLEN, DIM = bs, head, seq_len, dim
    
    # 数据准备
    nvtx.range_push("tensor_preparation")
    q,k,v = get_tensors(BS, HEAD, SEQLEN, DIM, dtype=torch.float16)
    nvtx.range_pop()
    
    warmup = 5
    epoch = 20
    is_causal = True
    sm_scale = 1.0 / math.sqrt(SEQLEN)

    # 基准测试
    nvtx.range_push("benchmark_flash_attention")
    flash2_cutlass_ref = flash_attention_v2_cutlass(q, k, v, is_causal, sm_scale)[0]
    nvtx.range_pop()
    
    nvtx.range_pop()  # main


if __name__ == "__main__":
    nvtx.range_push("script_start")
    
    epoch = 1
    for _ in range(epoch):
        for bs in [1]:
            for head in [1]:
                for seq_len in [256]:
                    for dim in [32]:
                        nvtx.range_push(f"iteration_bs{bs}_head{head}_seq{seq_len}")
                        main(bs, head, seq_len, dim)
                        nvtx.range_pop()
    
    nvtx.range_pop()  # script_start
