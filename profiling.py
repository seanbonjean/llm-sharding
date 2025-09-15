from utils.node_profiler import NodeProfiler
import torch

profiler = NodeProfiler(
    "C:/Users/sean-/Desktop/shards/Llama-2-7b-chat-hf_float16",  # 权重shard暂存SSD，避免硬盘瓶颈
    device="cuda:0",
    dtype=torch.float16
)
# profiler.go_through_every_shards(out_token_num=256)
profiler.profiling()
