from utils.node_profiler import NodeProfiler
import torch

profiler = NodeProfiler(
    "shards/Llama-3___2-3B-Instruct_float16",
    # "C:/Users/sean-/Desktop/shards/Llama-2-7b-chat-hf_float16",  # 权重shard暂存SSD，避免硬盘瓶颈
    device="cuda:0",
    dtype=torch.float16
)

profiler.assist_profile_compute_capability(
    target_max_layer_num=7,
    src_addr="tcp://*:40800",
    dst_addr="tcp://172.16.0.2:40800",
)
