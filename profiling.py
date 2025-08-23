from utils.node_profiler import NodeProfiler
import torch

profiler = NodeProfiler(
    "C:/Users/sean-/Desktop/shards/Llama-2-7b-chat-hf_float16",  # C盘是SSD更快一些
    device="cuda:0",
    dtype=torch.float16
)
profiler.go_through_every_shards(out_token_num=256)
