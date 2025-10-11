from utils.node_worker import NodeController
import torch

controller = NodeController(
    "shards/Llama-2-7b-chat-hf_float16",
    # "C:/Users/sean-/Desktop/shards/Llama-2-7b-chat-hf_float16",  # 权重shard暂存SSD，避免硬盘瓶颈
    device="cuda:0",
    dtype=torch.float16
)
controller.run_worker_loop()
