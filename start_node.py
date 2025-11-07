import sys
import torch
from utils.node_worker import NodeController

# 直接运行时，使用该端口号
port = 40700
# 由run_this.sh启动时，读取其给定的端口号
if len(sys.argv) > 2 and sys.argv[1] == "--port":
    port = int(sys.argv[2])
elif len(sys.argv) > 1:
    port = int(sys.argv[1])

controller = NodeController(
    "shards/Llama-2-7b-chat-hf_float16",
    # "C:/Users/sean-/Desktop/shards/Llama-2-7b-chat-hf_float16",  # 权重shard暂存SSD，避免硬盘瓶颈
    device="cuda:0",
    dtype=torch.float16,
    listen_port=port,
)
controller.run_worker_loop()
