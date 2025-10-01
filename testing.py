from utils.node_worker import NodeWorker, transfer_data, receive_data
import torch

# # transfer
# node_worker = NodeWorker(
#     can_receive_user_request=True,
#     # shards_path="shards/Llama-2-7b-chat-hf_float16",
#     shards_path="C:/Users/sean-/Desktop/shards/Llama-2-7b-chat-hf_float16",  # 权重shard暂存SSD，避免硬盘瓶颈
#     device="cuda:0",
#     dtype=torch.float16
# )
# node_worker.load_shards(0, 1)
# data = node_worker.receive_user_request(request="Write a poem about the blue sky.")
# data = node_worker.pass_through_shard(data)
# transfer_data(data, save_path="results/data.pt")

# receive
node_worker = NodeWorker(
    can_receive_user_request=False,
    # shards_path="shards/Llama-2-7b-chat-hf_float16",
    shards_path="C:/Users/sean-/Desktop/shards/Llama-2-7b-chat-hf_float16",  # 权重shard暂存SSD，避免硬盘瓶颈
    device="cuda:0",
    dtype=torch.float16
)
node_worker.load_shards(1, 2)
data = receive_data("results/data.pt")
data = node_worker.pass_through_shard(data)

# transfer_data(data, save_path="results/data.pt")

reached_eos, data = node_worker.receive_next_token(data)
