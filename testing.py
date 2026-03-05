# from utils.node_worker import NodeWorker, transfer_data, receive_data
# import torch
#
# # # transfer
# # node_worker = NodeWorker(
# #     can_receive_user_request=True,
# #     # shards_path="shards/Llama-2-7b-chat-hf_float16",
# #     shards_path="C:/Users/sean-/Desktop/shards/Llama-2-7b-chat-hf_float16",  # 权重shard暂存SSD，避免硬盘瓶颈
# #     device="cuda:0",
# #     dtype=torch.float16
# # )
# # node_worker.load_shards(0, 1)
# # data = node_worker.receive_user_request(request="Write a poem about the blue sky.")
# # data = node_worker.pass_through_shard(data)
# # transfer_data(data, save_path="results/data.pt")
#
# # receive
# node_worker = NodeWorker(
#     can_receive_user_request=False,
#     # shards_path="shards/Llama-2-7b-chat-hf_float16",
#     shards_path="C:/Users/sean-/Desktop/shards/Llama-2-7b-chat-hf_float16",  # 权重shard暂存SSD，避免硬盘瓶颈
#     device="cuda:0",
#     dtype=torch.float16
# )
# node_worker.load_shards(1, 2)
# data = receive_data("results/data.pt")
# data = node_worker.pass_through_shard(data)
#
# # transfer_data(data, save_path="results/data.pt")
#
# first_node = NodeWorker(
#     can_receive_user_request=True,
#     # shards_path="shards/Llama-2-7b-chat-hf_float16",
#     shards_path="C:/Users/sean-/Desktop/shards/Llama-2-7b-chat-hf_float16",  # 权重shard暂存SSD，避免硬盘瓶颈
#     device="cuda:0",
#     dtype=torch.float16
# )
# first_node.load_shards(0, 1)
# reached_eos, data = first_node.receive_next_token(data)

# from utils.node_profiler import NodeProfiler
# import torch
#
# profiler = NodeProfiler(
#     # "shards/Llama-2-7b-chat-hf_float16",
#     "C:/Users/sean-/Desktop/shards/Llama-2-7b-chat-hf_float16",  # 权重shard暂存SSD，避免硬盘瓶颈
#     device="cuda:0",
#     dtype=torch.float16
# )
# profiler.go_through_every_shards(out_token_num=256)

from utils.config_sender import ConfigSender

senders = list()
for i in range(3):
    senders.append(ConfigSender())

# senders[0].build_config(shards_start=0,
#                         shards_end=3,
#                         can_receive_user_request=True,
#                         src_addr="tcp://*:40800",
#                         dst_addr="tcp://172.16.0.3:40800",
#                         first_node_addr="tcp://172.16.0.2:40800",
#                         )
# senders[0].send_config(node_ip="172.16.0.2")
#
# senders[1].build_config(shards_start=3,
#                         shards_end=4,
#                         can_receive_user_request=True,
#                         src_addr="tcp://*:40800",
#                         dst_addr="tcp://172.16.0.1:40800",
#                         first_node_addr="tcp://172.16.0.2:40800",
#                         )
# senders[1].send_config(node_ip="172.16.0.3")

# senders[2].build_config(shards_start=7,
#                         shards_end=28,
#                         can_receive_user_request=True,
#                         src_addr="tcp://*:40800",
#                         dst_addr="tcp://172.16.0.2:40800",
#                         first_node_addr="tcp://172.16.0.2:40800",
#                         )
# senders[2].send_config(node_ip="172.16.0.1")

senders[0].build_config(shards_start=0,
                        shards_end=7,
                        can_receive_user_request=True,
                        src_addr="tcp://*:40800",
                        dst_addr="tcp://172.16.0.1:40800",
                        first_node_addr="tcp://172.16.0.2:40800",
                        )
senders[0].send_config(node_ip="172.16.0.2")

senders[1].build_config(shards_start=7,
                        shards_end=28,
                        can_receive_user_request=True,
                        src_addr="tcp://*:40800",
                        dst_addr="tcp://172.16.0.2:40800",
                        first_node_addr="tcp://172.16.0.2:40800",
                        )
senders[1].send_config(node_ip="172.16.0.1")

# 保持运行，保留 socket
while True:
    pass
