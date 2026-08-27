import torch

from utils.node_profiler import NodeProfiler

profiler = NodeProfiler(
    "shards/Llama-3___2-3B-Instruct_float16",
    # "C:/Users/sean-/Desktop/shards/Llama-2-7b-chat-hf_float16",  # 权重shard暂存SSD，避免硬盘瓶颈
    device="cuda:0",
    dtype=torch.float16,
)
# profiler.go_through_every_shards(out_token_num=256)
# profiler.profile_compute_capability(max_layer_num=None)
# profiler.profile_cold_start_latency(max_layer_num=None)

# profiler.profile_compute_capability(
#     max_layer_num=7,
#     assisted=True,
#     src_addr="tcp://*:40800",
#     dst_addr="tcp://172.16.0.1:40800"
# )

# profiler.profile_kv_cache_size()


#################################
### 不同设备长文本 prefill 时延实验

csv_path = profiler.profile_long_context_prefill(
    layer_num=6,
    model_type="llama3.2ins",
    use_thinking_question=False,
    sample_index=0,
    initial_context_length=32,
    context_length_unit="auto",
    repeat_num=3,
)
print(csv_path)

# png_path = NodeProfiler.plot_long_context_prefill(
#     "results/profiling/long_context_prefill/20260827-231615-zmq7xa6z/long_context_prefill_latency-NVIDIA_Jetson_Orin_Nano_Engineering_Reference_Developer_Kit_Super-6layers.csv",
# )
# print(png_path)
#################################
