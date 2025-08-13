from utils.profiler import LLMProfiler
import torch

profiler = LLMProfiler(
    model_path="weights/Llama-2-7b-chat-hf",
    device="cpu",
    dtype=torch.float32,
    shard_size=2,
    autoreg_steps=3
)
profiler.profile()
