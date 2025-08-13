import os
import re
import time
import json
import psutil
import torch
import safetensors.torch as st
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


class LLMProfiler:
    def __init__(self, model_path, device="cpu", dtype=torch.float32,
                 batch_size=1, prompt="Hello", autoreg_steps=5, shard_size=4,
                 output_file="results/profiling_result_" + time.strftime("%m%d_%H_%M_%S", time.localtime()) + ".json"):
        """
        :param model_path: 模型路径 (Hugging Face 格式)
        :param device: "cpu" 或 "cuda:0"
        :param dtype: torch.float32 / torch.float16
        :param batch_size: 批量大小
        :param prompt: 用于 prefill 测试的 prompt
        :param autoreg_steps: autoregressive 阶段生成 token 数（即：prefill阶段之后进行几次autoreg来测试设备的execution time）
        :param shard_size: 每个分片的层数（每次加载几层）
        :param output_file: 保存结果的文件
        """
        self.model_path = model_path
        self.device = torch.device(device)
        self.dtype = dtype
        self.batch_size = batch_size
        self.prompt = prompt
        self.autoreg_steps = autoreg_steps
        self.shard_size = shard_size
        self.output_file = output_file

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        # 模型配置只读一次（不用加载全部权重）
        config = AutoConfig.from_pretrained(self.model_path)
        tmp_model = AutoModelForCausalLM.from_config(config)
        self.num_layers = tmp_model.config.num_hidden_layers
        self.config = tmp_model.config
        del tmp_model

        self.layer_to_file = self._index_safetensors_files()

    def _index_safetensors_files(self):
        """
        扫描所有 safetensors 文件，构建 {layer_idx: filename} 映射
        也就是说，提前查询，记录好各层weight都存在哪个.safetensors里
        """
        mapping = {}
        for fname in os.listdir(self.model_path):
            if fname.endswith(".safetensors"):
                path = os.path.join(self.model_path, fname)
                try:
                    keys = st.safe_open(path, framework="pt", device="cpu").keys()
                except Exception as e:
                    print(f"读取 {fname} 出错: {e}")
                    continue
                for k in keys:
                    m = re.match(r"model\.layers\.(\d+)\.", k)
                    if m:
                        layer_idx = int(m.group(1))
                        mapping[layer_idx] = fname
        return mapping

    def _load_shard(self, start_layer, end_layer):
        """
        按需加载指定层范围的 shard，不会一次性加载全模型
        """
        # 1. 创建空模型结构（meta device）
        model = AutoModelForCausalLM.from_config(self.config)
        model.to("meta")

        # 2. 加载 embedding 和 lm_head（只需要一次）
        embed_file = next((f for f in os.listdir(self.model_path) if f.endswith(".safetensors")), None)
        if embed_file:
            with st.safe_open(os.path.join(self.model_path, embed_file), framework="pt", device="cpu") as f:
                embed_state = {k.replace("model.embed_tokens.", ""): f.get_tensor(k)
                               for k in f.keys() if k.startswith("model.embed_tokens.")}
                if embed_state:
                    model.model.embed_tokens.load_state_dict(embed_state)
                lm_head_state = {k.replace("lm_head.", ""): f.get_tensor(k)
                                 for k in f.keys() if k.startswith("lm_head.")}
                if lm_head_state:
                    model.lm_head.load_state_dict(lm_head_state)

        # 3. 逐层加载
        for layer_idx in range(start_layer, end_layer):
            if layer_idx not in self.layer_to_file:
                continue

            shard_file = os.path.join(self.model_path, self.layer_to_file[layer_idx])
            with st.safe_open(shard_file, framework="pt", device="cpu") as f:
                layer_state = {
                    k.replace(f"model.layers.{layer_idx}.", ""): f.get_tensor(k)
                    for k in f.keys()
                    if k.startswith(f"model.layers.{layer_idx}.") and "rotary_emb.inv_freq" not in k
                }

            model.model.layers[layer_idx].load_state_dict(layer_state, strict=False)
            torch.cuda.empty_cache()

        # 4. 放到目标设备
        model.to(self.device)
        model.eval()
        return model

    # def _load_shard(self, start_layer, end_layer):
    #     """
    #     加载部分层的权重到设备
    #     """
    #     model = AutoModelForCausalLM.from_pretrained(
    #         self.model_path,
    #         torch_dtype=self.dtype,
    #         device_map={"": "cpu"}  # 先加载到 CPU
    #     )

    #     # 把不需要的层移到 meta 以释放内存
    #     for i, layer in enumerate(model.model.layers):
    #         if i < start_layer or i >= end_layer:
    #             model.model.layers[i] = model.model.layers[i].to("meta")

    #     # 把 shard 放到目标 device
    #     for i in range(start_layer, end_layer):
    #         model.model.layers[i] = model.model.layers[i].to(self.device)

    #     # 词嵌入和输出头也放 device
    #     model.model.embed_tokens = model.model.embed_tokens.to(self.device)
    #     model.lm_head = model.lm_head.to(self.device)
    #     model.to(self.device)
    #     model.eval()
    #     return model

    def _sizeof_tensor(self, t):
        return t.numel() * t.element_size()

    def profile(self):
        results = {
            "device": str(self.device),
            "dtype": str(self.dtype),
            "free_mem_bytes": psutil.virtual_memory().available,
            "param_bytes_per_layer": [],
            "layer_times_prefill_ms": [],
            "activations_prefill_bytes": {},
            "layer_times_autoreg_ms": [],
            "activations_autoreg_bytes": {},
            "kv_cache_bytes_per_layer": []
        }

        # 准备输入
        input_ids = self.tokenizer(
            self.prompt, return_tensors="pt").input_ids.to(self.device)

        # 动态分片测每层
        for start in range(0, self.num_layers, self.shard_size):
            end = min(start + self.shard_size, self.num_layers)
            model = self._load_shard(start, end)

            # Prefill 测试
            for idx in range(start, end):
                layer = model.model.layers[idx]
                layer_params = sum(p.numel() * p.element_size()
                                   for p in layer.parameters())
                results["param_bytes_per_layer"].append(layer_params)

                torch.cuda.synchronize() if self.device.type == "cuda" else None
                t0 = time.time()
                with torch.inference_mode():
                    hidden_states = model.model.embed_tokens(input_ids)
                    hidden_states = layer(hidden_states)
                torch.cuda.synchronize() if self.device.type == "cuda" else None
                t1 = time.time()
                results["layer_times_prefill_ms"].append((t1 - t0) * 1000)
                results["activations_prefill_bytes"][idx] = self._sizeof_tensor(
                    hidden_states)

            # Autoregressive 测试
            past_key_values = None
            for step in range(self.autoreg_steps):
                next_input = torch.randint(
                    low=0,
                    high=model.config.vocab_size,
                    size=(self.batch_size, 1),
                    device=self.device
                )
                torch.cuda.synchronize() if self.device.type == "cuda" else None
                t0 = time.time()
                with torch.inference_mode():
                    out = model(
                        next_input, past_key_values=past_key_values, use_cache=True)
                torch.cuda.synchronize() if self.device.type == "cuda" else None
                t1 = time.time()
                results["layer_times_autoreg_ms"].append((t1 - t0) * 1000)
                past_key_values = out.past_key_values

            if past_key_values:
                for pkv in past_key_values:
                    k, v = pkv
                    results["kv_cache_bytes_per_layer"].append(
                        self._sizeof_tensor(k) + self._sizeof_tensor(v))

            del model
            torch.cuda.empty_cache()

        with open(self.output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[Profiler] Done. Results saved to {self.output_file}")


if __name__ == "__main__":
    profiler = LLMProfiler(
        model_path="../weights/Llama-2-7b-chat-hf",
        device="cpu",
        dtype=torch.float32,
        shard_size=2,
        autoreg_steps=3
    )
    profiler.profile()
