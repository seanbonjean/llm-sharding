import os
import time
import torch
from transformers import LlamaConfig, AutoTokenizer
from transformers.cache_utils import DynamicCache
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

from utils.node_worker import NodeWorker
from utils.shard_loader import LlamaShardPart
from utils.forwarding_utils import build_position_ids


class NodeProfiler:
    def __init__(self, shards_path: str, device="cpu", dtype=torch.float16):
        """
        :param shards_path: 切片路径
        :param device: "cpu" 或 "cuda:0" 等
        :param dtype: torch.float32 / torch.float16 等
        """
        self.shards_path = shards_path
        self.device = torch.device(device)
        self.dtype = dtype
        self.config = LlamaConfig.from_pretrained(self.shards_path)
        self.layer_num = self.config.num_hidden_layers
        # self.num_heads = self.config.num_attention_heads
        # self.hidden_size = self.config.hidden_size
        # self.head_dim = self.hidden_size // self.num_heads

        self.shards = []  # 保存各层权重

    def profile_max_layer_num(self):
        node_worker = NodeWorker(
            src_addr="tcp://*:40800",
            dst_addr="tcp://127.0.0.1:40801",
            can_receive_user_request=True,
            shards_path=self.shards_path,
            device=self.device,
            dtype=self.dtype
        )
        max_layer_num = 0  # 节点能放下的最大层数（包含embedding的情况下）
        try:
            for i in range(self.layer_num):
                node_worker.load_shards(0, i + 1)
                max_layer_num = i + 1
        except torch.cuda.OutOfMemoryError:
            print(f"[INFO] max layer num: {max_layer_num}")
        return max_layer_num

    def profile_compute_capability(self, max_layer_num: int = None):
        """
        :param max_layer_num: 节点能放下的最大层数（包含embedding的情况下）; -1代表设备必定能放下所有，忽略内存限制
        """
        if max_layer_num is None:
            print("[WARNING] max_layer_num needed, pls rerun this profile using the result after automaticly evoking profile_max_layer_num(), or if the device can load all model layers, this profile process will continue running without killing by CUDA OOM Exception.")
            max_layer_num = self.profile_max_layer_num()

        computation_latencies = list()
        requests_token_length = [8, 16, 32, 64, 128, 256, 512]
        if max(requests_token_length) > self.config.max_position_embeddings:
            raise ValueError("[ERROR] requested prompt length exceeds model max_position_embeddings.")

        node = NodeWorker(
            src_addr="tcp://*:40800",
            dst_addr="tcp://127.0.0.1:40800",
            can_receive_user_request=True,
            shards_path=self.shards_path,
            device=self.device,
            dtype=self.dtype
        )
        if max_layer_num == -1:
            loaded_layer_num = self.layer_num
        else:
            loaded_layer_num = max_layer_num - 1  # 预留给 KV Cache 一些空间
            if loaded_layer_num <= 0:
                raise ValueError("[ERROR] max_layer_num is too small to reserve space for KV cache.")
        node.load_shards(0, loaded_layer_num)

        # 由多个 frag 重复组成完整的长 prompt
        long_prompt_fragment = (
            "Distributed inference splits a language model across multiple edge devices so that "
            "each device processes part of the network while cooperating with the others. "
        )
        long_prompt = long_prompt_fragment
        long_prompt_input_ids = node.tokenizer(long_prompt, return_tensors="pt")["input_ids"]
        # 重复拼接 frag 直到最长长度达到需求
        while long_prompt_input_ids.shape[1] < max(requests_token_length):
            long_prompt += long_prompt_fragment
            long_prompt_input_ids = node.tokenizer(long_prompt, return_tensors="pt")["input_ids"]
        # 根据指定的各个 token 长度，构建不同长度的 prompt (直接以 token ids 格式截断，以保证 token length 正确)
        input_ids_of_each_request = [
            long_prompt_input_ids[:, :token_length].clone()
            for token_length in requests_token_length
        ]

        # ! 实验发现首次测试的时延总会出现异常值，因此在测试前先做 warm-up
        # 正式计时前先用最长 prompt 进行一次 warm-up：
        # 让 CUDA lazy initialization、kernel 首次加载、序列化、ZMQ 自发自收等首轮开销先发生，
        # 避免它们只落在第一个测试点上，导致第一个 latency 成为异常值并干扰后续拟合
        print("[INFO] warming up prefill profiling path...")
        warmup_input_ids = input_ids_of_each_request[-1]
        warmup_data0 = node.receive_user_request(input_ids=warmup_input_ids)
        warmup_data1 = node.pass_through_shard(warmup_data0)
        node.communicator.transfer_data(warmup_data1)
        warmup_data1_recv = node.communicator.receive_data()
        if loaded_layer_num == self.layer_num:
            node.receive_next_token(warmup_data1_recv)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        node.clear_KV_cache()
        del warmup_data0
        del warmup_data1
        del warmup_data1_recv
        print("[INFO] prefill profiling warm-up finished.")

        # 时延的实际测试，注意与 warm-up 代码同步修改
        for i in range(len(requests_token_length)):
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            start_time = time.perf_counter()
            data0 = node.receive_user_request(input_ids=input_ids_of_each_request[i])
            data1 = node.pass_through_shard(data0)
            node.communicator.transfer_data(data1)  # 同一个 node 自己给自己传数据
            data1_recv = node.communicator.receive_data()
            if loaded_layer_num == self.layer_num:
                _, _ = node.receive_next_token(data1_recv)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            end_time = time.perf_counter()
            computation_latency = end_time - start_time
            computation_latencies.append(computation_latency)
            node.clear_KV_cache()  # 没有遇到 EOS token，需要手动清除KV Cache

        if len(computation_latencies) != len(requests_token_length):
            raise ValueError("[ERROR] tested computation latency number is not equal to request number.")

        # 如果设备显存不足导致 profile 时没有完全加载所有 layer，则近似换算到完整加载所有 layer 时的 latency
        latency_scale = self.layer_num / loaded_layer_num
        normalized_latencies = [latency * latency_scale for latency in computation_latencies]
        print("[INFO] tested prompt token lengths=", requests_token_length)
        print("[INFO] normalized first-token latencies=", normalized_latencies)

        prefill_comp_capa_sum = 0  # prefill 阶段 计算能力 测试的所有计算能力之和 (仅用于求平均)
        print("[INFO] each compute capability c_k=", end=" ")
        for i, latency in enumerate(normalized_latencies):
            prefill_comp_capa = latency / requests_token_length[i]  # prefill 阶段 不同输入 token 长度下各自的计算能力
            prefill_comp_capa_sum += prefill_comp_capa
            print(prefill_comp_capa, end=" / ")
        prefill_comp_capa_avg = prefill_comp_capa_sum / len(requests_token_length)
        print("\n[INFO] average compute capability c_k=", str(prefill_comp_capa_avg), f" sec / (token * {self.layer_num}layer)")

        # 将测试点转换为 double tensor，后续拟合需要在这些离散点上求解模型参数
        # S 表示 prompt token 数，T(S) 表示对应的 first-token latency
        token_lengths = torch.tensor(requests_token_length, dtype=torch.float64)
        latencies = torch.tensor(normalized_latencies, dtype=torch.float64)

        # 一次时延模型假设 T(S) = a * S + b
        # 对每个测试点 S_i，方程可写成 [S_i, 1] @ [a, b]^T = T_i
        # 因此设计矩阵的每一行都是 [S_i, 1]
        linear_design_matrix = torch.stack(
            (token_lengths, torch.ones_like(token_lengths)),
            dim=1,
        )

        # 二次时延模型假设 T(S) = a * S^2 + b * S + c
        # 对每个测试点 S_i，方程可写成 [S_i^2, S_i, 1] @ [a, b, c]^T = T_i
        # 因此设计矩阵的每一行都是 [S_i^2, S_i, 1]
        quadratic_design_matrix = torch.stack(
            (token_lengths ** 2, token_lengths, torch.ones_like(token_lengths)),
            dim=1,
        )

        # torch.linalg.lstsq 以矩阵形式求解最小二乘问题：
        # 在线性模型中求使 ||X_linear * theta - y||_2 最小的 theta=[a, b]
        # 在二次模型中求使 ||X_quadratic * theta - y||_2 最小的 theta=[a, b, c]
        # 将 latency 变成列向量 [N, 1] 后再求解，可让设计矩阵与目标向量的形状更明确
        latency_column = latencies.unsqueeze(1)
        linear_coefficients = torch.linalg.lstsq(linear_design_matrix, latency_column).solution.squeeze(1)
        quadratic_coefficients = torch.linalg.lstsq(quadratic_design_matrix, latency_column).solution.squeeze(1)

        # 将求出的参数重新代回各自模型，得到每个已测长度上的预测时延
        # 后续误差指标就是通过“实测时延 - 拟合时延”计算出来的
        linear_fitted_latencies = linear_design_matrix @ linear_coefficients
        quadratic_fitted_latencies = quadratic_design_matrix @ quadratic_coefficients

        def fit_metrics(fitted_latencies: torch.Tensor) -> tuple[float, float]:
            """
            根据拟合值计算两个常见误差指标：
            1. RMSE: 均方根误差，单位仍为秒，越小表示拟合曲线离实测点越近
            2. R^2: 决定系数，表示模型解释了多少时延方差，越接近 1 通常表示拟合越好
            """
            # residuals_i = 实测时延 T_i - 拟合时延 \hat{T_i}
            residuals = latencies - fitted_latencies
            # RMSE = sqrt(mean(residuals_i^2))
            rmse = torch.sqrt(torch.mean(residuals ** 2)).item()
            # SST = sum((T_i - mean(T))^2)，表示所有实测点相对均值的总波动
            total_sum_of_squares = torch.sum((latencies - latencies.mean()) ** 2)
            if total_sum_of_squares.item() == 0:
                # 如果所有实测时延完全相同，则 SST=0，R^2 的分母为 0，没有定义
                r_squared = float("nan")
            else:
                # SSE = sum(residuals_i^2)，表示拟合后仍未解释掉的误差
                residual_sum_of_squares = torch.sum(residuals ** 2)
                # R^2 = 1 - SSE / SST
                r_squared = (1 - residual_sum_of_squares / total_sum_of_squares).item()
            return rmse, r_squared

        # 分别评估一次模型与二次模型，并拆出系数用于打印和后续绘图
        linear_rmse, linear_r_squared = fit_metrics(linear_fitted_latencies)
        quadratic_rmse, quadratic_r_squared = fit_metrics(quadratic_fitted_latencies)
        linear_a, linear_b = linear_coefficients.tolist()
        quadratic_a, quadratic_b, quadratic_c = quadratic_coefficients.tolist()
        print(
            "[INFO] linear latency model: "
            f"T(S) = {linear_a:.6e} * S + {linear_b:.6e}; "
            f"RMSE = {linear_rmse:.6e} sec, R^2 = {linear_r_squared:.6f}"
        )
        print(
            "[INFO] quadratic latency model: "
            f"T(S) = {quadratic_a:.6e} * S^2 + {quadratic_b:.6e} * S + {quadratic_c:.6e}; "
            f"RMSE = {quadratic_rmse:.6e} sec, R^2 = {quadratic_r_squared:.6f}"
        )

        import matplotlib
        matplotlib.use("Agg")  # 无头环境中，只将图保存到文件，不打开交互式显示窗口
        import matplotlib.pyplot as plt

        # 为了画出平滑曲线，不只在已有测试点上作图，而是在最短到最长 token 数之间
        # 额外均匀采样 200 个横坐标，再分别代入两个拟合模型得到对应纵坐标
        plot_token_lengths = torch.linspace(
            min(requests_token_length),
            max(requests_token_length),
            steps=200,
            dtype=torch.float64,
        )
        plot_linear_latencies = linear_a * plot_token_lengths + linear_b
        plot_quadratic_latencies = (
            quadratic_a * plot_token_lengths ** 2
            + quadratic_b * plot_token_lengths
            + quadratic_c
        )

        # 将实测点、一次拟合曲线、二次拟合曲线画在同一张图上，便于直观看出
        # 两种模型各自对不同 token 长度区间的贴合程度
        plot_dir = os.path.join("results", "profiling")
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, "profile_prefill_compute_capability.png")
        plt.figure(figsize=(8, 5))
        plt.scatter(requests_token_length, normalized_latencies, label="measured latency")
        plt.plot(plot_token_lengths.tolist(), plot_linear_latencies.tolist(), label="linear fit")
        plt.plot(plot_token_lengths.tolist(), plot_quadratic_latencies.tolist(), label="quadratic fit")
        plt.xlabel("Prompt length (tokens)")
        plt.ylabel("First-token latency (sec, full-model equivalent)")
        plt.title("Prefill first-token latency fit")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"[INFO] latency fit plot saved to {plot_path}")

        # TODO 对decode阶段的profile

    def go_through_every_shards(self, out_token_num: int = 50):
        node0 = NodeWorker(
            src_addr="tcp://*:40800",
            dst_addr="tcp://127.0.0.1:40801",
            can_receive_user_request=True,
            shards_path=self.shards_path,
            device=self.device,
            dtype=self.dtype
        )
        node1 = NodeWorker(
            src_addr="tcp://*:40801",
            dst_addr="tcp://127.0.0.1:40802",
            can_receive_user_request=False,
            shards_path=self.shards_path,
            device=self.device,
            dtype=self.dtype
        )
        node2 = NodeWorker(
            src_addr="tcp://*:40802",
            dst_addr="tcp://127.0.0.1:40803",
            can_receive_user_request=False,
            shards_path=self.shards_path,
            device=self.device,
            dtype=self.dtype
        )
        node3 = NodeWorker(
            src_addr="tcp://*:40803",
            dst_addr="tcp://127.0.0.1:40800",
            can_receive_user_request=False,
            shards_path=self.shards_path,
            device=self.device,
            dtype=self.dtype
        )
        node0.load_shards(0, 7)
        node1.load_shards(7, 14)
        node2.load_shards(14, 21)
        node3.load_shards(21, 28)

        # data0 = node0.receive_user_request(request="Write a poem about the blue sky.")
        # data0 = node0.receive_user_request(request="Write a haiku about the blue sky.")
        # data0 = node0.receive_user_request(request="The capital of France is")
        # data0 = node0.receive_user_request(request="Write a poem about the blue sky in one sentence.")
        data0 = node0.receive_user_request(request="Why the sky blue")
        for i in range(out_token_num):
            data1 = node0.pass_through_shard(data0)
            node0.communicator.transfer_data(data1)

            data1_recv = node1.communicator.receive_data()
            data2 = node1.pass_through_shard(data1_recv)
            node1.communicator.transfer_data(data2)

            data2_recv = node2.communicator.receive_data()
            data3 = node2.pass_through_shard(data2_recv)
            node2.communicator.transfer_data(data3)

            data3_recv = node3.communicator.receive_data()
            data4 = node3.pass_through_shard(data3_recv)
            node3.communicator.transfer_data(data4)

            data4_recv = node0.communicator.receive_data()
            reached_end, data0 = node0.receive_next_token(data4_recv)
            if reached_end:
                break

    def go_through_every_shards_only_by_profiler(self, out_token_num: int = 50):
        """
        只通过 node profiler 实现，不实例化 node worker
        """
        # 初始化 KV cache
        past_key_values = [DynamicCache() for _ in range(self.layer_num)]
        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(self.shards_path)
        # 加载嵌入层
        embed_tokens = torch.nn.Embedding(
            self.config.vocab_size, self.config.hidden_size).to(self.device, dtype=self.dtype)
        embed_tokens.load_state_dict(
            torch.load(os.path.join(self.shards_path, "embedding.pth"), map_location=self.device))
        # 加载旋转位置编码（RoPE）
        rope = LlamaRotaryEmbedding(config=self.config, device=self.device).to(self.device)
        # 加载所有分片
        self.shards = [LlamaShardPart(
            self.shards_path,
            ["block_" + str(i) + ".pth"],
            i, i + 1,
            device=self.device,
            dtype=self.dtype,
            add_final_norm=False,
            final_norm_weight=None
        ) for i in range(self.layer_num - 1)]
        self.shards.append(LlamaShardPart(
            self.shards_path,
            ["block_" + str(self.layer_num - 1) + ".pth"],
            self.layer_num - 1, self.layer_num,
            device=self.device,
            dtype=self.dtype,
            add_final_norm=True,
            final_norm_weight="final_norm.pth"
        ))
        # 所有分片转换到 eval模式（关闭dropout）
        for shard in self.shards:
            shard.eval()
        # 加载 lm_head
        lm_head = torch.nn.Linear(
            self.config.hidden_size, self.config.vocab_size, bias=False).to(self.device, dtype=self.dtype)
        lm_head.load_state_dict(
            torch.load(os.path.join(self.shards_path, "lm_head.pth"), map_location=self.device))

        # 分词器
        input_text = "Write a poem about the blue sky."
        print("input: " + input_text)
        inputs = tokenizer(input_text, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]  # 初始 prompt 张量化后的 token id 序列
        generated_ids = [input_ids]  # 存储所有 input 和 output 的 token id 序列

        # 经过嵌入层
        hidden_states = embed_tokens(input_ids)  # [B, S, H]
        batch_size, seq_len, _ = hidden_states.shape

        for i in range(out_token_num):
            # 旋转位置编码（RoPE）获取 cos/sin 表
            position_ids = build_position_ids(past_key_values[0], seq_len, device=self.device, batch_size=batch_size)
            cos, sin = rope(hidden_states, position_ids)  # [B, S, Hd] each; hidden_states 只是用作参考张量 x

            # 逐层跑每个 shard
            for i, shard in enumerate(self.shards):
                hidden_states = hidden_states.to(device=self.device, dtype=self.dtype)
                hidden_states = shard(
                    hidden_states,
                    # attention_mask=inputs["attention_mask"],
                    past_key_value=past_key_values[i],
                    rotary_emb=(cos, sin)
                )

            # 最后的 norm (此处不需要，因为 LlamaShardPart 的 forward 函数重载中，已经包含 final norm 的检测和执行了)
            # hidden_states = self.shards[-1].norm(hidden_states)

            # lm_head 输出 logits
            logits = lm_head(hidden_states)  # [batch, seq_len, vocab_size]
            # 取最后一个位置的预测 token
            next_token_id = torch.argmax(logits[:, -1, :], dim=-1)  # [1]
            # 拼接到已有序列 (不用拼回去，因为此时已经传递到远离用户侧了，只需返回当前的预测 token)
            # input_ids = torch.cat([input_ids, next_token_id.unsqueeze(-1)], dim=-1)
            generated_ids.append(next_token_id.unsqueeze(-1))
            # 解码
            next_token = tokenizer.decode(next_token_id.item())
            print(repr(next_token), end=" ", flush=True)

            if next_token == tokenizer.eos_token:
                break

            # 更新输入
            hidden_states = embed_tokens(next_token_id.unsqueeze(0))  # [B, 1, H]
            seq_len = 1

        print()
        # === 解码最终结果 ===
        final_ids = torch.cat(generated_ids, dim=-1)  # [B, seq_len + out_token_num]
        print("output: ", tokenizer.decode(final_ids[0]))


if __name__ == "__main__":
    profiler = NodeProfiler(
        "../shards/Llama-2-7b-chat-hf_float32",
        device="cuda:0",
        dtype=torch.float32
    )
    profiler.go_through_every_shards()
