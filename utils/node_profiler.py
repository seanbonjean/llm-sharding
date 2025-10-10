import os
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

    # TODO 建新方法来做真正的 profile
    def profiling(self):
        node_worker = NodeWorker(
            src_addr="tcp://127.0.0.1:40800",
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

    def go_through_every_shards(self, out_token_num: int = 50):
        node0 = NodeWorker(
            src_addr="tcp://127.0.0.1:40800",
            dst_addr="tcp://127.0.0.1:40801",
            can_receive_user_request=True,
            shards_path=self.shards_path,
            device=self.device,
            dtype=self.dtype
        )
        node1 = NodeWorker(
            src_addr="tcp://127.0.0.1:40801",
            dst_addr="tcp://127.0.0.1:40802",
            can_receive_user_request=False,
            shards_path=self.shards_path,
            device=self.device,
            dtype=self.dtype
        )
        node2 = NodeWorker(
            src_addr="tcp://127.0.0.1:40802",
            dst_addr="tcp://127.0.0.1:40803",
            can_receive_user_request=False,
            shards_path=self.shards_path,
            device=self.device,
            dtype=self.dtype
        )
        node3 = NodeWorker(
            src_addr="tcp://127.0.0.1:40803",
            dst_addr="tcp://127.0.0.1:40800",
            can_receive_user_request=False,
            shards_path=self.shards_path,
            device=self.device,
            dtype=self.dtype
        )
        node0.load_shards(0, 10)
        node1.load_shards(10, 20)
        node2.load_shards(20, 30)
        node3.load_shards(30, 32)

        data0 = node0.receive_user_request(request="Write a poem about the blue sky.")
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
            reached_eos, data0 = node0.receive_next_token(data4_recv)
            if reached_eos:
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
