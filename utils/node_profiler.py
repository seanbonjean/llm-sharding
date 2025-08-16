import os
import torch
from transformers import LlamaConfig, AutoTokenizer

from utils.shard_loader import LlamaShardPart


def build_rotary_emb(dim: int, max_position_embeddings: int = 2048, base: int = 10000, device=None, dtype=None):
    """
    手动构建 LLaMA 用的 rotary embedding (cos/sin 表)

    参数:
        dim (int): 每个 head 的 hidden dimension (通常是 head_dim)
        max_position_embeddings (int): 支持的最大序列长度
        base (int): RoPE 的 base，默认 10000（LLaMA 使用 10000）
        device: torch.device，默认当前
        dtype: torch.dtype，默认 torch.get_default_dtype()

    返回:
        cos, sin: shape = [max_position_embeddings, dim]
    """
    if device is None:
        device = torch.device("cpu")
    if dtype is None:
        dtype = torch.get_default_dtype()

    # rotary 只对偶数维度生效，所以一半用来算频率
    half_dim = dim // 2
    # [half_dim]
    inv_freq = 1.0 / (base ** (torch.arange(0, half_dim, device=device, dtype=dtype) / half_dim))

    # [max_position_embeddings, half_dim]
    positions = torch.arange(max_position_embeddings, device=device, dtype=dtype)
    freqs = torch.einsum("i,j->ij", positions, inv_freq)  # outer product

    # [max_position_embeddings, dim]
    emb = torch.cat([freqs, freqs], dim=-1)

    cos = emb.cos()[None, :, None, :]
    sin = emb.sin()[None, :, None, :]

    return cos, sin


class NodeProfiler:
    def __init__(self, shards_path: str, device="cpu", dtype=torch.float16):
        """
        :param shards_path: 切片路径（不用带dtype，会自动标明）
        :param device: "cpu" 或 "cuda:0" 等
        :param dtype: torch.float32 / torch.float16 等
        """
        self.shards_path = shards_path
        self.shards_path_full = shards_path + "_" + str(dtype).split(".")[-1]
        self.device = torch.device(device)
        self.dtype = dtype
        self.config = LlamaConfig.from_pretrained(self.shards_path_full)
        self.layer_num = self.config.num_hidden_layers

        self.shards = []  # 保存各层权重

    def go_through_every_shards(self):
        # 加载所有分片
        self.shards = [LlamaShardPart(
            self.shards_path,  # 用的不带dtype的路径，因为shardpart也会添加dtype
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
        for shard in self.shards:
            shard.eval()

        # 分词器
        tokenizer = AutoTokenizer.from_pretrained(self.shards_path_full)
        input_text = "Hello, this is a test for edge inference on sharded LLaMA."
        inputs = tokenizer(input_text, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]  # 初始 prompt 张量化后的 token id 序列

        # 嵌入层
        embed_tokens = torch.nn.Embedding(
            self.config.vocab_size, self.config.hidden_size).to(self.device, dtype=self.dtype)
        embed_tokens.load_state_dict(
            torch.load(os.path.join(self.shards_path_full, "embedding.pth"), map_location=self.device))

        hidden_states = embed_tokens(input_ids)

        # 初始化 KV cache
        past_key_values = [None] * self.layer_num

        # 旋转位置编码（RoPE）的 cos/sin 表
        seq_len = hidden_states.shape[1]
        # seq_len = 32
        cos, sin = build_rotary_emb(self.config.hidden_size // self.config.num_attention_heads,
                                    max_position_embeddings=self.config.max_position_embeddings,
                                    device=self.device, dtype=self.dtype)
        cos, sin = cos[:, :seq_len, :, :], sin[:, :seq_len, :, :]
        print(cos.shape)
        print(sin.shape)

        # 逐层跑每个 shard
        for i, shard in enumerate(self.shards):
            hidden_states = hidden_states.to(device=self.device, dtype=self.dtype)
            hidden_states, past = shard(hidden_states, attention_mask=inputs["attention_mask"],
                                        past_key_values=past_key_values[i],
                                        rotary_emb=(cos, sin))
            past_key_values[i] = past

        # 最后的 norm (此处不需要，因为 LlamaShardPart 的 forward 函数重载中，已经包含 final norm 的检测和执行了)
        # hidden_states = self.shards[-1].norm(hidden_states)

        # lm_head 输出 logits
        lm_head = torch.nn.Linear(
            self.config.hidden_size, self.config.vocab_size, bias=False).to(self.device, dtype=self.dtype)
        lm_head.load_state_dict(
            torch.load(os.path.join(self.shards_path_full, "lm_head.pth"), map_location=self.device))
        logits = lm_head(hidden_states)  # [batch, seq_len, vocab_size]

        # 取最后一个位置的预测 token
        next_token_id = torch.argmax(logits[:, -1, :], dim=-1)  # [1]

        # 拼接到已有序列
        input_ids = torch.cat([input_ids, next_token_id.unsqueeze(-1)], dim=-1)

        # 解码
        print(tokenizer.decode(next_token_id.item()))


if __name__ == "__main__":
    profiler = NodeProfiler(
        "../shards/Llama-2-7b-chat-hf",
        device="cuda:0",
        dtype=torch.float32
    )
    profiler.go_through_every_shards()
