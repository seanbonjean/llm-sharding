import os
import torch
from transformers import LlamaConfig, AutoTokenizer
from transformers.cache_utils import DynamicCache
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

from utils.shard_loader import LlamaShardPart


def build_position_ids(past_key_value, seq_len, device, batch_size: int = 1):
    """
    past_key_value:
      - HF Cache:  可用 past_key_value.get_seq_length()
      - (k, v)元组: 用 k.shape[-2] 作为已缓存长度 (k: [B, n_kv, past_len, Hd])
      - None: past_len = 0
    返回:
      position_ids: [B, S] LongTensor
    """
    if past_key_value is None:
        past_len = 0
    elif hasattr(past_key_value, "get_seq_length"):
        past_len = int(past_key_value.get_seq_length())
    elif isinstance(past_key_value, tuple) and len(past_key_value) == 2:
        k = past_key_value[0]
        past_len = int(k.shape[-2])
    else:
        # 如果你自定义了 cache 结构，这里替换成正确的取法
        raise ValueError("Unsupported past_key_value structure")

    pos = torch.arange(past_len, past_len + seq_len, device=device, dtype=torch.long)  # [S]
    position_ids = pos.unsqueeze(0).expand(batch_size, -1).contiguous()  # [B, S]
    return position_ids


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
        # self.num_heads = self.config.num_attention_heads
        # self.hidden_size = self.config.hidden_size
        # self.head_dim = self.hidden_size // self.num_heads

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

        hidden_states = embed_tokens(input_ids)  # [B, S, H]
        batch_size, seq_len, _ = hidden_states.shape

        # 初始化 KV cache
        # past_key_values = [None] * self.layer_num
        past_key_values = [DynamicCache() for _ in range(self.layer_num)]

        # 旋转位置编码（RoPE）的 cos/sin 表
        position_ids = build_position_ids(past_key_values[0], seq_len, device=self.device, batch_size=batch_size)
        rope = LlamaRotaryEmbedding(config=self.config, device=self.device).to(self.device)
        cos, sin = rope(hidden_states, position_ids)  # [B, S, Hd] each; hidden_states 只是用作参考张量 x
        # cos = cos[:, :, :seq_len, :].to(device=self.device, dtype=self.dtype)
        # sin = sin[:, :, :seq_len, :].to(device=self.device, dtype=self.dtype)
        # print(cos.shape)
        # print(sin.shape)

        # 逐层跑每个 shard
        for i, shard in enumerate(self.shards):
            hidden_states = hidden_states.to(device=self.device, dtype=self.dtype)
            hidden_states, past = shard(
                hidden_states,
                # attention_mask=inputs["attention_mask"],
                past_key_values=past_key_values[i],
                rotary_emb=(cos, sin)
            )
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
