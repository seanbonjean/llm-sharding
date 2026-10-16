import os
import torch
import torch.nn as nn
from transformers import LlamaConfig
from transformers.cache_utils import Cache
from transformers.masking_utils import create_causal_mask
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm


class LlamaShardPart(nn.Module):
    """
    加载进来的 某层/某几层 中间层对象
    """

    def __init__(self, shards_path: str, shard_weights: list[str], start: int, end: int,
                 device="cpu", dtype=torch.float32,
                 add_final_norm=False, final_norm_weight=None):
        """
        :param shards_path: 所有切片的保存位置
        :param shard_weights: 想要加载的切片权重文件（列表）
        :param start: 由于shard_weights中已经间接指定了个数，对该变量的使用只有一个层数的断言判断；同时也起到注解作用，方便调试时看
        :param end: 不包括end，用处同上
        :param device: "cpu" 或 "cuda:0" 等
        :param dtype: torch.float32 / torch.float16 等
        :param add_final_norm: 若为最后一层，需要额外加上最后的归一化层
        :param final_norm_weight: 归一化层权重文件
        """
        super().__init__()
        self.shards_path = shards_path
        self.shard_weights = shard_weights
        self.start = start
        self.end = end
        self.device = torch.device(device)
        self.dtype = dtype
        if add_final_norm:
            self.final_norm_weight = final_norm_weight
        self.config = LlamaConfig.from_pretrained(self.shards_path)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(self.config, layer_idx).to(device=self.device, dtype=self.dtype) for layer_idx in
             range(self.end - self.start)])  # 千万注意这里是分片后的相对索引
        # 加载权重
        if len(self.shard_weights) != self.end - self.start:
            raise ValueError("[ERROR] String list: shard_weights length must be equal to (end - start)")
        for relative_layer_idx, layer in enumerate(self.layers):
            state = torch.load(os.path.join(self.shards_path, self.shard_weights[relative_layer_idx]),
                               map_location=self.device)
            layer.load_state_dict(state)
            # self.load_state_dict(state, strict=False)

        # 如果需要最后的归一化层
        if add_final_norm:
            self.final_norm = LlamaRMSNorm(self.config.hidden_size).to(device=self.device, dtype=self.dtype)
            # 加载 final norm 权重
            if self.final_norm_weight is None:
                raise ValueError("[ERROR] final_norm_weight is required when add_final_norm is True")
            norm_state = torch.load(os.path.join(self.shards_path, self.final_norm_weight), map_location=self.device)
            self.final_norm.load_state_dict(norm_state)

    def prepare_attention_mask(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None,
        past_key_value: Cache | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Prepare the same causal mask and cache positions as the full Llama model.

        构造与完整 Llama 模型一致的因果掩码和 KV 写入位置。

        Args:
            self: Shard whose config selects the attention backend.
                当前分片；其配置决定 attention 后端。
            hidden_states: Current [batch, query_length, hidden_size] inputs.
                本轮输入，用于确定 batch、查询长度、设备和精度。
            attention_mask: None for unpadded inputs, a 2D keep/pad mask covering
                the entire cached and current sequence, or an already prepared
                backend-compatible 4D causal mask.
                无 padding 时为 None；二维掩码以 1/0 标记有效/补齐位置，必须覆盖
                历史和当前 token；四维掩码须已包含因果约束并符合所选后端格式。
            past_key_value: This shard's cache before any layer appends current KV.
                本分片在各层写入本轮 KV 前的缓存，用于确定历史长度。

        Returns:
            The backend-specific mask and absolute positions of current tokens.
            后端适用的掩码和本轮 token 的绝对缓存位置；SDPA 可合法返回 None。
        """
        past_length = past_key_value.get_seq_length() if past_key_value is not None else 0
        query_length = hidden_states.shape[1]
        if attention_mask is not None and attention_mask.ndim == 2:
            expected_shape = (hidden_states.shape[0], past_length + query_length)
            if tuple(attention_mask.shape) != expected_shape:
                raise ValueError(f"2D attention_mask must cover past and current tokens: {expected_shape}")
        cache_position = torch.arange(
            past_length, past_length + query_length, device=hidden_states.device,
        )
        causal_mask = create_causal_mask(
            config=self.config, input_embeds=hidden_states, attention_mask=attention_mask,
            cache_position=cache_position, past_key_values=past_key_value,
        )
        return causal_mask, cache_position

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None,
        past_key_value: Cache | None = None,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Run all shard layers with causal attention and optional final norm.

        Args:
            self: Loaded shard providing layers, mask preparation and optional norm.
                已加载的分片，提供模型层、掩码构造及可选最终归一化。
            hidden_states: Current [batch, query_length, hidden_size] inputs.
                本轮待处理的 hidden states。
            attention_mask: Padding or prepared causal mask accepted by prepare_attention_mask.
                传给 prepare_attention_mask 的 padding 或已准备因果掩码。
            past_key_value: Per-shard KV cache updated by each decoder layer.
                本分片的 KV 缓存，由各 decoder layer 更新。
            rotary_emb: Precomputed cosine/sine tensors for current absolute positions.
                按当前绝对位置预先计算的 RoPE cos/sin 张量。
        """
        # Snapshot positions once: layer 0 mutates the shared cache during this traversal.
        # 位置和掩码只构造一次；第 0 层会在本轮遍历中更新共享缓存的长度。
        causal_mask, cache_position = self.prepare_attention_mask(
            hidden_states, attention_mask=attention_mask, past_key_value=past_key_value,
        )
        for layer in self.layers:
            outputs = layer(
                hidden_states=hidden_states,
                attention_mask=causal_mask,
                past_key_value=past_key_value,
                cache_position=cache_position,
                position_embeddings=rotary_emb,
                use_cache=True
            )
            hidden_states = outputs[0]
        # 如果有最后的归一化层
        if hasattr(self, "final_norm"):
            hidden_states = self.final_norm(hidden_states)
        return hidden_states


if __name__ == '__main__':
    # 加载一层
    shard_block_0 = LlamaShardPart(
        "../shards/Llama-2-7b-chat-hf_float16",
        ["block_0.pth"],
        0, 1,
        device="cpu",
        dtype=torch.float16,
        add_final_norm=False,
        final_norm_weight=None
    )
    print(shard_block_0)
    # print(shard_block_0.layers[0].self_attn.__dict__.keys())
    # print(shard_block_0.layers[0].self_attn.config)
    # print(shard_block_0.layers[0].self_attn.layer_idx)
    # print(shard_block_0.layers[0].self_attn.head_dim)
    # print(shard_block_0.layers[0].self_attn.scaling)
    # print(shard_block_0.layers[0].self_attn.attention_dropout)
    # 加载多层
    shard_block_2_4 = LlamaShardPart(
        "../shards/Llama-2-7b-chat-hf_float16",
        ["block_2.pth", "block_3.pth", "block_4.pth"],
        2, 5,  # 不包括end
        device="cpu",
        dtype=torch.float16,
        add_final_norm=False,
        final_norm_weight=None
    )
    print(shard_block_2_4)
    # 加载最后一层
    shard_block_31 = LlamaShardPart(
        "../shards/Llama-2-7b-chat-hf_float16",
        ["block_31.pth"],
        31, 32,
        device="cpu",
        dtype=torch.float16,
        add_final_norm=True,
        final_norm_weight="final_norm.pth"
    )
    print(shard_block_31)
