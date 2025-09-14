import os
import torch
import torch.nn as nn
from transformers import LlamaConfig
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

    def forward(self, hidden_states, attention_mask=None, past_key_value=None, rotary_emb=None):
        """
        计算并返回hidden state
        :param hidden_states:
        :param attention_mask:
        :param past_key_value: 传入的KV cache
        :param rotary_emb: 旋转位置编码（RoPE）
        :return:
        """
        for layer in self.layers:
            outputs = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
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
