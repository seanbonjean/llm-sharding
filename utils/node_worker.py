import os
import gc
import torch
from transformers import LlamaConfig, AutoTokenizer
from transformers.cache_utils import DynamicCache
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

from utils.shard_loader import LlamaShardPart
from utils.forwarding_utils import build_position_ids


def transfer_data(data: torch.Tensor | dict, save_path="results/data.pt") -> str:
    torch.save(data, save_path)
    return save_path


def receive_data(data_path: str) -> torch.Tensor | dict:
    data = torch.load(data_path)
    return data


class NodeWorker:
    # 每个 node 上运行的 client
    def __init__(self, can_receive_user_request: bool, shards_path: str, device="cpu", dtype=torch.float16):
        """
        :param can_receive_user_request: 是否接收用户请求（关系到是否加载分词器、嵌入层）
        :param shards_path: 切片路径
        :param device: "cpu" 或 "cuda:0" 等
        :param dtype: torch.float32 / torch.float16 等
        """
        self.can_receive_user_request = can_receive_user_request
        self.shards_path = shards_path
        self.device = torch.device(device)
        self.dtype = dtype
        self.config = LlamaConfig.from_pretrained(self.shards_path)
        self.layer_num = self.config.num_hidden_layers
        # self.num_heads = self.config.num_attention_heads
        # self.hidden_size = self.config.hidden_size
        # self.head_dim = self.hidden_size // self.num_heads

        self.tokenizer = None  # 分词器
        self.embed_tokens = None  # 嵌入层
        self.rope = None  # 旋转位置编码（RoPE）
        self.shard = None  # （对应分片）隐藏层权重
        self.past_key_value = None  # KV cache
        self.lm_head = None

        # 运行时的缓存数据
        self.start = 0
        self.end = 0
        if can_receive_user_request:
            self.batch_size = 0
            self.generated_ids = []
            self._load_embedding()

    def _load_embedding(self):
        """
        加载嵌入层
        :return: None
        """
        print("[INFO] loading tokenizer...")
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(self.shards_path)
        print("[INFO] tokenizer loaded.")
        # 加载嵌入层
        print("[INFO] loading embedding layer...")
        self.embed_tokens = torch.nn.Embedding(
            self.config.vocab_size, self.config.hidden_size).to(self.device, dtype=self.dtype)
        self.embed_tokens.load_state_dict(
            torch.load(os.path.join(self.shards_path, "embedding.pth"), map_location=self.device))
        print("[INFO] embedding layer loaded.")

    def load_shards(self, start: int, end: int):
        """
        加载切片并删除旧切片（如有），从start到end（不包括end）
        :param start:
        :param end:
        :return: None
        """
        if start < 0 or start >= end or end > self.layer_num:
            raise ValueError("[ERROR] start or end is invalid")
        self.start = start
        self.end = end

        try:
            del self.rope
            del self.shard
            del self.lm_head
        except AttributeError:
            pass
        gc.collect()  # 强制 Python 做一次垃圾回收
        if self.device.type == "cuda":
            torch.cuda.empty_cache()  # 清空 CUDA 缓存池，让 nvidia-smi 立刻下降

        if self.start == 0:
            # 加载旋转位置编码（RoPE）
            print("[INFO] loading RoPE...")
            self.rope = LlamaRotaryEmbedding(config=self.config, device=self.device).to(self.device)
            print("[INFO] RoPE loaded.")

        if self.end == self.layer_num:
            add_final_norm = True
            final_norm_weight = "final_norm.pth"
            # 加载 lm_head
            print("[INFO] loading lm_head...")
            self.lm_head = torch.nn.Linear(
                self.config.hidden_size, self.config.vocab_size, bias=False).to(self.device, dtype=self.dtype)
            self.lm_head.load_state_dict(
                torch.load(os.path.join(self.shards_path, "lm_head.pth"), map_location=self.device))
            print("[INFO] lm_head loaded.")
        else:
            add_final_norm = False
            final_norm_weight = None

        print(f"[INFO] loading hidden layer {start}~{end}(end excluded)...")
        self.shard = LlamaShardPart(
            self.shards_path,
            ["block_" + str(i) + ".pth" for i in range(start, end)],
            start, end,
            device=self.device,
            dtype=self.dtype,
            add_final_norm=add_final_norm,
            final_norm_weight=final_norm_weight
        )
        self.shard.eval()
        print(f"[INFO] hidden layer {start}~{end}(end excluded) loaded.")

        # 初始化 KV cache
        print("[INFO] loading KV cache...")
        self.past_key_value = DynamicCache()
        print("[INFO] KV cache loaded.")

    def receive_user_request(self, request="Write a poem about the blue sky.") -> dict:
        """
        接收用户请求
        :return: input_token_info: dict，包含隐藏层参数和用于 RoPE 的 batch_size & seq_len
        """
        if not self.can_receive_user_request:
            raise RuntimeError("[ERROR] this node does not have embedding layer while receiving user request.")

        input_text = request
        print("[INFO] input: " + input_text)

        # 分词器tokenize
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]  # 初始 prompt 张量化后的 token id 序列
        self.generated_ids = [input_ids]  # 存储所有 input 和 output 的 token id 序列

        # 经过嵌入层
        hidden_states = self.embed_tokens(input_ids)  # [B, S, H]
        batch_size, seq_len, _ = hidden_states.shape
        self.batch_size = batch_size

        input_token_info = {
            "hidden_states": hidden_states,
            "batch_size": batch_size,
            "seq_len": seq_len,
        }
        return input_token_info

    def pass_through_shard(self, state_info: dict) -> torch.Tensor | dict:
        """
        对隐藏层前向传播
        :param state_info: 传递hidden_states。若存在batch_size和seq_len：先计算RoPE的cos,sin；若存在cos,sin：直接传入shard
        :return: next_token_id: torch.Tensor | next_state_info: dict
        """
        if "batch_size" in state_info and "seq_len" in state_info:
            if self.start != 0:
                raise RuntimeError("[ERROR] after embedding layer, the states should first passing hidden layer 0!")
            # 旋转位置编码（RoPE）获取 cos/sin 表
            position_ids = build_position_ids(self.past_key_value, state_info["seq_len"], device=self.device,
                                              batch_size=state_info["batch_size"])
            cos, sin = self.rope(state_info["hidden_states"], position_ids)  # [B, S, Hd] each; hidden_states 只是用作参考张量 x
        else:
            cos, sin = state_info["cos"], state_info["sin"]
        hidden_states = state_info["hidden_states"]

        hidden_states = hidden_states.to(device=self.device, dtype=self.dtype)
        next_hidden_states = self.shard(
            hidden_states,
            # attention_mask=inputs["attention_mask"],
            past_key_value=self.past_key_value,
            rotary_emb=(cos, sin)
        )

        if self.end == self.layer_num:
            # lm_head 输出 logits
            logits = self.lm_head(next_hidden_states)  # [batch, seq_len, vocab_size]
            # 取最后一个位置的预测 token
            next_token_id = torch.argmax(logits[:, -1, :], dim=-1)  # [1]
            return next_token_id
        else:
            next_state_info = {
                "hidden_states": next_hidden_states,
                "cos": cos,
                "sin": sin,
            }
            return next_state_info

    def receive_next_token(self, next_token_id: torch.Tensor) -> tuple:
        """
        接收最后一层的输出token id
        :param next_token_id:
        :return: (reached_eos: bool, input_token_info: dict|None): tuple 如果reached_eos=True，则input_token_info=None
        """
        if not self.can_receive_user_request:
            raise RuntimeError(
                '[ERROR] this node does not store the "generated token id" list, but received a next_token_id.')
        # 拼接到已有序列
        self.generated_ids.append(next_token_id.unsqueeze(-1))
        # 解码
        next_token = self.tokenizer.decode(next_token_id.item())
        print(repr(next_token), end=" ", flush=True)

        reached_eos = (next_token == self.tokenizer.eos_token)
        if reached_eos:
            # 解码最终结果
            print()
            final_ids = torch.cat(self.generated_ids, dim=-1)  # [B, seq_len + out_token_num]
            print("output: ", self.tokenizer.decode(final_ids[0]))
            return reached_eos, None
        else:
            # 更新输入
            hidden_states = self.embed_tokens(next_token_id.unsqueeze(0))  # [B, 1, H]
            seq_len = 1
            input_token_info = {
                "hidden_states": hidden_states,
                "batch_size": self.batch_size,
                "seq_len": seq_len,
            }
            return reached_eos, input_token_info
