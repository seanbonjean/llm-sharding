import os
import shutil
import torch
from transformers import AutoModelForCausalLM


class ModelSharder:
    """
    模型切割器，将模型按层切开
    """

    def __init__(self, model_path: str, model_type: str, shard_save_folder: str, device="cpu", dtype=torch.float32):
        """
        :param model_path: 模型路径
        :param model_type: 模型类型 "llama" 或 "gpt" 等
        :param shard_save_folder: 保存分片的文件夹（不用带dtype，会自动标明）
        :param device: "cpu" 或 "cuda:0" 等
        :param dtype: torch.float32 / torch.float16 等
        """
        self.model_path = model_path
        self.model_type = model_type
        self.device = torch.device(device)
        self.dtype = dtype
        self.shard_save_folder = shard_save_folder + "_" + str(dtype).split(".")[-1]
        os.makedirs(self.shard_save_folder, exist_ok=True)

        print(f"Loading full model {self.model_path} to {self.device} ...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self.dtype,
            device_map=self.device,
        )
        # self.model.to(self.device)

    def save_shards(self):
        # 保存 config.json 和 tokenizer 所需文件
        for filename in os.listdir(self.model_path):
            # 跳过权重文件
            if filename.endswith((".bin", ".safetensors")):
                continue
            src = os.path.join(self.model_path, filename)
            dst = os.path.join(self.shard_save_folder, filename)
            # 只复制文件，忽略文件夹
            if os.path.isfile(src):
                shutil.copyfile(src, dst)
                print(f"Copied {filename} -> {dst}")

        if self.model_type == "llama":
            # LLaMA 结构解析
            embed_tokens = self.model.model.embed_tokens  # 词嵌入层
            layers = list(self.model.model.layers)  # Transformer block 列表
            final_layer_normalization = self.model.model.norm  # 最后 LayerNorm
            language_modeling_head = self.model.lm_head  # LM Head

            # 保存 embedding 层（LLaMA 没有位置 embedding）
            torch.save(embed_tokens.state_dict(),
                       os.path.join(self.shard_save_folder, "embedding.pth"))
            print("Saved embedding layer.")

            # 保存每一层 Transformer block
            for i, layer in enumerate(layers):
                shard_path = os.path.join(self.shard_save_folder, f"block_{i}.pth")
                torch.save(layer.state_dict(), shard_path)
                print(f"Saved block {i} → {shard_path}")
                del layer
                torch.cuda.empty_cache()

            # 保存最后 LayerNorm 和 LM Head
            torch.save(final_layer_normalization.state_dict(),
                       os.path.join(self.shard_save_folder, "final_norm.pth"))
            torch.save(language_modeling_head.state_dict(),
                       os.path.join(self.shard_save_folder, "lm_head.pth"))
            print("Saved final normalization and lm_head.")

            # 释放模型
            del self.model
            torch.cuda.empty_cache()
            print("Sharding complete.")

        elif self.model_type == "gpt":
            # GPT-2 模型结构解析
            # 以下所有层均按照模型结构顺序排列
            word_token_embedding = self.model.transformer.wte  # 词嵌入层
            position_embedding = self.model.transformer.wpe  # 位置嵌入层
            dropout = self.model.transformer.drop  # dropout层
            layers = list(self.model.transformer.h.children())  # 所有 Transformer 中间层 (block)
            final_layer_normalization = self.model.transformer.ln_f  # 最后的归一化层，对最后的 hidden states 做归一化
            language_modeling_head = self.model.lm_head  # 语言建模头，对 hidden states 做线性映射，把 Transformer 的隐藏向量映射到词表大小（vocab_size）的 logits

            # 保存 embedding 层
            torch.save({
                "wte": word_token_embedding.state_dict(),
                "wpe": position_embedding.state_dict(),
                "drop": dropout.state_dict(),
            }, os.path.join(self.shard_save_folder, "embedding.pth"))
            print("Saved embedding layer.")

            # 保存每一层 transformer block
            for i, layer in enumerate(layers):
                shard_path = os.path.join(self.shard_save_folder, f"block_{i}.pth")
                torch.save(layer.state_dict(), shard_path)
                print(f"Saved block {i} → {shard_path}")
                del layer
                torch.cuda.empty_cache()

            # 保存最后的归一化和 lm_head
            torch.save(final_layer_normalization.state_dict(),
                       os.path.join(self.shard_save_folder, "ln_f.pth"))
            torch.save(language_modeling_head.state_dict(),
                       os.path.join(self.shard_save_folder, "lm_head.pth"))
            print("Saved final normalization and lm_head.")

            # 释放大模型
            del self.model
            torch.cuda.empty_cache()
            print("Sharding complete.")
        else:
            raise ValueError(f"[ERROR] Unsupported model type: {self.model_type}")


if __name__ == "__main__":
    sharder = ModelSharder(
        "../weights/Llama-2-7b-chat-hf",
        "llama",
        "../shards/Llama-2-7b-chat-hf",
        device="cuda:0",
        dtype=torch.float32
    )
    sharder.save_shards()
