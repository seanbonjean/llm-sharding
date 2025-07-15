from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 加载 tokenizer 和模型（自动从本地路径加载）
model_path = "./weights/llama-2-7b-hf"  # 你的模型下载目录
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")

# 模型推理（单轮问答）
prompt = "Tell me about the Great Wall of China."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=200, do_sample=True, top_p=0.95, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
