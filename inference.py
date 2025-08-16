from transformers import AutoTokenizer, AutoModelForCausalLM
# from transformers import LlamaTokenizer, LlamaForCausalLM
import torch

# prompt = "介绍一下边缘计算。"

# 非instruct版本（不是对话专用），需要使用ChatML格式prompt
prompt = "<|begin_of_text|><|user|>\n介绍一下边缘计算。\n<|assistant|>\n"

model_path = "./weights/llama-2-7b-hf"
# model_path = "./weights/Llama-3___2-3B"

# 加载tokenizer和model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,  # 也可以是int8/int4量化
    device_map="auto",
).to("cuda")  # 放置到GPU上

# tokenizer = LlamaTokenizer.from_pretrained(model_path)
# model = LlamaForCausalLM.from_pretrained(
#     model_path,
#     torch_dtype=torch.float16,  # 也可以是int8/int4量化
#     device_map="auto",
# )

# tokenize
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# 推理
outputs = model.generate(
    **inputs,
    max_new_tokens=1024,
    do_sample=True,
    top_p=0.95,
    temperature=0.7,
    eos_token_id=tokenizer.eos_token_id,
    repetition_penalty=1.2,  # 惩罚重复
)

# 输出
output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(output_text)
