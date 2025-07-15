from transformers import AutoTokenizer, AutoModelForCausalLM
# from transformers import LlamaTokenizer, LlamaForCausalLM
import torch

prompt = "Tell me about the Great Wall of China."

model_path = "./weights/Llama-3___2-1B"
model_id = model_path

# 加载tokenizer和model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,  # 也可以是int8/int4量化
    device_map="auto",
)
# tokenizer = LlamaTokenizer.from_pretrained(model_id)
# model = LlamaForCausalLM.from_pretrained(
#     model_id,
#     torch_dtype=torch.float16,  # 也可以是int8/int4量化
#     device_map="auto",
# )

# tokenize
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# 推理
outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    do_sample=True,
    top_p=0.95,
    temperature=0.7,
    eos_token_id=tokenizer.eos_token_id
)

# 输出
output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(output_text)
