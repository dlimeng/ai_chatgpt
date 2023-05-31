import torch
from transformers import AutoTokenizer, AutoModelWithLMHead

# 加载预训练的CodeBERT模型及其分词器
MODEL_NAME = "microsoft/codebert-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelWithLMHead.from_pretrained(MODEL_NAME)

# 使用自然语言提示生成代码
prompt = "Write a Python function to calculate the factorial of a given number."
input_ids = tokenizer.encode(prompt, return_tensors="pt")

# 生成代码
output = model.generate(input_ids, max_length=100, num_return_sequences=1, temperature=0.7)
generated_code = tokenizer.decode(output[0], skip_special_tokens=True)

print("Generated code:")
print(generated_code)