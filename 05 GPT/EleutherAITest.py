from transformers import AutoTokenizer, AutoModelForCausalLM

# 加载预训练模型及其分词器  当然可以。我为您提供了一个使用 Hugging Face Model Hub 中的 EleutherAI/gpt-neo-1.3B 模型的示例代码。这个模型是一个基于 GPT-3 架构的大型开源模型，性能相当不错。
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")

# 对输入文本进行编码
input_text = "Hello, can you tell me about ChatGPT?"
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# 生成回应
output = model.generate(
    input_ids,
    max_length=100,
    num_return_sequences=1,
    pad_token_id=tokenizer.eos_token_id,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=0.7,
)

# 对输出进行解码并打印
response = tokenizer.decode(output[0], skip_special_tokens=True)
print(response)