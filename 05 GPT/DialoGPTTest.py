pip install transformers

from transformers import AutoTokenizer, AutoModelForCausalLM

# 加载预训练模型及其分词器  Microsoft 提供的 "microsoft/DialoGPT-medium" 预训练模型。
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# 对输入文本进行编码 ,只能输入英文
input_text = "hello"
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