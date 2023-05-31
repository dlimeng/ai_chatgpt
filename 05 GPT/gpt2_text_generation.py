import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 选择预训练的GPT-2模型
MODEL_NAME = "gpt2-large"  # 可以尝试"gpt2", "gpt2-medium", "gpt2-large"或"gpt2-xl"

# 加载分词器和模型
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)


def generate_text(prompt, max_length=100, num_return_sequences=1, temperature=0.7):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        no_repeat_ngram_size=2,
        temperature=temperature,
    )

    generated_text = [tokenizer.decode(sequence, skip_special_tokens=True) for sequence in output]
    return generated_text


# 提供一个文本提示
prompt_text = "Once upon a time in a land far away, there was a"
generated_texts = generate_text(prompt_text, max_length=150, num_return_sequences=3)

# 输出生成的文本
for i, text in enumerate(generated_texts):
    print(f"Generated text {i + 1}:")
    print(text)
    print()