import torch
from transformers import T5ForConditionalGeneration, RobertaTokenizer

# Load the pretrained CodeT5 model and its tokenizer
MODEL_NAME = "salesforce/codet5-base"
tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

# Use a natural language prompt for generating code
prompt = "Write a Python function named 'factorial' that takes a single argument 'n' and calculates the factorial of the given number."
input_text = f"codegeneration {prompt}"
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Generate code
output = model.generate(input_ids, max_length=200, num_return_sequences=2, num_beams=10, temperature=0.1)
generated_code = tokenizer.decode(output[0], skip_special_tokens=True)

print("Generated code:")
print(generated_code)