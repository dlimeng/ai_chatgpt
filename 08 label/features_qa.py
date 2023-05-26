from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch

# 加载微调后的模型和tokenizer
model = DistilBertForSequenceClassification.from_pretrained("sentiment_model")
tokenizer = DistilBertTokenizerFast.from_pretrained("sentiment_model")

# 将模型设置为评估模式
model.eval()

# 输入要预测的文本
text = "I really enjoyed this movie."

# 使用tokenizer对输入文本进行编码
inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt")

# 使用模型进行预测
with torch.no_grad():
    outputs = model(**inputs)

# 获取预测概率
logits = outputs.logits
probs = torch.softmax(logits, dim=-1)

# 获取预测标签
predicted_label = torch.argmax(probs).item()

# 输出预测结果
if predicted_label == 1:
    print("Positive sentiment")
else:
    print("Negative sentiment")