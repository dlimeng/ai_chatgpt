import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, AdamW

# 准备标注数据
samples = [
    {"text": "I love this movie!", "label": 1},
    {"text": "This film is amazing!", "label": 1},
    {"text": "I didn't like this movie.", "label": 0},
    {"text": "This movie is terrible.", "label": 0},
]

# 创建自定义数据集
class SentimentDataset(Dataset):
    def __init__(self, samples, tokenizer):
        self.samples = samples
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        encoding = self.tokenizer(sample["text"], truncation=True, padding="max_length", max_length=64, return_tensors="pt")
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(sample["label"], dtype=torch.long),
        }

# 初始化tokenizer和预训练模型
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# 创建数据加载器
dataset = SentimentDataset(samples, tokenizer)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# 准备训练
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)

# 微调模型
num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{num_epochs}: Loss = {loss.item()}")

# 保存微调后的模型
model.save_pretrained("sentiment_model")
tokenizer.save_pretrained("sentiment_model")