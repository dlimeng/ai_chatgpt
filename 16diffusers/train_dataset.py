import torch
from datasets import load_dataset

## 数据集的使用。通过 datasets 工具包加载数据集

# 加载数据集
config.dataset_name = "huggan/smithsonian_butterflies_subset"
dataset = load_dataset(config.dataset_name, split="train")

# 封装成训练用的格式
train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)