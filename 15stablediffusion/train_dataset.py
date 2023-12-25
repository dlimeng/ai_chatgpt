# 数据集下载
from PIL import Image
from io import BytesIO
from datasets import load_dataset
import os
from tqdm import tqdm

dataset = load_dataset("nelorth/oxford-flowers")

# 创建一个用于保存图片的文件夹
images_dir = "./oxford-datasets/raw-images"
os.makedirs(images_dir, exist_ok=True)

# 遍历所有图片并保存，针对oxford-flowers，整个过程要持续15分钟左右
for split in dataset.keys():
    for index, item in enumerate(tqdm(dataset[split])):
        image = item['image']
        image.save(os.path.join(images_dir, f"{split}_image_{index}.png"))