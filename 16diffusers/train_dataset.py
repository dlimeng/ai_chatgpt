import os

import torch
from datasets import load_dataset
from dataclasses import dataclass

from torchvision import transforms
import matplotlib.pyplot as plt
from diffusers import UNet2DModel
from diffusers import DDPMScheduler
import torch
from PIL import Image
import torch.nn.functional as F

from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers import DDPMPipeline

import math

from accelerate import Accelerator
from huggingface_hub import HfFolder, Repository, whoami

from tqdm.auto import tqdm
from pathlib import Path


@dataclass
class TrainingConfig:
    image_size = 128  # 生成图像的分辨率
    train_batch_size = 16
    eval_batch_size = 16  # 在模型评估阶段，生成图片的数量
    num_epochs = 50
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 10
    save_model_epochs = 30
    mixed_precision = 'fp16'  # 设置为`no`则使用float32训练, 设置为`fp16`则使用混合精度训练（automatic mixed precision），降低计算开销
    output_dir = 'ddpm-butterflies-128'  # 模型保存路径

    push_to_hub = False  # 模型是否上传到你的Hugging Face账号
    hub_private_repo = False
    overwrite_output_dir = True  # 当你重新运行jupyter时，会覆盖已经训练完成的旧模型
    seed = 0


config = TrainingConfig()

## 数据集的使用。通过 datasets 工具包加载数据集

# 加载数据集
config.dataset_name = "huggan/smithsonian_butterflies_subset"
dataset = load_dataset(config.dataset_name, split="train")

## 查看数据集
# fig, axs = plt.subplots(1, 4, figsize=(16, 4))
# for i, image in enumerate(dataset[:4]["image"]):
#     axs[i].imshow(image)
#     axs[i].set_axis_off()
# fig.show()

# 封装成训练用的格式
train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)

### 数据增广后的效果
preprocess = transforms.Compose(
    [
        transforms.Resize((config.image_size, config.image_size)),
        transforms.RandomHorizontalFlip(),  # 训练的时候图片左右随机翻转
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)


def transform(examples):
    images = [preprocess(image.convert("RGB")) for image in examples["image"]]
    return {"images": images}


dataset.set_transform(transform)

## 数据增广后作用
# fig, axs = plt.subplots(1, 4, figsize=(16, 4))
# for i, image in enumerate(dataset[:4]["images"]):
#     axs[i].imshow(image.permute(1, 2, 0).numpy() / 2 + 0.5)
#     axs[i].set_axis_off()
# fig.show()


## 将数据集构造为Pytorch可以使用的Dataloader


model = UNet2DModel(
    sample_size=config.image_size,  # 目标图像分辨率
    in_channels=3,  # 对于RGB图像，输入和输出都是3通道
    out_channels=3,  # 对于RGB图像，输入和输出都是3通道
    layers_per_block=2,  # 每个UNet模块中ResNet的层数
    block_out_channels=(128, 128, 256, 256, 512, 512),  # 各个UNet模块的输出通道数
    down_block_types=(
        "DownBlock2D",  # 常规的ResNet降采样模块
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",  # 带自注意力机制的ResNet降采样模块
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",  # 常规的ResNet上采样模块
        "AttnUpBlock2D",  # 带自注意力机制的ResNet上采样模块
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D"
    ),
)

## 检查输入数据维度
sample_image = dataset[0]['images'].unsqueeze(0)
print('Input shape:', sample_image.shape)

## 检查输出结果维度
print('Output shape:', model(sample_image, timestep=0).sample.shape)

## 需要确定我们加噪用的采样器，帮助我们通过一步计算得到第t步的加噪结果。
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

## 我们使用DDPM采样器，希望得到第50步加噪之后的结果。
noise = torch.randn(sample_image.shape)
timesteps = torch.LongTensor([50])
noisy_image = noise_scheduler.add_noise(sample_image, noise, timesteps)

Image.fromarray(((noisy_image.permute(0, 2, 3, 1) + 1.0) * 127.5).type(torch.uint8).numpy()[0])

### 损失函数计算方式

noise_pred = model(noisy_image, timesteps).sample
loss = F.mse_loss(noise_pred, noise)

## 选择模型的优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

## 设置学习率

lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)


## 在模型训练的过程中，需要对模型中间效果进行评估。我们可以采样一批图片。
def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid


def evaluate(config, epoch, pipeline):
    # 从随机噪声出发进行图像生成
    # pipeline的输出结果类型为：`List[PIL.Image]`
    images = pipeline(
        batch_size=config.eval_batch_size,
        generator=torch.manual_seed(config.seed),
    ).images

    # 将图片拼接成4x4的网格形状
    image_grid = make_grid(images, rows=4, cols=4)

    # 保存图像
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")


def get_full_repo_name(model_id: str, organization: str = None, token: str = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    # Accelerator可以加速扩散模型的训练
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs")
    )
    if accelerator.is_main_process:
        if config.push_to_hub:
            repo_name = get_full_repo_name(Path(config.output_dir).name)
            repo = Repository(config.output_dir, clone_from=repo_name)
        elif config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers("train_example")

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0

    # 开始训练模型
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch['images']
            # 随机采样噪声
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]

            # 随机选择时间步t
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bs,), device=clean_images.device).long()

            # 加噪过程，一步计算加噪结果
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                # 预测噪声
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        # 每一个训练轮次后，调用评估函数，生成当前轮次模型的图片效果
        if accelerator.is_main_process:
            pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                evaluate(config, epoch, pipeline)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                if config.push_to_hub:
                    repo.push_to_hub(commit_message=f"Epoch {epoch}", blocking=True)
                else:
                    pipeline.save_pretrained(config.output_dir)


## 启动训练
from accelerate import notebook_launcher

args = (config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)

notebook_launcher(train_loop, args, num_processes=1)
