# 模型训练
import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import multiprocessing

"""
用于训练一个基于去噪扩散模型（Denoising Diffusion Probabilistic Model, DDPM）的图像生成模型。这种类型的模型可以通过学习训练数据集中的图像分布来生成新的图像。

加载模型：首先，您需要加载训练好的模型。这通常涉及到加载保存的模型权重。

预处理输入数据：如果您的模型需要特定格式的输入（例如特定大小的图像），您可能需要预处理输入数据以符合这些要求。

生成图像：使用模型对输入数据进行推断，生成新的图像。这通常涉及到将输入数据（在这种情况下可能是噪声）传递给模型，并让模型进行去噪过程以生成图像。

后处理和显示：生成的图像可能需要一些后处理（例如缩放到特定大小，裁剪，或颜色校正）才能正确显示。
"""

if __name__ == '__main__':
    multiprocessing.freeze_support()

    model = Unet(
        dim=64,  # 模型中特征图的通道数大小,也就是embedding size。这控制了模型的宽度。
        dim_mults=(1, 2, 4, 8)  # 每个resolution下特征图相对dim的扩大倍数。Unet通过下采样减少resolution来扩大感受野。这控制了模型的深度。
    ).cuda()

    diffusion = GaussianDiffusion(
        model,
        image_size=128,
        timesteps=1000  # 加噪总步数
    ).cuda()

    trainer = Trainer(
        diffusion,
        './oxford-datasets/raw-images',
        train_batch_size=16,
        train_lr=2e-5,
        train_num_steps=20000,  # 总共训练20000步
        gradient_accumulate_every=2,  # 梯度累积步数
        ema_decay=0.995,  # 指数滑动平均decay参数
        amp=True,  # 使用混合精度训练加速
        calculate_fid=False,  # 我们关闭FID评测指标计算（比较耗时）。FID用于评测生成质量。
        save_and_sample_every=2000  # 每隔2000步保存一次模型
    )

    trainer.train()