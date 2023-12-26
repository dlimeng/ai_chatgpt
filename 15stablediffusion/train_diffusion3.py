# 模型训练
import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import multiprocessing

from matplotlib import pyplot as plt

"""
用于训练一个基于去噪扩散模型（Denoising Diffusion Probabilistic Model, DDPM）的图像生成模型。这种类型的模型可以通过学习训练数据集中的图像分布来生成新的图像。

加载模型：首先，您需要加载训练好的模型。这通常涉及到加载保存的模型权重。

预处理输入数据：如果您的模型需要特定格式的输入（例如特定大小的图像），您可能需要预处理输入数据以符合这些要求。

生成图像：使用模型对输入数据进行推断，生成新的图像。这通常涉及到将输入数据（在这种情况下可能是噪声）传递给模型，并让模型进行去噪过程以生成图像。

后处理和显示：生成的图像可能需要一些后处理（例如缩放到特定大小，裁剪，或颜色校正）才能正确显示。
"""

if __name__ == '__main__':
    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8)
    ).cuda()

    diffusion = GaussianDiffusion(
        model,
        image_size=128,
        timesteps=1000  # 加噪总步数
    ).cuda()

    model_files = [f"./results/model-{i}.pt" for i in range(1, 11)]

    # 遍历每个模型文件并加载
    for model_file in model_files:
        # 加载模型状态
        checkpoint = torch.load(model_file)
        # 如果模型状态字典包含在更大的字典中的一个条目
        if 'model' in checkpoint:
            model_state_dict = checkpoint['model']
            # 如果键名中包含前缀 'model.'
            model_state_dict = {k.split('model.')[-1]: v for k, v in model_state_dict.items() if 'model.' in k}
        else:
            model_state_dict = checkpoint

        model.load_state_dict(model_state_dict)

        # 将模型设置为评估模式
        model.eval()

    input_noise_shape = [1, 3, 128, 128]
    ##input_noise = torch.randn([1, 3, 128, 128]).cuda()  # 假设图像大小为 128x128，3个颜色通道

    with torch.no_grad():
        generated_images = diffusion.p_sample_loop(input_noise_shape, 1000)  # 使用1000个时间步

    print("Generated image shape:", generated_images.shape)

    # 去除批量维度，并选择最后一个时间步的图像
    final_image = generated_images.squeeze(0)[-1]  # 现在形状应该是 [3, 128, 128]

    print("Final image shape:", final_image.shape)

    # 将图像从 GPU 移到 CPU，并转换为 NumPy 数组
    final_image_np = final_image.cpu().numpy().transpose(1, 2, 0)

    print("Final image np shape:", final_image_np.shape)
    # 调整数据范围并显示图像
    final_image_np = (final_image_np + 1) / 2  # 将数据范围从 [-1, 1] 调整到 [0, 1]
    plt.imshow(final_image_np)
    plt.show()