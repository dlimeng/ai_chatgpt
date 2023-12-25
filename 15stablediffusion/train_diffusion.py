# 创建UNet和扩散模型

from denoising_diffusion_pytorch import Unet, GaussianDiffusion
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torchvision.transforms as transforms

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8)
).cuda()

diffusion = GaussianDiffusion(
    model,
    image_size = 128,
    timesteps = 1000   # number of steps
).cuda()


# 使用随机初始化的图片进行一次训练
training_images = torch.randn(8, 3, 128, 128)
loss = diffusion(training_images.cuda())
loss.backward()

# 采样1000步生成一张图片
sampled_images = diffusion.sample(batch_size = 4)


# 如果张量在 GPU上，需要移动到 CPU上
if sampled_images.is_cuda:
    sampled_images = sampled_images.cpu()

# 检查我们生成的一张图
img = sampled_images[0].clone().detach().permute(1, 2, 0)

plt.imshow(img)

