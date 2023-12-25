from PIL import Image
import numpy as np
import torch
from diffusers import AutoencoderKL
import requests
from io import BytesIO

# 设置设备并加载模型
device = 'cuda'
vae = AutoencoderKL.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder='vae')
vae = vae.to(device)

# 定义编码函数
def encode_img_latents(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    img_arr = np.stack([np.array(img) for img in imgs], axis=0)
    img_arr = img_arr / 255.0
    img_arr = torch.from_numpy(img_arr).float().permute(0, 3, 1, 2)
    img_arr = 2 * (img_arr - 0.5)
    latent_dists = vae.encode(img_arr.to(device))
    latent_samples = latent_dists.latent_dist.sample()
    latent_samples *= 0.18215
    return latent_samples

# 定义解码函数
def decode_img_latents(latents):
    latents = 1 / 0.18215 * latents
    with torch.no_grad():
        imgs = vae.decode(latents)
    imgs = (imgs.sample / 2 + 0.5).clamp(0, 1)
    imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
    imgs = (imgs * 255).round().astype('uint8')
    pil_images = [Image.fromarray(image) for image in imgs]
    return pil_images

# 拼接图像函数
def concat_image(im_list):
    img_number = len(im_list)
    if img_number == 0:
        return None
    dst = Image.new('RGB', (img_number * im_list[0].width, im_list[0].height))
    for idx, im in enumerate(im_list, 0):
        dst.paste(im, (idx * im.width, 0))
    return dst

# 测试用例
pths = ["", ""]  # 替换为实际图像 URL
for pth in pths:
    response = requests.get(pth)
    img = Image.open(BytesIO(response.content)).convert('RGB')
    img = img.resize((256, 256))
    img_latents = encode_img_latents(img)
    dec_img = decode_img_latents(img_latents)[0]
    concat_image([img, dec_img]).show()  # 显示拼接的图像
