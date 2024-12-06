import os
from PIL import Image
import torch
import numpy as np
import pandas as pd
from utils.condition_function import AdaptBlurOperator
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
import logging

# 配置路径和参数
config = {
    'blur_path': '/media/gsn/1208B30C08B2EE3B/WSN_Deblur/BID/blind-dps/results/adapt_blind_blur/input',
    'gt_path': '/media/gsn/1208B30C08B2EE3B/WSN_Deblur/BID/blind-dps/results/adapt_blind_blur/label',
    'test_kernel_path': 'results/blind_dps',
    'save_path': 'results/blind_dps_reblur',
    'ksize': 32,
    'is_save': True
}

# 确保路径存在
if not os.path.exists(config['blur_path']):
    raise FileNotFoundError(f"blur_path does not exist: {config['blur_path']}")
if not os.path.exists(config['gt_path']):
    raise FileNotFoundError(f"gt_path does not exist: {config['gt_path']}")

if not os.path.exists(config['save_path']):
    os.makedirs(config['save_path'])

# 日志配置
logging.basicConfig(level=logging.INFO)

# 设备设置
device = torch.device('cuda')
blur_operator = AdaptBlurOperator(device, ksize=config['ksize'])

psnr_list = []
ssim_list = []
results = []  # 用于存储每个图像的结果

# 预处理所有图像路径
kernel_paths = [os.path.join(config['test_kernel_path'], kernel_name) for kernel_name in
                os.listdir(config['test_kernel_path']) if kernel_name.startswith('ker')]
image_sp_names = [kernel_name.replace('ker', 'img') for kernel_name in os.listdir(config['test_kernel_path']) if
                  kernel_name.startswith('ker')]
image_sp_paths = [os.path.join(config['gt_path'], image_sp_name) for image_sp_name in image_sp_names]
blur_gt_paths = [os.path.join(config['blur_path'], image_sp_name) for image_sp_name in image_sp_names]

# 处理每个测试核
for kernel_path, image_sp_path, blur_gt_path in zip(kernel_paths, image_sp_paths, blur_gt_paths):
    try:
        kernel_test = Image.open(kernel_path)
    except (IOError, OSError) as e:
        logging.error(f"Failed to open kernel file: {kernel_path}. Error: {e}")
        continue

    if kernel_test.mode == 'L' or kernel_test.mode == 'RGBA':
        kernel_test = kernel_test.convert('RGB')
    kernel_test = np.array(kernel_test) / 255.
    kernel_test = (torch.from_numpy(kernel_test).permute(2, 0, 1)
                   .unsqueeze(0).to(device)).to(torch.float32)

    try:
        image_sp = Image.open(image_sp_path)
    except (IOError, OSError) as e:
        logging.error(f"Failed to open sharp image file: {image_sp_path}. Error: {e}")
        continue

    if image_sp.mode == 'L' or image_sp.mode == 'RGBA':
        image_sp = image_sp.convert('RGB')

    image_sp = np.array(image_sp) / 255.
    image_sp = (torch.from_numpy(image_sp).permute(2, 0, 1)
                .unsqueeze(0).to(device)).to(torch.float32)

    # 应用模糊算子
    reblurred_image = blur_operator.forward(image_sp, kernel_test)
    # 保存重新模糊的图像
    reblurred_image_np = reblurred_image.squeeze().cpu().numpy().transpose(1, 2, 0)

    try:
        blur_gt = Image.open(blur_gt_path)
    except (IOError, OSError) as e:
        logging.error(f"Failed to open blur ground truth file: {blur_gt_path}. Error: {e}")
        continue

    if blur_gt.mode == 'L' or blur_gt.mode == 'RGBA':
        blur_gt = blur_gt.convert('RGB')

    blur_gt = np.array(blur_gt) / 255.
    blur_gt.astype(np.float32)

    psnr = peak_signal_noise_ratio(blur_gt, reblurred_image_np)
    ssim = structural_similarity(blur_gt, reblurred_image_np, data_range=blur_gt.max() - blur_gt.min(), multichannel=True, channel_axis=-1)
    if not np.isnan(psnr) and not np.isnan(ssim):
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        results.append({
            'Kernel Name': os.path.basename(kernel_path),
            'PSNR': psnr,
            'SSIM': ssim
        })
    print(f"PSNR: {psnr}, SSIM: {ssim}")
    print(f"kernel name: {os.path.basename(kernel_path)}")
    print('-' * 50)

    if config['is_save']:
        reblurred_image_np = (reblurred_image_np * 255).astype(np.uint8)
        reblurred_image_pil = Image.fromarray(reblurred_image_np)
        reblurred_image_pil.save(os.path.join(config['save_path'], f"reblurred_{os.path.basename(kernel_path)}"))

# 创建 DataFrame 并保存为 CSV 文件
df_results = pd.DataFrame(results)
csv_save_path = os.path.join(config['save_path'], 'results.csv')
df_results.to_csv(csv_save_path, index=False)

print(f"Average PSNR: {np.mean(psnr_list)}, Average SSIM: {np.mean(ssim_list)}")
print('-' * 50)
print(f"PSNR std: {np.std(psnr_list)}, SSIM std: {np.std(ssim_list)}")
