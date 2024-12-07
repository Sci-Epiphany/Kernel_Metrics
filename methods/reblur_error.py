import os

import yaml
from PIL import Image
import torch
import numpy as np
import pandas as pd
from tensorboard.compat.tensorflow_stub import string

from utils.condition_function import AdaptBlurOperator
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from methods.visualization import draw_masked_flow, normalize_with_patch
import logging
import argparse

# Create an argument parser object
parser = argparse.ArgumentParser(description='Evaluation arguments')
parser.add_argument('--config', type=str, default='configs/reblur_blind_dps.yaml', help='Configs of reblur settings')
parser.add_argument('--plt_flow',type=bool, default=True, help='Whether to plot flow')
parser.add_argument('--plt_gt_flow',type=bool, default=False, help='Whether to plot ground truth flow')

# Parse command line arguments
args = parser.parse_args()

# Read configuration file
with open(args.config, 'r') as file:
    config = yaml.safe_load(file)

# Ensure paths exist
if not os.path.exists(config['blur_path']):
    raise FileNotFoundError(f"blur_path does not exist: {config['blur_path']}")
if not os.path.exists(config['gt_path']):
    raise FileNotFoundError(f"gt_path does not exist: {config['gt_path']}")

if not os.path.exists(config['save_path']):
    os.makedirs(config['save_path'])

if not os.path.exists(config['flow_save_path']):
    os.makedirs(config['flow_save_path'])

# Configure logging
logging.basicConfig(level=logging.INFO)

# Set device
device = torch.device('cuda')
blur_operator = AdaptBlurOperator(device, ksize=config['ksize'])

psnr_list = []
ssim_list = []
results = []  # Store results for each image

# Preprocess all image paths
kernel_paths = [os.path.join(config['test_kernel_path'], kernel_name) for kernel_name in
                os.listdir(config['test_kernel_path']) if kernel_name.startswith('ker')]
image_sp_names = [kernel_name.replace('ker', 'img') for kernel_name in os.listdir(config['test_kernel_path']) if
                  kernel_name.startswith('ker')]
kernel_gt_names = [kernel_name for kernel_name in os.listdir(config['test_kernel_path']) if
                  kernel_name.startswith('ker')]
kernel_gt_paths = [os.path.join(config['gt_path'], kernel_gt_name) for kernel_gt_name in kernel_gt_names]
image_sp_paths = [os.path.join(config['gt_path'], image_sp_name) for image_sp_name in image_sp_names]
blur_gt_paths = [os.path.join(config['blur_path'], image_sp_name) for image_sp_name in image_sp_names]

# Process each test kernel
for kernel_path, image_sp_path, kernel_gt_path, blur_gt_path in zip(kernel_paths, image_sp_paths, kernel_gt_paths, blur_gt_paths):
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

    # Apply blur operator
    reblurred_image = blur_operator.forward(image_sp, kernel_test)
    # Save the reblurred image
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
        if args.plt_flow:
            kernel_test = kernel_test.squeeze().cpu().numpy()
            kernel_test_np = normalize_with_patch(kernel_test[0, :, :])
            flow = draw_masked_flow(blur_gt.astype(np.float32), kernel_test_np)
            flow = Image.fromarray(flow)
            flow.save(os.path.join(config['flow_save_path'], f"flow_{os.path.basename(kernel_path)}"))
        if args.plt_gt_flow:
            try:
                kernel_gt = Image.open(kernel_gt_path)
            except (IOError, OSError) as e:
                logging.error(f"Failed to open sharp image file: {kernel_gt_path}. Error: {e}")
                continue

            if kernel_gt.mode == 'L' or kernel_gt.mode == 'RGBA':
                kernel_gt = kernel_gt.convert('RGB')

            kernel_gt = np.array(kernel_gt) / 255.
            kernel_gt_np = normalize_with_patch(kernel_gt[:, :, 0])
            flow = draw_masked_flow(blur_gt.astype(np.float32), kernel_gt_np)
            flow = Image.fromarray(flow)
            flow.save(os.path.join(config['flow_save_path'], f"gt_flow_{os.path.basename(kernel_path)}"))

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

# Create DataFrame and save as CSV file
df_results = pd.DataFrame(results)
csv_save_path = os.path.join(config['save_path'], 'results.csv')
df_results.to_csv(csv_save_path, index=False)

print(f"Average PSNR: {np.mean(psnr_list)}, Average SSIM: {np.mean(ssim_list)}")
print('-' * 50)
print(f"PSNR std: {np.std(psnr_list)}, SSIM std: {np.std(ssim_list)}")
