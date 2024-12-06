import os
import cv2
from PIL import Image
import numpy as np


def validate_path(path):
    """验证路径是否合法"""
    if not os.path.abspath(path).startswith("/media/gsn/1208B30C08B2EE3B/WSN_Deblur/"):
        raise ValueError("Invalid path")
    return path


def create_directory(dir_path):
    """创建目录，并处理异常"""
    try:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
    except OSError as e:
        print(f"Error creating directory {dir_path}: {e}")
        return False
    return True


def process_image(file_path, save_path, ksize=27, body_size=2):
    """处理单个图像"""
    try:
        nu_k = Image.open(file_path)
    except IOError as e:
        print(f"Error opening file {file_path}: {e}")
        return

    nu_k = nu_k.convert("L")
    nu_k = np.array(nu_k)
    new_k = np.zeros_like(nu_k)

    for i in range(0, nu_k.shape[0], ksize):
        for j in range(0, nu_k.shape[1], ksize):
            nu_k_patch = nu_k[i:i + ksize, j:j + ksize]
            nu_k_patch = nu_k_patch[body_size:-body_size, body_size:-body_size]
            nu_k_patch = np.pad(nu_k_patch,
                                ((body_size, body_size), (body_size, body_size)), 'constant')
            new_k[i:i + ksize, j:j + ksize] = nu_k_patch

    new_k = new_k.astype(np.uint8)
    new_k = Image.fromarray(new_k)
    new_k = new_k.resize((256, 256))
    try:
        new_k.save(save_path)
    except IOError as e:
        print(f"Error saving file {save_path}: {e}")


def process_deblur_fast_nonuniform(ksize=27):
    root_dir = validate_path("/media/gsn/1208B30C08B2EE3B/WSN_Deblur/BID/EM_SVKernels/deblur_fast_nonuniform_v1.0")
    file_name = "s01_it0005_filters.png"
    save_root_dir = "../results"
    if not create_directory(save_root_dir):
        return

    body_size = 2
    dict_list = os.listdir(root_dir)
    for dict_name in dict_list:
        if dict_name.startswith("000"):
            file_path = os.path.join(root_dir, dict_name, file_name)
            save_path = os.path.join(save_root_dir, dict_name + ".png")
            process_image(file_path, save_path, ksize, body_size)


def process_blind_dps(ksize=27):
    root_dir = validate_path("/media/gsn/1208B30C08B2EE3B/WSN_Deblur/BID/blind-dps/results/adapt_blind_blur/recon")
    save_dir = "../results/blind_dps"
    if not create_directory(save_dir):
        return

    for ker_name in os.listdir(root_dir):
        if ker_name.startswith("ker"):
            ker_path = os.path.join(root_dir, ker_name)
            try:
                ker = Image.open(ker_path)
            except IOError as e:
                print(f"Error opening file {ker_path}: {e}")
                continue

            ker = ker.convert("L")
            ker = np.array(ker)

            # kernel_size = 3
            # ker = cv2.GaussianBlur(ker, (kernel_size, kernel_size), 0)

            threshold = np.mean(ker)
            ker[ker < threshold] = 0

            # 将处理后的图像保存
            ker_filtered_img = Image.fromarray(ker.astype(np.uint8))
            try:
                ker_filtered_img.save(os.path.join(save_dir, ker_name))
            except IOError as e:
                print(f"Error saving file {ker_name}: {e}")


# process_deblur_fast_nonuniform()
process_blind_dps()
