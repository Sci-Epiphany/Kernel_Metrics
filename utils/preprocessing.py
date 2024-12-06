import os
from PIL import Image
import numpy as np


def process_deblur_fast_nonuniform(ksize=27):
    root_dir = "/media/gsn/1208B30C08B2EE3B/WSN_Deblur/BID/EM_SVKernels/deblur_fast_nonuniform_v1.0"
    file_name = "s01_it0005_filters.png"
    save_root_dir = "../results"
    if not os.path.exists(save_root_dir):
        os.mkdir(save_root_dir)

    body_size = 2
    dict_list = os.listdir(root_dir)
    for dict_name in dict_list:
        if dict_name.startswith("000"):
            file_path = os.path.join(root_dir, dict_name, file_name)
            print(file_path)
            nu_k = Image.open(file_path)
            nu_k = nu_k.convert("L")

            nu_k = np.array(nu_k)
            new_k = np.zeros_like(nu_k)

            for i in range(0, nu_k.shape[0], 27):
                for j in range(0, nu_k.shape[1], 27):
                    nu_k_patch = nu_k[i:i+ksize, j:j+ksize]
                    nu_k_patch = nu_k_patch[body_size:-body_size, body_size:-body_size]
                    nu_k_patch = np.pad(nu_k_patch,
                                        ((body_size, body_size), (body_size, body_size)), 'constant')
                    new_k[i:i+ksize, j:j+ksize] = nu_k_patch
            new_k = new_k.astype(np.uint8)
            new_k = Image.fromarray(new_k)
            new_k = new_k.resize((256, 256))
            new_k.save(os.path.join(save_root_dir, dict_name+".png"))

from scipy.ndimage import median_filter

def process_blind_dps(ksize=27):
    root_dir = "/media/gsn/1208B30C08B2EE3B/WSN_Deblur/BID/blind-dps/results/adapt_blind_blur/recon"
    save_dir = "../results/blind_dps"
    ker_list = os.listdir(root_dir)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for ker_name in ker_list:
        if ker_name.startswith("ker"):
            ker_path = os.path.join(root_dir, ker_name)
            ker = Image.open(ker_path)
            ker = ker.convert("L")
            ker = np.array(ker)

            # # 对 ker 进行中值滤波
            # ker_filtered = median_filter(ker, size=3)  # 你可以根据需要调整 size 参数

            # 将处理后的图像保存
            ker_filtered_img = Image.fromarray(ker.astype(np.uint8))
            ker_filtered_img.save(os.path.join(save_dir, ker_name))




# process_deblur_fast_nonuniform()
process_blind_dps()