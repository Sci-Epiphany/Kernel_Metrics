import numpy as np
import cv2
import matplotlib.pyplot as plt

def draw_masked_flow(base_img, kernel, inter=32, color=(0, 255, 0)):
    # Convert base_img to numpy array and preprocess
    base_img = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
    base_img = np.tile(np.expand_dims(base_img, -1), (1, 1, 3))
    base_img = 0.6 * 1 + 0.4 * base_img
    base_img = np.clip(base_img, 0, 1)  # Ensure values are within [0, 1]
    base_img = np.uint8(base_img * 255)
    H, W = base_img.shape[0:2]

    # Convert kernel to numpy array and preprocess
    kernel = np.uint8(kernel * 255)
    # Initialize flow map
    flow_map = np.copy(base_img)

    # Draw masked flow on the flow map by overlaying with a color
    for i in range(0, H, inter):
        for j in range(0, W, inter):
            # Extract the region from kernel
            kernel_region = kernel[i:i + inter, j:j + inter]
            # Blend the colored region with the flow_map region
            flow_map[i:i + inter, j:j + inter,0] = (flow_map[i:i + inter, j:j + inter,0] *
                                                  (1 - kernel_region / 255)) + 0.5 * kernel_region
            flow_map[i:i + inter, j:j + inter,1:] = (flow_map[i:i + inter, j:j + inter,1:] *
                                                  (1 - kernel_region.reshape(inter, inter, 1).repeat(2, axis=2) / 255))

    return flow_map


def normalize_with_patch(img, patch_size=32):
    """
    读取图像，将其分割为8x8的分块，对每个分块进行归一化处理，然后拼合为一张图像。

    :param image_path: 图像文件路径
    :param patch_size: 分块大小，默认为8
    :return: 处理后的图像
    """
    # 读取图像
    image = img
    width, height = image.shape

    # 确保图像尺寸是patch_size的整数倍
    if width % patch_size != 0 or height % patch_size != 0:
        raise ValueError("图像宽度和高度必须是patch_size的整数倍")

    # 计算分块数量
    num_patches_x = width // patch_size
    num_patches_y = height // patch_size

    # 创建一个空的列表来存储归一化后的分块
    normalized_patches = []

    # 遍历每个分块
    for y in range(num_patches_y):
        for x in range(num_patches_x):
            # 提取分块
            patch = image[y*patch_size:(y+1)*patch_size, x*patch_size:(x+1)*patch_size]
            # 归一化分块
            normalized_patch = (patch - np.min(patch)) / (np.max(patch) - np.min(patch))
            normalized_patches.append(normalized_patch)

    # 将归一化后的分块重新拼合成图像
    normalized_image_array = np.block([[normalized_patches[i * num_patches_x + j]
                            for j in range(num_patches_x)] for i in range(num_patches_y)])

    # 将NumPy数组转换回PIL图像


    return normalized_image_array
