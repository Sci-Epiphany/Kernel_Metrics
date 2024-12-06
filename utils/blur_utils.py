import torch
import torch.nn.functional as F

def blockConv2d(image, kernels, expand=0):
    """
       Args:
           image: [B,C,H,W]
           kernels: [w_h,h_h,k_w,k_h,1]
           expend: 0, expand of origin image, whose default value is 0.
       Returns:
           output: [B,C,H,W]
       """
    B, C, W, H = image.shape
    grid_w, grid_h = kernels.shape[1:3]
    k_size = kernels.shape[-1]
    patch_size = (W - expand * 2) // grid_w
    to_pad = k_size // 2 - expand
    images_pad = F.pad(image, (to_pad, to_pad, to_pad, to_pad), mode='circular')
    output = torch.zeros(B, C, W - expand * 2, H - expand * 2).to(image.device)
    for b in range(B):
        for w_ in range(grid_w):
            for h_ in range(grid_h):
                x_start = w_ * patch_size
                x_end = (w_ + 1) * patch_size + k_size // 2 * 2
                y_start = h_ * patch_size
                y_end = (h_ + 1) * patch_size + k_size // 2 * 2
                patch = images_pad[b, :, x_start:x_end, y_start:y_end]
                for c in range(C):
                    output[b, c, x_start:x_start + patch_size, y_start:y_start + patch_size] = F.conv2d(
                        patch[c:c + 1, :, :].unsqueeze(0), kernels[b, w_, h_, c:c + 1, :, :].unsqueeze(0),
                        padding=0)
    return output