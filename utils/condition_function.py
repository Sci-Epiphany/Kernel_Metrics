from abc import ABC, abstractmethod
from utils.blur_utils import blockConv2d
import torch

class BlurOperator(ABC):
    @abstractmethod
    def forward(self, data, kernel, **kwargs):
        pass


class AdaptBlurOperator(BlurOperator):
    def __init__(self, device,ksize=32, **kwargs) -> None:
        self.device = device
        self.ksize = ksize

    def forward(self, data, kernel, **kwargs):
        return self.apply_kernel(data, kernel)

    def apply_kernel(self, data, kernel):
        # TODO: Find an adaptive convolution with spatial awareness.
        # TODO: Now Kernel is just set to 1 channel, needing to be updated three channels in the future.
        # here just expand the kernel channel by duplicating 3 times.
        B, C, H, W = kernel.shape
        k_size = self.ksize
        w_ = W // k_size
        h_ = H // k_size
        if k_size % 2 == 0:
            kernels = torch.zeros([B, w_, h_, C, k_size - 1, k_size - 1])
        else:
            kernels = torch.zeros([B, w_, h_, C, k_size, k_size])
        for b in range(B):
            for i in range(w_):
                for j in range(h_):
                    current_kernel = kernel[b, :, i * k_size + 1:(i + 1) * k_size, j * k_size + 1:(j + 1) * k_size]
                    # current_kernel = (current_kernel + 1.0) / 2.0    # just test the normalization
                    channel_sums = torch.sum(current_kernel, dim=(1, 2))
                    for c in range(current_kernel.shape[0]):
                        current_kernel[c] /= channel_sums[c]
                    kernels[b, i, j, :, :, :] = current_kernel

        b_img = blockConv2d(data, kernels.to(self.device))

        return b_img

