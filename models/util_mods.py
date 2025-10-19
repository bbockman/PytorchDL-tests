import torch
from torch import nn
import torch.nn.functional as F

class MaxBlurPool2d(nn.Module):
    """
    MaxBlurPool2d combines max pooling and a blur filter to reduce aliasing.
    It matches MaxPool2d(kernel_size=2, stride=2) in output size.
    """
    def __init__(self, channels, kernel_size=2, stride=2):
        super().__init__()
        self.stride = stride
        self.kernel_size = kernel_size

        # Fixed 3x3 blur kernel (binomial)
        blur_kernel = torch.tensor([1., 2., 1.])
        blur_kernel = blur_kernel[:, None] * blur_kernel[None, :]
        blur_kernel = blur_kernel / blur_kernel.sum()

        # Register as buffer (not trainable)
        self.register_buffer(
            "kernel",
            blur_kernel[None, None, :, :].repeat(channels, 1, 1, 1)
        )

        # MaxPool with stride=1 to keep same spatial size before blur
        self.max_pool = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x):
        # Step 1: Max pool (stride=1)
        x = self.max_pool(x)

        # Step 2: Blur convolution (depthwise)
        x = F.conv2d(
            x,
            self.kernel,
            stride=self.stride,     # downsample by 2
            padding=1,              # maintain correct center alignment
            groups=x.shape[1]
        )
        return x

class SEBlock(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c // r, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c // r, c, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.fc(x)