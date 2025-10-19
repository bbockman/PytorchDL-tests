import torch
from torch import nn

from models.mnist_cnn import MaxBlurPool2d
import torch.nn.functional as F

class ChannelLinearCombo(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Define learnable weights: one scalar per input channel per output channel
        # Shape: (out_channels, in_channels)
        self.weights = nn.Parameter(torch.randn(out_channels, in_channels))

    def forward(self, x):
        """
        x: (B, C, H, W)
        returns: (B, out_channels, H, W)
        """
        B, C, H, W = x.shape

        # Reshape x to (B, 1, C, H, W) so we can multiply by weights
        x_expanded = x.unsqueeze(1)  # (B, 1, C, H, W)

        # weights shape: (out_channels, in_channels) -> (1, out_channels, C, 1, 1)
        w_expanded = self.weights.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        # Multiply and sum over input channels
        out = (x_expanded * w_expanded).sum(dim=2)  # sum over C

        return out  # (B, out_channels, H, W)

class MNIST_BatchNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.pool = MaxBlurPool2d(32, kernel_size=2, stride=2)

        self.fc1 = nn.Linear(32 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class MNIST_Toy(nn.Module):
    def __init__(self):
        super().__init__()
        # Pooling after conv, similar to reference model gives better performance, but is much slower
        self.conv1 = nn.Conv2d(1, 1, 1, padding=0)
        self.conv2 = nn.Conv2d(1, 1, 3, padding=1)
        self.conv3 = nn.Conv2d(1, 1, 5, padding=2)
        self.conv4 = nn.Conv2d(1, 1, 7, padding=3)
        self.conv5 = nn.Conv2d(1, 1, 9, padding=4)

        self.conv6 = nn.Conv2d(5, 128, 3, padding=1)
        self.conv7 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool = MaxBlurPool2d(128, kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(128)
        self.drop1 = nn.Dropout(0.25)
        self.drop2 = nn.Dropout(0.25)

        self.fc1 = nn.Linear(128 * 14 * 14, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x5 = self.conv5(x)
        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        x = F.gelu(x)
        x = self.drop1(x)
        x = F.gelu(self.conv6(x))
        x = self.pool(x)
        x = F.gelu(self.conv7(x))
        x = torch.flatten(x, start_dim=1)
        x = F.gelu(self.fc1(x))
        x = self.drop2(x)
        return self.fc2(x)