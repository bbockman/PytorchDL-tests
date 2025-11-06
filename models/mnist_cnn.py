import torch
import torch.nn as nn
import torch.nn.functional as F

from models.util_mods import MaxBlurPool2d


class MNIST_CNNs(nn.Module):
    def __init__(self):
        super().__init__()
        # Conv layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)  # output: 16x28x28
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1) # output: 16x28x28
        self.pool = nn.MaxPool2d(2, 2)                           # downsample: 16x14x14

        # Fully connected layers
        self.fc1 = nn.Linear(16*14*14, 64)
        self.fc2 = nn.Linear(64, 10)  # 10 classes

    def forward(self, x):
        x = self.pool(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)             # raw logits
        return x


class MNIST_CNNb(nn.Module):
    def __init__(self):
        super().__init__()
        # Conv layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # output: 32x28x28
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # output: 64x28x28
        self.pool = nn.MaxPool2d(2, 2)                           # downsample: 64x14x14

        # Fully connected layers
        self.fc1 = nn.Linear(64*14*14, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 classes

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)             # raw logits
        return x


class MNIST_Custom(nn.Module):
    def __init__(self):
        super().__init__()
        # Pooling after conv, similar to reference model gives better performance, but is much slower
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = MaxBlurPool2d(32, kernel_size=2, stride=2)
        self.drop1 = nn.Dropout(0.25)

        self.fc1 = nn.Linear(32 * 14 * 14, 128)
        self.drop2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, start_dim=1)
        x = F.gelu(self.fc1(x))
        x = self.drop2(x)
        return self.fc2(x)


class MNIST_Refer(nn.Module):
    """
    Well-balanced baseline CNN for MNIST (≈99.2–99.4% test accuracy)
    Structure:
      Conv(1→32) → Conv(32→64) → MaxPool → Dropout
      Conv(64→128) → MaxPool → Dropout
      FC(128*7*7 → 256) → Dropout → FC(256→10)
    """
    def __init__(self):
        super().__init__()
        # Convolutional block 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)   # → (B,32,28,28)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # → (B,64,28,28)
        self.pool1 = MaxBlurPool2d(64, 2, 2)                           # → (B,64,14,14)
        self.drop1 = nn.Dropout(0.25)

        # Convolutional block 2
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # → (B,128,14,14)
        self.pool2 = MaxBlurPool2d(128, 2, 2)                           # → (B,128,7,7)
        self.drop2 = nn.Dropout(0.25)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 7 * 7, 256)
        self.drop3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = self.drop1(x)

        x = F.relu(self.conv3(x))
        x = self.pool2(x)
        x = self.drop2(x)

        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.drop3(x)
        x = self.fc2(x)
        return x


class MNIST_ResRef(nn.Module):

    def __init__(self):
        super().__init__()
        # Convolutional block 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)   # → (B,32,28,28)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # → (B,64,28,28)
        self.pool1 = MaxBlurPool2d(64, 2, 2)                           # → (B,64,14,14)
        self.drop1 = nn.Dropout(0.25)

        # Convolutional block 2
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # → (B,128,14,14)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = MaxBlurPool2d(128, 2, 2)
        self.drop3 = nn.Dropout(0.25)

        # Post-residual + Batch-norm conv layers
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.drop4 = nn.Dropout(0.25)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)

        # Fully connected layers
        self.fc1 = nn.Linear(512 * 7 * 7, 256)
        self.drop3 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.bn3(x.repeat(1, 2, 1, 1) + self.conv3(F.relu(x)))
        x = self.pool3(x)
        x = self.drop3(x)

        x = F.relu(self.conv4(x))
        x = self.drop4(x)
        x = F.relu(self.conv5(x))

        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.drop3(x)
        x = self.fc2(x)
        return x