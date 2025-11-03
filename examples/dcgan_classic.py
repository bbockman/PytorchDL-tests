import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import dataloader


# --- Generator ---
class Generator(nn.Module):
    def __init__(self, z_dim=100, img_dim=28*28):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.ReLU(),
            nn.Linear(256, img_dim),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)

# --- Discriminator ---
class Discriminator(nn.Module):
    def __init__(self, img_dim=28*28):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(img_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.net(x)

# --- Init ---
device = "cuda" if torch.cuda.is_available() else "cpu"
gen = Generator().to(device)
disc = Discriminator().to(device)

lr = 2e-4
z_dim = 100
criterion = nn.BCEWithLogitsLoss()
opt_gen = optim.Adam(gen.parameters(), lr=lr)
opt_disc = optim.Adam(disc.parameters(), lr=lr)


for real, _ in dataloader:
    real = real.view(-1, 784).to(device)
    batch_size = real.size(0)
    noise = torch.randn(batch_size, z_dim).to(device)
    fake = gen(noise)

    ### Train Discriminator ###
    disc_real = disc(real).view(-1)
    disc_fake = disc(fake.detach()).view(-1)
    loss_real = criterion(disc_real, torch.ones_like(disc_real))
    loss_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
    loss_disc = (loss_real + loss_fake) / 2
    opt_disc.zero_grad()
    loss_disc.backward()
    opt_disc.step()

    ### Train Generator ###
    output = disc(fake).view(-1)
    loss_gen = criterion(output, torch.ones_like(output))
    opt_gen.zero_grad()
    loss_gen.backward()
    opt_gen.step()
