"""WGAN-GP training script on MNIST

Saves generator and critic checkpoints and sample grids.

Usage:
    python wgan_gp_mnist.py --epochs 10 --batch-size 64 --outdir ./checkpoints

This script:
 - Defines a simple conv Generator and Critic (critic outputs scalar score)
 - Uses Wasserstein loss with gradient penalty (WGAN-GP)
 - Trains on MNIST (grayscale 28x28)
 - Saves checkpoints and sample images

Designed to be readable and easy to adapt.
"""

import os
import argparse
from math import inf

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

# -------------------------------
# Models
# -------------------------------

class Generator(nn.Module):
    def __init__(self, z_dim=100, out_channels=1, base_channels=64):
        super().__init__()
        self.net = nn.Sequential(
            # input z: (N, z_dim, 1, 1) after view
            nn.ConvTranspose2d(z_dim, base_channels*4, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(base_channels*4),
            nn.ReLU(True),                 # -> (B, base*4, 3, 3)

            nn.ConvTranspose2d(base_channels*4, base_channels*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU(True),                 # -> (B, base*2, 6, 6)

            nn.ConvTranspose2d(base_channels*2, base_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(True),                 # -> (B, base, 12, 12)

            nn.ConvTranspose2d(base_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()                      # -> (B, 1, 28, 28) (since starting from small conv we arrive close to 28)
        )

    def forward(self, z):
        # z: (B, z_dim)
        z = z.view(z.size(0), z.size(1), 1, 1)
        return self.net(z)


class Critic(nn.Module):
    def __init__(self, in_channels=1, base_channels=64):
        super().__init__()
        self.net = nn.Sequential(
            # input: (B, 1, 28, 28)
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),   # (B, base, 14, 14)

            nn.Conv2d(base_channels, base_channels*2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(base_channels*2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),   # (B, base*2, 7, 7)

            nn.Conv2d(base_channels*2, base_channels*4, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(base_channels*4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),   # (B, base*4, 4, 4)

            nn.Conv2d(base_channels*4, 1, kernel_size=4, stride=1, padding=0),
        )

    def forward(self, x):
        # returns scalar per sample (no sigmoid)
        out = self.net(x)
        return out.view(-1)


# -------------------------------
# Utilities: gradient penalty, saving
# -------------------------------

def gradient_penalty(critic, real, fake, device, lambda_gp=10.0):
    batch_size = real.size(0)
    # interpolate
    eps = torch.rand(batch_size, 1, 1, 1, device=device)
    x_hat = eps * real + (1 - eps) * fake
    x_hat.requires_grad_(True)

    pred = critic(x_hat)
    grad_outputs = torch.ones_like(pred, device=device)

    grads = torch.autograd.grad(
        outputs=pred,
        inputs=x_hat,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    grads = grads.view(batch_size, -1)
    gp = ((grads.norm(2, dim=1) - 1) ** 2).mean() * lambda_gp
    return gp


def save_checkpoint(state, outdir, step, prefix="ckpt"):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{prefix}_{step}.pt")
    torch.save(state, path)
    return path


# -------------------------------
# Training loop
# -------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),  # range [-1,1]
    ])

    dataset = datasets.MNIST(root=args.datadir, train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)

    gen = Generator(z_dim=args.z_dim).to(device)
    critic = Critic().to(device)

    opt_g = optim.Adam(gen.parameters(), lr=args.lr, betas=(0.5, 0.9))
    opt_c = optim.Adam(critic.parameters(), lr=args.lr, betas=(0.5, 0.9))

    steps = 0
    fixed_noise = torch.randn(64, args.z_dim, device=device)

    best_wasserstein = inf

    for epoch in range(args.epochs):
        for real, _ in dataloader:
            real = real.to(device)
            batch_size = real.size(0)

            # ---------- train critic ----------
            for _ in range(args.n_critic):
                z = torch.randn(batch_size, args.z_dim, device=device)
                fake = gen(z).detach()

                crit_real = critic(real)
                crit_fake = critic(fake)

                gp = gradient_penalty(critic, real, fake, device, lambda_gp=args.lambda_gp)

                loss_c = crit_fake.mean() - crit_real.mean() + gp

                opt_c.zero_grad()
                loss_c.backward()
                opt_c.step()

            # ---------- train generator ----------
            z = torch.randn(batch_size, args.z_dim, device=device)
            fake = gen(z)
            loss_g = -critic(fake).mean()

            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()

            steps += 1

            if steps % args.log_interval == 0:
                with torch.no_grad():
                    fake_vis = gen(fixed_noise).cpu()
                    grid = utils.make_grid((fake_vis + 1) / 2, nrow=8)
                    utils.save_image(grid, os.path.join(args.outdir, f"samples_{steps}.png"))

                print(f"Epoch[{epoch}/{args.epochs}] Step {steps} loss_c={loss_c.item():.4f} loss_g={loss_g.item():.4f} gp={gp.item():.4f}")

            if steps % args.save_interval == 0:
                save_checkpoint({
                    'gen_state_dict': gen.state_dict(),
                    'crit_state_dict': critic.state_dict(),
                    'opt_g_state': opt_g.state_dict(),
                    'opt_c_state': opt_c.state_dict(),
                    'steps': steps,
                    'args': vars(args)
                }, args.outdir, steps, prefix="wgangp")

    # final save
    save_checkpoint({
        'gen_state_dict': gen.state_dict(),
        'crit_state_dict': critic.state_dict(),
        'opt_g_state': opt_g.state_dict(),
        'opt_c_state': opt_c.state_dict(),
        'steps': steps,
        'args': vars(args)
    }, args.outdir, steps, prefix="wgangp_final")


# -------------------------------
# CLI
# -------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--datadir', type=str, default='./data')
    p.add_argument('--outdir', type=str, default='./checkpoints')
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--z-dim', type=int, default=100)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--n-critic', type=int, default=5)
    p.add_argument('--lambda-gp', type=float, default=10.0)
    p.add_argument('--log-interval', type=int, default=200)
    p.add_argument('--save-interval', type=int, default=1000)
    p.add_argument('--cpu', action='store_true')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    train(args)
