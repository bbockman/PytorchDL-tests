import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def compute_mean_std(dataset, batch_size=1000):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    mean, std, num_batches = 0.0, 0.0, 0
    for images, _ in loader:
        mean += images.mean().item()
        std += images.std().item()
        num_batches += 1
    return mean / num_batches, std / num_batches


def get_mnist_loaders(batch_size=64, num_workers=8, pin_memory=True):
    raw_transform = transforms.Compose([transforms.ToTensor()])
    base_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=raw_transform)

    mean, std = compute_mean_std(base_dataset)
    print(f"Computed mean: {mean:.4f}, std: {std:.4f}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize((mean,), (std,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    test_loader  = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    return train_loader, test_loader
