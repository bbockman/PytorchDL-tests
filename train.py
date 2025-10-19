import time

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import get_gpu_stats, init_gpu_monitor, count_parameters
from data import get_mnist_loaders


def evaluate_model(model, loader, device):
    """Evaluate model accuracy on a dataset loader."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    return correct / total


def train_model(model, epochs=20, lr=1e-3, save_path=None):
    """Train a PyTorch model on MNIST with AdamW + Cosine Annealing."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.float().to(device)

    train_loader, test_loader = get_mnist_loaders()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-5)

    handle = init_gpu_monitor()
    count_parameters(model)

    best_loss = float('inf')
    best_state = model.state_dict()
    start_time = time.perf_counter()

    for epoch in range(epochs):
        model.train()
        epoch_loss = correct = total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.sum()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        stats = get_gpu_stats(handle)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {correct / total:.4f}")
        print(f"GPU Util: {stats['gpu_util']}%, Mem: {stats['mem_used']:.1f}/{stats['mem_total']:.1f} MB")

        # Track best state
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = model.state_dict()

        scheduler.step()


    # Load best weights
    model.load_state_dict(best_state)
    elapsed = time.perf_counter() - start_time

    # Evaluation
    test_acc = evaluate_model(model, test_loader, device)
    print(f"Test Accuracy: {100*test_acc:.2f}%")
    print(f"Elapsed Time: {elapsed:.2f}s")

    # Save best model if path is provided
    if save_path:
        torch.save(best_state, save_path)
        print(f"Saved best model to {save_path}")

    return model, best_state, test_acc
