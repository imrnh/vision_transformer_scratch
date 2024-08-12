"""
    Courtesy to: https://medium.com/@brianpulfer/vision-transformers-from-scratch-pytorch-a-step-by-step-guide-96c3313c2e0c
"""

import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms

import numpy as np
from tqdm import tqdm, trange
from torchview import draw_graph
from types import SimpleNamespace

from vit import ViT

np.random.seed(0)
torch.manual_seed(0)

train_conf = SimpleNamespace(
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    n_epochs=5,
    lr=0.001,
)


def main():
    # Loading data
    transform = transforms.ToTensor()

    train_set = MNIST(root="./../datasets", train=True, download=True, transform=transform)
    test_set = MNIST(root="./../datasets", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, shuffle=True, batch_size=128)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=128)

    model = ViT((1, 28, 28), n_patches=7, n_blocks=2, hidden_d=8, n_heads=2, out_d=10).to(train_conf.device)

    # Training loop
    optimizer = Adam(model.parameters(), lr=train_conf.lr)
    criterion = CrossEntropyLoss()
    for epoch in range(train_conf.n_epochs):
        train_loss = 0.0
        for batch in tqdm(train_loader):
            x, y = batch
            x, y = x.to(train_conf.device), y.to(train_conf.device)
            y_hat = model(x)
            loss = criterion(y_hat, y)

            train_loss += loss.detach().cpu().item() / len(train_loader)

            optimizer.zero_grad()
            loss.backward()  # backprop calculation
            optimizer.step()  # Updating the weight based on these calculation.

        print(f"Epoch {epoch + 1}/{train_conf.n_epochs} loss: {train_loss:.2f}")

    # Test loop
    with torch.no_grad():
        correct, total = 0, 0
        test_loss = 0.0
        for batch in tqdm(test_loader, desc="Testing"):
            x, y = batch
            x, y = x.to(train_conf.device), y.to(train_conf.device)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            test_loss += loss.detach().cpu().item() / len(test_loader)

            correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
            total += len(x)
        print(f"Test loss: {test_loss:.2f}")
        print(f"Test accuracy: {correct / total * 100:.2f}%")


if __name__ == "__main__":
    main()
