"""
Task Data Loaders

Provides data for all 5 benchmark tasks:
- Digits (8x8): sklearn digits dataset
- MNIST: torchvision MNIST
- Fashion-MNIST: torchvision FashionMNIST
- CartPole: Synthetic behavioral cloning data
- Acrobot: Synthetic behavioral cloning data
"""

import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# Try to import torchvision, but make it optional
try:
    from torchvision import datasets, transforms
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False


def get_digits(batch_size=64):
    """8x8 digits from sklearn (fast, no download)."""
    digits = load_digits()
    X = digits.data.astype(np.float32) / 16.0  # Normalize
    y = digits.target.astype(np.int64)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            DataLoader(test_ds, batch_size=batch_size), 64, 10)


def get_mnist(batch_size=128, dataset_size=5000):
    """MNIST digits (requires torchvision)."""
    if not HAS_TORCHVISION:
        raise ImportError("torchvision required for MNIST")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: torch.flatten(x))
    ])
    
    train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    if dataset_size and dataset_size < len(train_data):
        train_data = Subset(train_data, torch.randperm(len(train_data))[:dataset_size])
    
    return (DataLoader(train_data, batch_size=batch_size, shuffle=True),
            DataLoader(test_data, batch_size=batch_size), 784, 10)


def get_fashion_mnist(batch_size=128, dataset_size=5000):
    """Fashion-MNIST (requires torchvision)."""
    if not HAS_TORCHVISION:
        raise ImportError("torchvision required for Fashion-MNIST")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
        transforms.Lambda(lambda x: torch.flatten(x))
    ])
    
    train_data = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
    test_data = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)
    
    if dataset_size and dataset_size < len(train_data):
        train_data = Subset(train_data, torch.randperm(len(train_data))[:dataset_size])
    
    return (DataLoader(train_data, batch_size=batch_size, shuffle=True),
            DataLoader(test_data, batch_size=batch_size), 784, 10)


def get_cartpole(batch_size=64, dataset_size=5000):
    """CartPole behavioral cloning (synthetic expert)."""
    X = np.random.uniform(-0.2, 0.2, (dataset_size, 4)).astype(np.float32)
    y = np.where(X[:, 2] < 0, 0, 1).astype(np.int64)  # Heuristic: angle < 0 → left
    
    split = int(0.8 * dataset_size)
    train_ds = TensorDataset(torch.from_numpy(X[:split]), torch.from_numpy(y[:split]))
    test_ds = TensorDataset(torch.from_numpy(X[split:]), torch.from_numpy(y[split:]))
    
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            DataLoader(test_ds, batch_size=batch_size), 4, 2)


def get_acrobot(batch_size=64, dataset_size=5000):
    """Acrobot behavioral cloning (synthetic expert)."""
    X = np.zeros((dataset_size, 6), dtype=np.float32)
    theta1 = np.random.uniform(-np.pi, np.pi, dataset_size)
    theta2 = np.random.uniform(-np.pi, np.pi, dataset_size)
    X[:, 0], X[:, 1] = np.cos(theta1), np.sin(theta1)
    X[:, 2], X[:, 3] = np.cos(theta2), np.sin(theta2)
    X[:, 4] = np.random.uniform(-4, 4, dataset_size)
    X[:, 5] = np.random.uniform(-9, 9, dataset_size)
    
    # Swing-up heuristic
    tip = X[:, 0] + X[:, 0] * X[:, 2] - X[:, 1] * X[:, 3]
    momentum = X[:, 4] + X[:, 5]
    y = np.where(tip > 0.5, 1, np.where(momentum > 0, 2, 0)).astype(np.int64)
    
    split = int(0.8 * dataset_size)
    train_ds = TensorDataset(torch.from_numpy(X[:split]), torch.from_numpy(y[:split]))
    test_ds = TensorDataset(torch.from_numpy(X[split:]), torch.from_numpy(y[split:]))
    
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            DataLoader(test_ds, batch_size=batch_size), 6, 3)


def get_task(name, **kwargs):
    """Get data loaders for a named task."""
    loaders = {
        'digits': get_digits,
        'mnist': get_mnist,
        'fashion': get_fashion_mnist,
        'cartpole': get_cartpole,
        'acrobot': get_acrobot,
    }
    return loaders[name](**kwargs)
