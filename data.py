import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from config import *

DATA_ROOT = r"C:\ML_data\cifar10"


def get_cifar10():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616)
        )
    ])
    train_set = datasets.CIFAR10(DATA_ROOT, train=True,  download=True, transform=transform)
    test_set  = datasets.CIFAR10(DATA_ROOT, train=False, download=True, transform=transform)
    return train_set, test_set


def dirichlet_partition(dataset, num_clients, alpha, seed=SEED):
    """
    Partition dataset indices across clients using a Dirichlet distribution.
    Low alpha → skewed (non-IID), high alpha → uniform (IID).
    """
    np.random.seed(seed)
    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    num_classes = len(np.unique(labels))
    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        class_idx = np.where(labels == c)[0]
        np.random.shuffle(class_idx)

        proportions = np.random.dirichlet(alpha=np.repeat(alpha, num_clients))
        split_points = (np.cumsum(proportions) * len(class_idx)).astype(int)[:-1]
        splits = np.split(class_idx, split_points)

        for client_id, split in enumerate(splits):
            client_indices[client_id].extend(split.tolist())

    return client_indices


def get_client_loaders(train_set, test_set, batch_size=BATCH_SIZE):
    client_indices = dirichlet_partition(train_set, NUM_CLIENTS, DIRICHLET_ALPHA)

    train_loaders = []
    for idx in client_indices:
        subset = Subset(train_set, idx)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True,
                            num_workers=0, pin_memory=False, drop_last=True)
        train_loaders.append(loader)

    test_loader = DataLoader(test_set, batch_size=batch_size * 2,
                             shuffle=False, num_workers=0, pin_memory=False)

    return train_loaders, test_loader, client_indices


def print_partition_stats(client_indices, train_set):
    labels = np.array([train_set[i][1] for i in range(len(train_set))])
    print(f"\n{'Client':<10} {'Samples':<10} Class Distribution")
    print("-" * 70)
    for i, idx in enumerate(client_indices):
        client_labels = labels[idx]
        dist = np.bincount(client_labels, minlength=10)
        dist_str = " ".join(f"{d:4d}" for d in dist)
        print(f"Client {i:<4} {len(idx):<10} {dist_str}")
    print()
