"""
Utility function to describe a PyTorch Geometric dataset.
"""


import torch
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader


def load_qm9_dataset(root):
    """
    Load the QM9 dataset and return it.
    Args:
        root (str): The root directory where the dataset will be stored.
    Returns:
        dataset (torch_geometric.data.Dataset): The loaded QM9 dataset.
    """
    dataset = QM9(root=root)
    return dataset


def describe_dataset(dataset):
    """
    Print out basic information about a PyTorch Geometric dataset.
    Args:
        dataset (torch_geometric.data.Dataset): The dataset to describe.
    """

    print(f'Dataset: {dataset}:')
    print('[====================]')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')
    print()

    print('First node data:')
    print('[====================]')
    data = dataset[0]
    print(data)
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Has isolated nodes: {data.has_isolated_nodes()}')
    print(f'Has self-loops: {data.has_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')


def split_dataset(dataset, training_perc=0.9):
    """
    Split the dataset into training and test sets.
    Args:
        dataset (torch_geometric.data.Dataset): The dataset to split.
        training_perc (float): Percentage of data to use for training (default
            is 0.9).
    Returns:
        train_dataset (torch_geometric.data.Dataset): Training dataset.
        test_dataset (torch_geometric.data.Dataset): Test dataset.
    """
    dataset = dataset.shuffle()

    train_dataset = dataset[:int(len(dataset) * training_perc)]
    test_dataset = dataset[int(len(dataset) * training_perc):]

    print(f'Number of training graphs: {len(train_dataset)}')
    print(f'Number of test graphs: {len(test_dataset)}')

    return train_dataset, test_dataset


def normalize_train_set(train_dataset, target_idx=11):
    """
    Compute the mean and standard deviation of the target property in the
    training dataset.
    Args:
        train_dataset (torch_geometric.data.Dataset): The training dataset.
        target_idx (int): The index of the target property in the y tensor
            (default is 11 for QM9).
    Returns:
        train_mean (float): Mean of the target property in the training set.
        train_std (float): Standard deviation of the target property in the
            training set.
    """
    ys_train = torch.stack([d.y for d in train_dataset])[:, 0, target_idx]
    train_mean = ys_train.mean()
    train_std = ys_train.std()
    if train_std == 0:
        train_std = 1.0

    print(f"Target statistics - Mean: {train_mean:.4f}, Std: {train_std:.4f}")

    return train_mean, train_std