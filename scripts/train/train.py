"""
Training loop for QM9 dataset.
"""


import torch


def train_one_epoch(
        model, train_loader, optimizer, criterion, device,
        train_mean, train_std, target_idx, model_type="gcn"):
    """
    Train one epoch for GCN model.
    Args:
        model: The GCN model to train.
        train_loader: DataLoader for training data.
        optimizer: Optimizer for model parameters.
        criterion: Loss function.
        device: Device to run the training on (CPU or GPU).
        train_mean: Mean of the target property in the training set.
        train_std: Standard deviation of the target property in the training set.
        target_idx: Index of the target property in the dataset.
        model_type: Type of the model ("gcn", "nnconv" or "dimenet").
    Returns:
        Average loss over the training epoch.
    """
    model.train()
    running_loss = 0.0
    for data in train_loader:
        optimizer.zero_grad()
        data = data.to(device)

        if model_type == "gcn":
            model_input = (data.x, data.edge_index, data.batch)
        elif model_type == "nnconv":
            model_input = (data.x, data.edge_index, data.batch, data.edge_attr)
        elif model_type == "dimenet":
            model_input = (data.z, data.pos, data.batch)
        out = model(*model_input).view(-1)
        y_norm = (data.y[:, target_idx] - train_mean) / train_std

        loss = criterion(out, y_norm)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running_loss += loss.item() * y_norm.size(0)
    return running_loss / len(train_loader.dataset)

    