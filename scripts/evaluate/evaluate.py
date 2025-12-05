"""
Evaluate the model on a given dataset.
"""


import torch


def evaluate(model, loader, device, train_mean, train_std, target_idx,
             model_type="gcn"):
    """
    Evaluate the model on the given data loader.
    Args:
        model: The model to evaluate.
        loader: DataLoader for evaluation data.
        device: Device to run the evaluation on (CPU or GPU).
        train_mean: Mean of the target property in the training set.
        train_std: Standard deviation of the target property in the training set.
        target_idx: Index of the target property in the dataset.
        model_type: Type of the model ("gcn" or "dimenet").
    Returns:
        Mean Absolute Error (MAE) over the evaluation dataset.
    """
    model.eval()
    with torch.no_grad():
        mae = 0.0
        n = 0
        for data in loader:
            data = data.to(device)
            if model_type == "gcn":
                model_input = (data.x, data.edge_index, data.batch)
            elif model_type == "nnconv":
                model_input = (data.x, data.edge_index, data.batch, data.edge_attr)
            elif model_type == "dimenet":
                model_input = (data.z, data.pos, data.batch)
            out = model(*model_input).view(-1)
            out_denorm = out * train_std + train_mean
            y = data.y[:, target_idx]
            mae += torch.sum(torch.abs(out_denorm - y)).item()
            n += y.size(0)
    return mae / n


def compare_ytrue_ypred(model, loader, device, train_mean, train_std, 
                            target_idx, model_type="gcn"):
    """
    Compare true and predicted values from the model on the given data loader.
    Args:
        model: The model to evaluate.
        loader: DataLoader for evaluation data.
        device: Device to run the evaluation on (CPU or GPU).
        train_mean: Mean of the target property in the training set.
        train_std: Standard deviation of the target property in the training set.
        target_idx: Index of the target property in the dataset.
        model_type: Type of the model ("gcn" or "dimenet").
    Returns:
        Two lists: true values and predicted values.
    """
    model.eval()
    ypred = []
    ytrue = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            if model_type == "gcn":
                model_input = (data.x, data.edge_index, data.batch)
            elif model_type == "nnconv":
                model_input = (data.x, data.edge_index, data.batch, data.edge_attr)
            elif model_type == "dimenet":
                model_input = (data.z, data.pos, data.batch)
            out = model(*model_input).view(-1)
            out_denorm = out * train_std + train_mean
            y = data.y[:, target_idx]
            ypred.append(out_denorm)
            ytrue.append(y)
    return ytrue, ypred