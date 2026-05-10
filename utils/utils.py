import torch
import numpy as np
from typing import Dict, Tuple

# Centralized hyperparameter configuration
config: Dict = {
    # Model parameters
    "hidden_dim": 256,
    "num_mamba_layers": 2,
    "latent_dim": 64,
    "dropout_rate": 0.1,
    "causal_gate_temperature": 0.1,

    # Training parameters
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "grad_clip_value": 1.0,
    "num_epochs": 100,
    "batch_size": 16,
    "validation_split": 0.2,

    # Loss weights
    "nb_loss_weight": 1.0,
    "sparsity_loss_weight": 0.5,
    "decoupling_loss_weight": 0.0,   # disabled
    "smoothness_loss_weight": 0.0,  # disabled

    # Inference parameters
    "perturbation_value": 10.0,   # overexpression value
    "knockout_value": 0.0,      # knockout value

    # Demo parameters
    "max_train_cells": 100000,   # limit training cells, 0 = use all
}

def normalize_expression(expression: torch.Tensor) -> torch.Tensor:
    """
    Normalize gene expression values to [0, 1] range.
    Args:
        expression: [batch_size, num_genes] raw expression matrix
    Returns:
        normalized: [batch_size, num_genes] normalized expression matrix
    """
    min_val = torch.min(expression, dim=1, keepdim=True)[0]
    max_val = torch.max(expression, dim=1, keepdim=True)[0]
    normalized = (expression - min_val) / (max_val - min_val + 1e-8)
    return normalized

def calculate_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """
    Compute model performance metrics.
    Args:
        predictions: [batch_size, num_genes] predicted expression values
        targets:     [batch_size, num_genes] ground truth expression values
    Returns:
        metrics: dict with MSE, MAE, Pearson correlation
    """
    mse = torch.mean((predictions - targets) ** 2).item()
    mae = torch.mean(torch.abs(predictions - targets)).item()

    # Pearson correlation coefficient
    target_mean = torch.mean(targets, dim=1, keepdim=True)
    pred_mean = torch.mean(predictions, dim=1, keepdim=True)
    covariance = torch.mean((targets - target_mean) * (predictions - pred_mean), dim=1)
    target_std = torch.std(targets, dim=1)
    pred_std = torch.std(predictions, dim=1)
    pearson = torch.mean(covariance / (target_std * pred_std + 1e-8)).item()

    return {
        "mse": mse,
        "mae": mae,
        "pearson": pearson
    }

def save_checkpoint(model: torch.nn.Module, path: str) -> None:
    """Save model checkpoint."""
    torch.save(model.state_dict(), path)

def load_checkpoint(model: torch.nn.Module, path: str, device: torch.device) -> None:
    """Load model checkpoint. Uses strict=False to gracefully skip mismatched keys
    (e.g. old regulator_gate Parameter vs new RegulatorGate module)."""
    model.load_state_dict(torch.load(path, map_location=device), strict=False)
