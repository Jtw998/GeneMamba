import torch
import torch.nn.functional as F
from utils import config

def negative_binomial_loss(predictions: torch.Tensor, targets: torch.Tensor, theta: float = 10.0) -> torch.Tensor:
    """
    Numerically stable negative binomial log-likelihood loss for scRNA-seq count data.
    Args:
        predictions: [batch_size, num_genes] predicted expression values (non-negative)
        targets:    [batch_size, num_genes] ground truth expression counts
        theta:      negative binomial dispersion parameter
    Returns:
        loss: scalar
    """
    theta = torch.tensor(theta, device=predictions.device, dtype=predictions.dtype)

    predictions = F.softplus(predictions)

    t1 = torch.lgamma(targets + theta) - torch.lgamma(targets + 1) - torch.lgamma(theta)
    t2 = theta * torch.log(theta / (theta + predictions + 1e-8))
    t3 = targets * torch.log(predictions / (theta + predictions + 1e-8))
    log_likelihood = t1 + t2 + t3

    return -torch.mean(log_likelihood)

def sparsity_loss(latent_mean: torch.Tensor) -> torch.Tensor:
    """
    L1 sparsity regularization on latent variables to encourage sparse representations.
    Args:
        latent_mean: [batch_size, num_genes, latent_dim] latent variable mean
    Returns:
        loss: scalar
    """
    return torch.mean(torch.abs(latent_mean))

def decoupling_loss(latent_sample: torch.Tensor) -> torch.Tensor:
    """
    Penalizes correlation between latent dimensions to encourage independence.
    Args:
        latent_sample: [batch_size, num_genes, latent_dim] sampled latent variables
    Returns:
        loss: scalar
    """
    batch_size, num_genes, latent_dim = latent_sample.shape
    latent_flat = latent_sample.reshape(-1, latent_dim)

    # Covariance matrix
    mean = torch.mean(latent_flat, dim=0, keepdim=True)
    centered = latent_flat - mean
    covariance = torch.matmul(centered.T, centered) / (latent_flat.shape[0] - 1)

    # Penalize off-diagonal elements
    diagonal = torch.diag(covariance)
    off_diagonal = covariance - torch.diag_embed(diagonal)

    return torch.mean(torch.abs(off_diagonal))

def smoothness_loss(predictions: torch.Tensor) -> torch.Tensor:
    """
    Penalizes differences between adjacent gene predictions to encourage smooth expression profiles.
    Args:
        predictions: [batch_size, num_genes] predicted expression values
    Returns:
        loss: scalar
    """
    diff = predictions[:, 1:] - predictions[:, :-1]
    return torch.mean(torch.abs(diff))

def compute_total_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    latent_mean: torch.Tensor,
    latent_sample: torch.Tensor
) -> torch.Tensor:
    """
    Weighted combination of all loss terms.
    """
    nb_loss = negative_binomial_loss(predictions, targets) * config["nb_loss_weight"]
    sparse_loss = sparsity_loss(latent_mean) * config["sparsity_loss_weight"]
    decouple_loss = decoupling_loss(latent_sample) * config["decoupling_loss_weight"]
    smooth_loss = smoothness_loss(predictions) * config["smoothness_loss_weight"]

    total_loss = nb_loss + sparse_loss + decouple_loss + smooth_loss
    return total_loss, {
        "nb_loss": nb_loss.item(),
        "sparsity_loss": sparse_loss.item(),
        "decoupling_loss": decouple_loss.item(),
        "smoothness_loss": smooth_loss.item()
    }
