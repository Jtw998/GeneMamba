import torch
import torch.nn.functional as F
from utils import config

def negative_binomial_loss(predictions: torch.Tensor, targets: torch.Tensor, theta: float = 10.0) -> torch.Tensor:
    """
    数值稳定的负二项似然损失（适用于scRNA-seq计数数据）
    Args:
        predictions: [batch_size, num_genes] 预测的表达值（非负）
        targets: [batch_size, num_genes] 真实的表达计数
        theta: 负二项分布的离散参数
    Returns:
        loss: 标量损失值
    """
    # 把theta转成和输入同设备的tensor，避免类型错误
    theta = torch.tensor(theta, device=predictions.device, dtype=predictions.dtype)

    # 确保预测值非负
    predictions = F.softplus(predictions)

    # 数值稳定的负二项对数似然计算
    t1 = torch.lgamma(targets + theta) - torch.lgamma(targets + 1) - torch.lgamma(theta)
    t2 = theta * torch.log(theta / (theta + predictions + 1e-8))
    t3 = targets * torch.log(predictions / (theta + predictions + 1e-8))
    log_likelihood = t1 + t2 + t3

    return -torch.mean(log_likelihood)

def sparsity_loss(latent_mean: torch.Tensor) -> torch.Tensor:
    """
    稀疏损失（L1正则化隐变量，鼓励稀疏表示）
    Args:
        latent_mean: [batch_size, num_genes, latent_dim] 隐变量均值
    Returns:
        loss: 标量损失值
    """
    return torch.mean(torch.abs(latent_mean))

def decoupling_loss(latent_sample: torch.Tensor) -> torch.Tensor:
    """
    解耦损失（鼓励隐变量各维度独立）
    Args:
        latent_sample: [batch_size, num_genes, latent_dim] 采样的隐变量
    Returns:
        loss: 标量损失值
    """
    batch_size, num_genes, latent_dim = latent_sample.shape
    latent_flat = latent_sample.reshape(-1, latent_dim)

    # 计算协方差矩阵
    mean = torch.mean(latent_flat, dim=0, keepdim=True)
    centered = latent_flat - mean
    covariance = torch.matmul(centered.T, centered) / (latent_flat.shape[0] - 1)

    # 惩罚非对角元素（协方差）
    diagonal = torch.diag(covariance)
    off_diagonal = covariance - torch.diag_embed(diagonal)

    return torch.mean(torch.abs(off_diagonal))

def smoothness_loss(predictions: torch.Tensor) -> torch.Tensor:
    """
    平滑损失（鼓励相邻基因的预测表达值平滑变化）
    Args:
        predictions: [batch_size, num_genes] 预测的表达值
    Returns:
        loss: 标量损失值
    """
    # 计算相邻基因的差值
    diff = predictions[:, 1:] - predictions[:, :-1]
    return torch.mean(torch.abs(diff))

def compute_total_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    latent_mean: torch.Tensor,
    latent_sample: torch.Tensor
) -> torch.Tensor:
    """
    计算总损失（加权组合所有损失项）
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
