import torch
import numpy as np
from typing import Dict, Tuple

# 统一超参数配置
config: Dict = {
    # 模型参数
    "hidden_dim": 256,
    "num_mamba_layers": 2,
    "latent_dim": 64,
    "dropout_rate": 0.1,
    "causal_gate_temperature": 0.1,

    # 训练参数
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "grad_clip_value": 1.0,
    "num_epochs": 100,
    "batch_size": 16,
    "validation_split": 0.2,

    # 损失权重：双损失组合，仅保留NB主损失+稀疏损失
    "nb_loss_weight": 1.0,
    "sparsity_loss_weight": 0.5,
    "decoupling_loss_weight": 0.0,  # 关闭解耦损失
    "smoothness_loss_weight": 0.0,  # 关闭平滑损失

    # 推理参数
    "perturbation_value": 10.0,  # 过表达值
    "knockout_value": 0.0,       # 敲除值

    # Demo 参数
    "max_train_cells": 100000,  # 限制训练细胞数，0=使用全部数据
}

def normalize_expression(expression: torch.Tensor) -> torch.Tensor:
    """
    归一化基因表达值到[0, 1]范围
    Args:
        expression: [batch_size, num_genes] 原始表达矩阵
    Returns:
        normalized: [batch_size, num_genes] 归一化后表达矩阵
    """
    min_val = torch.min(expression, dim=1, keepdim=True)[0]
    max_val = torch.max(expression, dim=1, keepdim=True)[0]
    normalized = (expression - min_val) / (max_val - min_val + 1e-8)
    return normalized

def calculate_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """
    计算模型性能指标
    Args:
        predictions: [batch_size, num_genes] 预测表达值
        targets: [batch_size, num_genes] 真实表达值
    Returns:
        metrics: 包含MSE, MAE, Pearson相关系数的字典
    """
    mse = torch.mean((predictions - targets) ** 2).item()
    mae = torch.mean(torch.abs(predictions - targets)).item()

    # 计算皮尔逊相关系数
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
    """保存模型 checkpoint"""
    torch.save(model.state_dict(), path)

def load_checkpoint(model: torch.nn.Module, path: str, device: torch.device) -> None:
    """加载模型 checkpoint"""
    model.load_state_dict(torch.load(path, map_location=device))
