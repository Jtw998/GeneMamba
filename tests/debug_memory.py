#!/usr/bin/env python3
"""
调试model2显存占用，逐段打印内存使用
"""
import torch
import sys
from models import GeneMambaV0_1

def print_memory(stage: str):
    """打印当前显存占用"""
    if torch.backends.mps.is_available():
        allocated = torch.mps.current_allocated_memory() / 1024**3
        reserved = torch.mps.driver_allocated_memory() / 1024**3
        print(f"[{stage}] MPS实际占用: {allocated:.2f} GB | 系统预留: {reserved:.2f} GB")
    elif torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[{stage}] CUDA实际占用: {allocated:.2f} GB | 系统预留: {reserved:.2f} GB")
    else:
        print(f"[{stage}] CPU模式，无GPU显存")

def main():
    # 配置
    batch_size = 4
    num_genes = 21900
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 1. 初始化模型
    print_memory("模型初始化前")
    model = GeneMambaV0_1(num_genes=num_genes)
    model = model.to(device)
    print(f"max_regulators设置: {model.max_regulators}")
    print_memory("模型初始化后")

    # 2. 测试输入
    x = torch.randn(batch_size, num_genes, device=device)
    print_memory("输入加载完成")

    # 3. 前向传播测试（带梯度，模拟训练）
    model.train()
    print_memory("前向开始前")
    pred, latent_mean, latent_log_var, latent_sample = model(x)
    print_memory("前向结束后")
    print(f"输出形状正常: pred={pred.shape}, latent={latent_mean.shape}")

    # 4. 反向传播测试
    loss = pred.sum()
    loss.backward()
    print_memory("反向结束后")

    # 5. 检查反式分支K值
    regulator_gate_vals = torch.sigmoid(model.regulator_gate)
    topk_vals, topk_idx = torch.topk(regulator_gate_vals, model.max_regulators)
    print(f"\n反式分支验证:")
    print(f"  理论K值: {model.max_regulators}")
    print(f"  实际活跃K值: {len(topk_idx)}")
    print(f"  门控值范围: [{regulator_gate_vals.min().item():.3f}, {regulator_gate_vals.max().item():.3f}]")
    print(f"  门控>0.5的数量: {(regulator_gate_vals > 0.5).sum().item()}")

    print("\n✅ 调试完成")

if __name__ == "__main__":
    main()
