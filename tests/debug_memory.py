#!/usr/bin/env python3
"""
Debug memory usage of GeneMamba model, printing memory at each stage.
"""
import torch
import sys
from models import GeneMamba

def print_memory(stage: str):
    """Print current GPU memory usage."""
    if torch.backends.mps.is_available():
        allocated = torch.mps.current_allocated_memory() / 1024**3
        reserved = torch.mps.driver_allocated_memory() / 1024**3
        print(f"[{stage}] MPS allocated: {allocated:.2f} GB | reserved: {reserved:.2f} GB")
    elif torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[{stage}] CUDA allocated: {allocated:.2f} GB | reserved: {reserved:.2f} GB")
    else:
        print(f"[{stage}] CPU mode, no GPU memory")

def main():
    batch_size = 4
    num_genes = 21900
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    # 1. Model init
    print_memory("before_model_init")
    model = GeneMamba(num_genes=num_genes)
    model = model.to(device)
    print(f"max_regulators: {model.max_regulators}")
    print_memory("after_model_init")

    # 2. Test input
    x = torch.randn(batch_size, num_genes, device=device)
    print_memory("input_loaded")

    # 3. Forward pass (with grad, simulating training)
    model.train()
    print_memory("before_forward")
    pred, latent_mean, latent_log_var, latent_sample = model(x)
    print_memory("after_forward")
    print(f"Output shapes: pred={pred.shape}, latent={latent_mean.shape}")

    # 4. Backward pass
    loss = pred.sum()
    loss.backward()
    print_memory("after_backward")

    # 5. Trans branch gate inspection (RegulatorGate is a module, not a tensor)
    if hasattr(model, 'regulator_gate') and hasattr(model.regulator_gate, 'mlp'):
        print(f"\nTrans branch inspection:")
        print(f"  max_regulators: {model.max_regulators}")
        print(f"  RegulatorGate type: module (MLP-based, cross-dataset compatible)")
    else:
        print(f"\nTrans branch inspection:")
        print(f"  regulator_gate type: tensor (legacy Parameter-based)")

    print("\nDone")

if __name__ == "__main__":
    main()
