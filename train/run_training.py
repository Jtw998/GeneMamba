#!/usr/bin/env python3
"""
Hayat training script.
Supports any dataset directory via GENE_DATA_DIR environment variable (set by train.py).
"""
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from typing import Dict, Tuple

from models import Hayat
from utils import compute_total_loss
from utils import config, calculate_metrics, save_checkpoint, load_checkpoint


def create_dataloaders(train_data, val_data, gene_names=None):
    # gene_names: list[str] in same order as expression tensor columns
    # We pass the SAME gene_names list to every batch (gene set is fixed during training)
    # so we only need one copy, not per-batch copies
    if gene_names is not None:
        train_loader = DataLoader(
            TensorDataset(train_data),
            batch_size=config["batch_size"], shuffle=True,
            collate_fn=lambda batch: (torch.stack([s[0] for s in batch]), gene_names),
        )
        val_loader = DataLoader(
            TensorDataset(val_data),
            batch_size=config["batch_size"], shuffle=False,
            collate_fn=lambda batch: (torch.stack([s[0] for s in batch]), gene_names),
        )
    else:
        train_loader = DataLoader(TensorDataset(train_data), batch_size=config["batch_size"], shuffle=True)
        val_loader = DataLoader(TensorDataset(val_data), batch_size=config["batch_size"], shuffle=False)
    return train_loader, val_loader


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    all_predictions, all_targets = [], []
    loss_components = {}

    for batch in tqdm(dataloader, desc="Training"):
        expression = batch[0].to(device)
        gene_names_list = batch[1] if len(batch) > 1 else None
        optimizer.zero_grad()
        with torch.autocast("mps", dtype=torch.bfloat16):
            predictions, latent_mean, latent_log_var, latent_sample = model(expression, gene_names_list=gene_names_list)
        loss, components = compute_total_loss(predictions, expression, latent_mean, latent_sample)
        if model.gene_embedding is not None and gene_names_list is not None:
            cached_gene_emb = model.get_cached_gene_emb(gene_names_list)
            gate_vals = model.regulator_gate.current_gate_vals(cached_gene_emb)
            sparsity_loss = 1e-6 * torch.sum(gate_vals ** 2)
        else:
            sparsity_loss = torch.tensor(0.0, device=device)
        total_loss_total = loss + sparsity_loss
        total_loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip_value"])
        optimizer.step()
        model.regulator_gate.clear_cache()

        total_loss += loss.item()
        all_predictions.append(predictions.detach().cpu())
        all_targets.append(expression.cpu())
        for k, v in components.items():
            loss_components[k] = loss_components.get(k, 0.0) + v
        loss_components["sparsity_loss"] = loss_components.get("sparsity_loss", 0.0) + sparsity_loss.item()
        del expression, predictions, latent_mean, latent_log_var, latent_sample, loss, components, sparsity_loss, total_loss_total

    avg_loss = total_loss / len(dataloader)
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = calculate_metrics(all_predictions, all_targets)
    for k in loss_components:
        loss_components[k] /= len(dataloader)
    return avg_loss, loss_components, metrics


def val_epoch(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    all_predictions, all_targets = [], []
    loss_components = {}
    with torch.no_grad(), torch.autocast("mps"):
        for batch in tqdm(dataloader, desc="Validation"):
            expression = batch[0].to(device)
            gene_names_list = batch[1] if len(batch) > 1 else None
            predictions, latent_mean, latent_log_var, _ = model(expression, gene_names_list=gene_names_list)
            loss, components = compute_total_loss(predictions, expression, latent_mean, latent_mean)
            total_loss += loss.item()
            all_predictions.append(predictions.cpu())
            all_targets.append(expression.cpu())
            for k, v in components.items():
                loss_components[k] = loss_components.get(k, 0.0) + v
    avg_loss = total_loss / len(dataloader)
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = calculate_metrics(all_predictions, all_targets)
    for k in loss_components:
        loss_components[k] /= len(dataloader)
    return avg_loss, loss_components, metrics


def train_model(model, train_data, val_data, gene_names, config, device, checkpoint_path):
    train_loader, val_loader = create_dataloaders(train_data, val_data, gene_names)
    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    best_val_loss = float("inf")
    for epoch in range(config["num_epochs"]):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        train_loss, train_components, train_metrics = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_components, val_metrics = val_epoch(model, val_loader, device)
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Train Pearson: {train_metrics['pearson']:.4f} | Val Pearson: {val_metrics['pearson']:.4f}")
        print(f"Loss components: {val_components}")
        num_regulators = 0
        if model.gene_embedding is not None and gene_names is not None:
            cached = model.get_cached_gene_emb(gene_names)
            if cached is not None:
                num_regulators = model.regulator_gate.num_active(cached, threshold=0.5)
        print(f"Number of currently identified regulatory factors: {num_regulators}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, checkpoint_path)
            print(f"Best model saved to {checkpoint_path}")
    load_checkpoint(model, checkpoint_path, device)
    return model


if __name__ == "__main__":
    data_dir = os.environ.get("GENE_DATA_DIR", "../data")
    is_schmidt = (data_dir == "Schmidt" or data_dir.endswith("Schmidt"))
    if is_schmidt:
        data_file = f"../{data_dir}/schmidt_data.pt"
        emb_file = f"../{data_dir}/schmidt_gene_embeddings.pt"
        chrom_file = f"../{data_dir}/schmidt_chrom_boundaries.pt"
    else:
        # Use 100k subset if available (smaller memory footprint)
        data_file_100k = f"{data_dir}/processed_data_100k.pt"
        if os.path.exists(data_file_100k):
            data_file = data_file_100k
        else:
            data_file = f"{data_dir}/processed_data.pt"
        emb_file = f"{data_dir}/gene_embeddings.pt"
        chrom_file = f"{data_dir}/chrom_boundaries.pt"

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Data directory: {data_dir}")

    data = torch.load(data_file)

    # Load gene names (for cross-dataset embedding lookup)
    gene_names = None
    if "gene_names" in data and data["gene_names"] is not None:
        # Stored as list of strings
        gene_names = data["gene_names"]
    elif os.path.exists(f"{data_dir}/gene_meta.csv"):
        import pandas as pd
        meta = pd.read_csv(f"{data_dir}/gene_meta.csv")
        gene_names = meta["gene_name"].tolist()
        print(f"Loaded {len(gene_names)} gene names from gene_meta.csv")
    elif os.path.exists(f"../{data_dir}/schmidt_gene_meta.csv"):
        import pandas as pd
        meta = pd.read_csv(f"../{data_dir}/schmidt_gene_meta.csv")
        gene_names = meta["gene_name"].tolist()
        print(f"Loaded {len(gene_names)} gene names from schmidt_gene_meta.csv")

    if is_schmidt:
        expr = data["expression"]
        max_cells = config.get("max_train_cells", 0)
        if max_cells > 0 and expr.shape[0] > max_cells:
            expr = expr[:max_cells]
            print(f"[Demo mode] Subsampled to {expr.shape[0]} cells")
        split = int(expr.shape[0] * 0.8)
        rng = torch.Generator()
        indices = torch.randperm(expr.shape[0], generator=rng)
        train_data = expr[indices[:split]]
        val_data = expr[indices[split:]]
    else:
        train_data = data["train"]
        val_data = data["val"]
        max_cells = config.get("max_train_cells", 0)
        if max_cells > 0:
            if train_data.shape[0] > max_cells:
                train_data = train_data[:max_cells]
                print(f"[Demo mode] Subsampled training set to {train_data.shape[0]} cells")
            val_max = max(max_cells // 5, 200)  # val ~20% of train, min 200
            if val_data.shape[0] > val_max:
                val_data = val_data[:val_max]
                print(f"[Demo mode] Subsampled validation set to {val_data.shape[0]} cells")

    if os.path.exists(emb_file):
        gene_emb = torch.load(emb_file)
    else:
        gene_emb = None
        print(f"No gene embeddings at {emb_file}, using zero embeddings")

    assert gene_emb is None or gene_emb.shape[0] == train_data.shape[1], \
        f"Gene embeddings {gene_emb.shape[0]} != genes in data {train_data.shape[1]}"
    print(f"Training set: {train_data.shape[0]} cells x {train_data.shape[1]} genes")
    print(f"Validation set: {val_data.shape[0]} cells x {val_data.shape[1]} genes")

    pos_table_path = "../position_table.pt"
    if not os.path.exists(pos_table_path):
        pos_table_path = None
        print("Position table not found, using fallback Fourier encoder")

    model = Hayat(
        num_genes=train_data.shape[1],
        gene_emb_dim=gene_emb.shape[1] if gene_emb is not None else 512,
        gene_emb=gene_emb,
        gene_names=gene_names,
        freeze_gene_emb=(gene_emb is not None),
        chrom_boundaries_path=chrom_file,
        position_table_path=pos_table_path,
        hidden_dim=config["hidden_dim"],
        num_mamba_layers=config["num_mamba_layers"]
    )
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    print(f"\nModel parameter statistics:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {frozen_params:,}")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB")

    ckpt_name = f"hayat_{data_dir.replace('/', '_')}.pt"
    ckpt_path = f"../checkpoints/{ckpt_name}"
    print(f"Checkpoint: {ckpt_path}")
    train_model(model, train_data, val_data, gene_names, config, device=device, checkpoint_path=ckpt_path)
