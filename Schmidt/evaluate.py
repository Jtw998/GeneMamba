#!/usr/bin/env python3
"""
Schmidt perturbation evaluation script.
Computes 6 metrics for each perturbed gene.

Metrics:
  Population-average: MSE, E-distance, PCC-delta
  Population-distribution: Wasserstein distance, KL-divergence, Common-DEGs

Usage:
  python3 evaluate.py --checkpoint ../checkpoints/hayat_checkpoint.pt
"""
import argparse
import torch
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from scipy.spatial.distance import cdist
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import Hayat
from utils import load_checkpoint


def _wasserstein_1d(u, v):
    return np.mean(np.abs(np.sort(u) - np.sort(v)))


def _kde_kl(p, q, n=500):
    p = p[p > 0]
    q = q[q > 0]
    if len(p) < 20 or len(q) < 20:
        return np.nan
    try:
        lo, hi = max(p.min(), q.min()), max(p.max(), q.max())
        xs = np.linspace(lo, hi, n)
        pk = gaussian_kde(p)(xs) + 1e-10
        qk = gaussian_kde(q)(xs) + 1e-10
        return np.sum(pk * np.log(pk / qk)) * (xs[1] - xs[0])
    except Exception:
        return np.nan


def compute_metrics(pred_mean, true_mat, ctrl_mean, pred_ctrl_mean):
    true_mean = true_mat.mean(axis=0)

    # 1. MSE
    mse = np.mean((true_mean - pred_mean) ** 2)

    # 2. E-distance
    n_sample = min(200, true_mat.shape[0])
    rng = np.random.default_rng(42)
    idx = rng.choice(true_mat.shape[0], n_sample, replace=False)
    pred_mat = np.tile(pred_mean, (n_sample, 1))
    ctrl_mat = np.tile(ctrl_mean, (n_sample, 1))
    inter = cdist(pred_mat, ctrl_mat).mean()
    intra = cdist(true_mat[idx], true_mat[idx]).mean()
    e_dist = 2 * inter - intra

    # 3. PCC-delta
    delta_pred = pred_mean - pred_ctrl_mean
    delta_true = true_mean - ctrl_mean
    mask = (np.std(true_mat, axis=0) > 1e-6) & (np.std(pred_mat, axis=0) > 1e-6)
    pcc_delta = np.corrcoef(delta_pred[mask], delta_true[mask])[0, 1] if mask.sum() > 10 else np.nan

    # 4. Wasserstein (mean over genes)
    w_vals = [_wasserstein_1d(true_mat[:, gi], pred_mat[:, gi]) for gi in range(pred_mean.shape[0])]
    wass = np.nanmean(w_vals)

    # 5. KL-divergence (mean over genes)
    kl_vals = [_kde_kl(pred_mat[:, gi], true_mat[:, gi]) for gi in range(pred_mean.shape[0])]
    kl_div = np.nanmean([v for v in kl_vals if not np.isnan(v)])

    # 6. Common-DEGs
    K = min(100, pred_mean.shape[0])
    true_fc = np.abs(true_mean - ctrl_mean)
    pred_fc = np.abs(pred_mean - pred_ctrl_mean)
    true_top = set(np.argsort(true_fc)[-K:])
    pred_top = set(np.argsort(pred_fc)[-K:])
    common_degs = len(true_top & pred_top) / K

    return {
        "MSE": mse,
        "E_distance": e_dist,
        "PCC_delta": pcc_delta if not np.isnan(pcc_delta) else 0.0,
        "Wasserstein": wass if not np.isnan(wass) else 0.0,
        "KL_divergence": kl_div if not np.isnan(kl_div) else 0.0,
        "Common_DEGs": common_degs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", default=".")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    print("Loading data...")
    data = torch.load(os.path.join(args.data_dir, "schmidt_data.pt"))
    pert_data = torch.load(os.path.join(args.data_dir, "schmidt_perturb_labels.pt"))
    expr = data["expression"]
    gene_names = data["gene_names"]
    all_pert = pert_data["perturbation"]

    print(f"Dataset: {expr.shape[0]} cells x {expr.shape[1]} genes")

    # Gene embeddings
    emb_path = os.path.join(args.data_dir, "schmidt_gene_embeddings.pt")
    gene_emb = torch.load(emb_path) if os.path.exists(emb_path) else None
    gene_emb_dim = gene_emb.shape[1] if gene_emb is not None else 512
    print(f"Embeddings: {'loaded' if gene_emb is not None else 'none'}")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    model = Hayat(
        num_genes=expr.shape[1],
        gene_emb_dim=gene_emb_dim,
        gene_emb=gene_emb,
        freeze_gene_emb=(gene_emb is not None),
        chrom_boundaries_path=os.path.join(args.data_dir, "schmidt_chrom_boundaries.pt"),
        hidden_dim=256,
        num_mamba_layers=2,
    )
    load_checkpoint(model, args.checkpoint, device)
    model = model.to(device)
    model.eval()

    expr_np = expr.numpy()
    all_pert_np = np.array(all_pert)

    # Control baseline
    ctrl_mask = all_pert_np == "control"
    ctrl_idx = np.where(ctrl_mask)[0]
    print(f"Control cells: {len(ctrl_idx)}")

    ctrl_cells = expr_np[ctrl_idx]
    ctrl_mean = ctrl_cells.mean(axis=0)

    # Model reconstruction of control
    print("Reconstructing control baseline...")
    with torch.no_grad():
        ctrl_tensor = torch.tensor(ctrl_cells, dtype=torch.float32)
        pred_ctrl_list = []
        for i in range(0, len(ctrl_tensor), args.batch_size):
            batch = ctrl_tensor[i:i+args.batch_size].to(device)
            pred, *_ = model(batch)
            pred_ctrl_list.append(pred.cpu().numpy())
        pred_ctrl_mean = np.concatenate(pred_ctrl_list, axis=0).mean(axis=0)

    pert_values = sorted(set(all_pert) - {"control"})
    print(f"Predicting {len(pert_values)} perturbed genes...")

    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    rows = []

    for pert_gene in pert_values:
        pert_idx = np.where(all_pert_np == pert_gene)[0]
        gi = gene_to_idx.get(pert_gene)
        if gi is None or len(pert_idx) < 5:
            continue

        true_mat = expr_np[pert_idx[:500]]
        true_mean = true_mat.mean(axis=0)

        pred_input = ctrl_mean.copy()
        pred_input[gi] = 0.0
        with torch.no_grad():
            pred_out, *_ = model(torch.tensor(pred_input, dtype=torch.float32).unsqueeze(0).to(device))
            pred_mean = pred_out.squeeze(0).cpu().numpy()

        metrics = compute_metrics(pred_mean, true_mat, ctrl_mean, pred_ctrl_mean)
        metrics["gene"] = pert_gene
        metrics["n_cells"] = len(pert_idx)
        rows.append(metrics)
        print(f"  {pert_gene}: MSE={metrics['MSE']:.4f}, PCC_delta={metrics['PCC_delta']:.3f}, "
              f"Wass={metrics['Wasserstein']:.4f}, KL={metrics['KL_divergence']:.3f}, "
              f"CommonDEGs={metrics['Common_DEGs']:.3f}")

    df = pd.DataFrame(rows)
    out_path = os.path.join(args.data_dir, "schmidt_metrics.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    metric_cols = ["MSE", "E_distance", "PCC_delta", "Wasserstein", "KL_divergence", "Common_DEGs"]
    print(f"\n===== Summary ({len(rows)} genes) =====")
    print(f"{'Metric':<20} {'Mean':>12} {'Std':>12}")
    print("-" * 46)
    for col in metric_cols:
        vals = df[col].dropna()
        print(f"{col:<20} {vals.mean():>12.4f} {vals.std():>12.4f}")


if __name__ == "__main__":
    main()
