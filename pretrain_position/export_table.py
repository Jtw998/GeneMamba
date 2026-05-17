"""
Export pretrained Fourier encoder as a lookup table.

Loads best_model.pt, computes position embeddings for all genes,
and saves position_table.pt for use in Hayat PositionEncoder.

Output: position_table.pt
  {
    "table": {gene_name: embedding_tensor},   # dict[str → Tensor[embed_dim]]
    "embed_dim": int,
    "num_genes": int,
    "metadata": {gene_name: {chrom, tss}},   # for reference
  }
"""

import json
import pickle
import numpy as np
import torch
from pathlib import Path

from fourier_encoder import FourierPositionEncoder


def export_lookup_table(
    data_dir: str = None,
    checkpoint_path: str = None,
    output_path: str = None,
):
    if data_dir is None:
        data_dir = Path(__file__).parent
    else:
        data_dir = Path(data_dir)

    if checkpoint_path is None:
        checkpoint_path = data_dir / "checkpoints" / "best_model.pt"
    if output_path is None:
        output_path = data_dir.parent / "position_table.pt"

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg = ckpt["config"]

    # Reconstruct encoder
    encoder = FourierPositionEncoder(
        input_dim=cfg["input_dim"],
        embed_dim=cfg["embed_dim"],
        num_frequencies=cfg["num_frequencies"],
        f0=cfg["f0"],
        sigma=cfg["sigma"],
    )
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    encoder.eval()

    # Load gene coords
    coords = np.load(data_dir / "gene_coords.npy")

    with open(data_dir / "gene_to_idx.json") as f:
        gene_to_idx = json.load(f)

    with open(data_dir / "gene_list.json") as f:
        gene_list = json.load(f)

    with open(data_dir / "gene_coords_full.pkl", "rb") as f:
        gene_coords_full = pickle.load(f)

    # Compute all embeddings
    with torch.no_grad():
        all_embeddings = encoder(torch.from_numpy(coords).float())
    all_embeddings = all_embeddings.cpu().numpy()

    # Build table
    table = {}
    metadata = {}
    for gene_name in gene_list:
        idx = gene_to_idx[gene_name]
        table[gene_name] = torch.from_numpy(all_embeddings[idx]).float()
        info = gene_coords_full.get(gene_name, {})
        metadata[gene_name] = {
            "chrom": info.get("chr", ""),
            "tss": info.get("start", 0),
        }

    # Save
    save_dict = {
        "table": table,
        "embed_dim": cfg["embed_dim"],
        "num_genes": len(table),
        "metadata": metadata,
    }
    torch.save(save_dict, output_path)
    print(f"\n✅ Position table exported!")
    print(f"   Genes: {len(table)}")
    print(f"   Embed dim: {cfg['embed_dim']}")
    print(f"   File: {output_path}")
    print(f"   Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Quick sanity check: nearby genes should have similar embeddings
    print("\n--- Embedding quality check ---")
    test_pairs = [
        ("TP53", "WRAP53"),   # adjacent on chr17
        ("TP53", "BRCA1"),    # both chr17 but far apart
    ]
    for g1, g2 in test_pairs:
        if g1 in table and g2 in table:
            sim = torch.nn.functional.cosine_similarity(
                table[g1].unsqueeze(0), table[g2].unsqueeze(0)
            ).item()
            print(f"  {g1} ↔ {g2}: cos_sim = {sim:.4f}")


if __name__ == "__main__":
    export_lookup_table()
