"""
Prepare gene coordinate data for Fourier position encoder pretraining.
Reads from gene_meta.csv files, filters to standard chromosomes,
generates positive/negative gene pairs based on genomic distance.

Input:  data/gene_meta.csv or Schmidt/schmidt_gene_meta.csv
Output: pretrain_position/gene_coords.npy
        pretrain_position/gene_to_idx.json
        pretrain_position/gene_list.json
        pretrain_position/pairs.json
        pretrain_position/gene_coords_full.pkl
"""

import json
import pickle
import random
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

CONFIG = {
    "pos_threshold": 100_000,   # < 100kb → positive
    "neg_threshold": 1_000_000,  # > 1Mb → negative
    "neg_ratio": 3,              # negative : positive ratio
    "max_pairs": 500_000,
    "seed": 42,
    "min_genes_per_chr": 5,    # skip chromosomes with fewer genes
}

STANDARD_CHRS = {str(i) for i in range(1, 23)} | {"X", "Y", "MT"}


def load_gene_meta(csv_path: str) -> dict:
    """Load gene_meta.csv → {gene_name: {chr, start}}"""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    # Normalize chromosome names
    df["chr"] = df["chr"].astype(str).str.strip()

    # Filter to standard chromosomes
    df = df[df["chr"].isin(STANDARD_CHRS)]
    df["start"] = df["start"].astype(int)

    coords = {}
    for _, row in df.iterrows():
        coords[row["gene_name"]] = {
            "chr": row["chr"],
            "start": row["start"],
        }
    print(f"  Loaded {len(coords)} genes from {csv_path}")
    return coords


def build_dataset(gene_coords: dict, config: dict) -> tuple:
    """Generate positive/negative gene pairs by linear genomic distance."""
    pos_threshold = config["pos_threshold"]
    neg_threshold = config["neg_threshold"]
    neg_ratio = config["neg_ratio"]
    max_pairs = config["max_pairs"]
    seed = config["seed"]

    random.seed(seed)

    # Group by chromosome, sort by position
    chrom_genes = defaultdict(list)
    for gene, info in gene_coords.items():
        chrom_genes[info["chr"]].append((gene, info["start"]))

    for chrom in chrom_genes:
        chrom_genes[chrom].sort(key=lambda x: x[1])

    # Filter chromosomes with too few genes
    chrom_genes = {
        c: genes for c, genes in chrom_genes.items()
        if len(genes) >= config["min_genes_per_chr"]
    }

    pos_pairs, neg_pairs = [], []

    for chrom, genes in chrom_genes.items():
        n = len(genes)
        for i in range(n):
            for j in range(i + 1, n):
                g1, p1 = genes[i]
                g2, p2 = genes[j]
                dist = abs(p1 - p2)

                if dist < pos_threshold:
                    pos_pairs.append((g1, g2, 1, dist, chrom))
                elif dist > neg_threshold:
                    neg_pairs.append((g1, g2, 0, dist, chrom))

                if dist > neg_threshold * 3:
                    break  # sorted → further genes are even farther

    print(f"  Positive pairs: {len(pos_pairs)}")
    print(f"  Negative pairs: {len(neg_pairs)}")

    # Sample negatives
    if len(neg_pairs) > len(pos_pairs) * neg_ratio:
        random.shuffle(neg_pairs)
        neg_pairs = neg_pairs[: len(pos_pairs) * neg_ratio]

    all_pairs = pos_pairs + neg_pairs
    random.shuffle(all_pairs)

    if len(all_pairs) > max_pairs:
        all_pairs = all_pairs[:max_pairs]

    pos_n = sum(1 for p in all_pairs if p[2] == 1)
    neg_n = len(all_pairs) - pos_n
    print(f"  Final: {len(all_pairs)} pairs (pos={pos_n}, neg={neg_n})")
    return all_pairs


def build_gene_index(gene_coords: dict, pairs: list) -> tuple:
    """Build gene_name → integer index."""
    all_genes = set()
    for g1, g2, *_ in pairs:
        all_genes.add(g1)
        all_genes.add(g2)

    gene_list = sorted(all_genes)
    gene_to_idx = {g: i for i, g in enumerate(gene_list)}
    print(f"  Unique genes: {len(gene_list)}")
    return gene_list, gene_to_idx


def encode_coords(gene_coords: dict, gene_to_idx: dict) -> np.ndarray:
    """Encode gene coordinates as [N, 3] float array: [chrom_idx, tss_norm, log_tss_norm]."""
    chrom_list = [str(i) for i in range(1, 23)] + ["X", "Y", "MT"]
    chrom_to_idx = {c: i for i, c in enumerate(chrom_list)}

    max_tss = max(info["start"] for info in gene_coords.values())

    n = len(gene_to_idx)
    coords = np.zeros((n, 3), dtype=np.float32)

    for gene, idx in gene_to_idx.items():
        info = gene_coords[gene]
        chrom_idx = chrom_to_idx.get(info["chr"], 0)
        tss = info["start"]

        coords[idx, 0] = chrom_idx / len(chrom_list)
        coords[idx, 1] = tss / max_tss
        coords[idx, 2] = np.log1p(tss) / np.log1p(max_tss)

    return coords, chrom_to_idx


def main():
    output_dir = Path(__file__).parent
    output_dir.mkdir(exist_ok=True)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)

    # Collect all gene coordinates from all available gene_meta.csv files
    all_gene_coords = {}

    data_dir = Path(__file__).parent.parent

    meta_files = [
        data_dir / "data" / "gene_meta.csv",
        data_dir / "Schmidt" / "schmidt_gene_meta.csv",
    ]

    for path in meta_files:
        if path.exists():
            coords = load_gene_meta(str(path))
            # Merge (overwrite with last if duplicate)
            all_gene_coords.update(coords)

    print(f"Total unique genes: {len(all_gene_coords)}")

    # Deduplicate by gene name (keep first occurrence)
    seen = {}
    for name, info in list(all_gene_coords.items()):
        if name not in seen:
            seen[name] = info
    all_gene_coords = seen
    print(f"After deduplication: {len(all_gene_coords)}")

    # Build pairs
    pairs = build_dataset(all_gene_coords, CONFIG)

    # Build index
    gene_list, gene_to_idx = build_gene_index(all_gene_coords, pairs)

    # Encode coordinates
    coords, chrom_to_idx = encode_coords(all_gene_coords, gene_to_idx)

    # Save
    np.save(output_dir / "gene_coords.npy", coords)

    with open(output_dir / "gene_to_idx.json", "w") as f:
        json.dump(gene_to_idx, f)

    with open(output_dir / "gene_list.json", "w") as f:
        json.dump(gene_list, f)

    with open(output_dir / "chrom_to_idx.json", "w") as f:
        json.dump(chrom_to_idx, f)

    # Save pairs as indices
    pair_indices = []
    for g1, g2, label, dist, chrom in pairs:
        pair_indices.append({
            "i": gene_to_idx[g1],
            "j": gene_to_idx[g2],
            "label": label,
            "dist": dist,
            "chrom": chrom,
        })

    with open(output_dir / "pairs.json", "w") as f:
        json.dump(pair_indices, f)

    # Save full coords for export_table
    with open(output_dir / "gene_coords_full.pkl", "wb") as f:
        pickle.dump(all_gene_coords, f)

    with open(output_dir / "prep_config.json", "w") as f:
        json.dump(CONFIG, f, indent=2)

    print(f"\n✅ Data prepared! Saved to {output_dir}/")
    print(f"   gene_coords.npy    : {coords.shape}")
    print(f"   gene_to_idx.json   : {len(gene_to_idx)} genes")
    print(f"   pairs.json         : {len(pair_indices)} pairs")
    print(f"   gene_coords_full.pkl: {len(all_gene_coords)} genes (full coords)")


if __name__ == "__main__":
    main()
