"""
Core scGPT gene embedding matching: extracts scGPT embeddings for a gene list.
Reusable: run with --scgpt_dir pointing to your scGPT checkpoint.
Steps:
1. Read preprocessed gene list
2. Extract matched 512-dim scGPT embeddings from local scGPT checkpoint
3. Unmatched genes: zero-filled (default), random init (--random-fill), or dropped (--drop-unmatched)
4. --drop-unmatched: synchronously filters processed_data.pt and gene_meta.csv so all three are 100% aligned
5. Outputs embeddings aligned with gene order
"""
import torch
import pandas as pd
import json
import argparse
from pathlib import Path

def main(gene_meta_path, scgpt_dir, output_path='data/gene_embeddings.pt', zero_fill=True, drop_unmatched=False):
    # Load gene list
    gene_meta_path = Path(gene_meta_path)
    gene_meta = pd.read_csv(gene_meta_path)
    target_genes = gene_meta['gene_name'].tolist()
    total_genes = len(target_genes)
    print(f"Target genes: {total_genes}")

    # Load scGPT data
    scgpt_dir = Path(scgpt_dir)
    gene_to_idx = json.load(open(scgpt_dir / "vocab.json"))
    model = torch.load(scgpt_dir / "best_model.pt", map_location='cpu')
    emb_layer = model['encoder.embedding.weight']
    emb_dim = emb_layer.shape[1]
    print(f"scGPT embedding dim: {emb_dim}")

    # Match embeddings
    if drop_unmatched:
        # Drop unmatched genes mode
        matched_indices = []
        matched_emb_list = []
        for i, g in enumerate(target_genes):
            if g in gene_to_idx:
                matched_indices.append(i)
                matched_emb_list.append(emb_layer[gene_to_idx[g]])

        matched_count = len(matched_emb_list)
        matched_emb = torch.stack(matched_emb_list)
        print(f"Matched: {matched_count}/{total_genes} ({matched_count/total_genes*100:.1f}%)")
        print(f"Dropped unmatched: {total_genes - matched_count} genes")

        # Synchronously filter dataset
        data_path = gene_meta_path.parent / "processed_data.pt"
        if data_path.exists():
            print(f"Filtering dataset: {data_path}")
            data = torch.load(data_path)
            filtered_train = data['train'][:, matched_indices]
            filtered_val = data['val'][:, matched_indices]
            torch.save({
                'train': filtered_train,
                'val': filtered_val
            }, data_path)
            print(f"Updated train: {filtered_train.shape}")
            print(f"Updated val: {filtered_val.shape}")
        else:
            print(f"Dataset not found: {data_path}, skipping")

        # Synchronously filter gene metadata
        filtered_gene_meta = gene_meta.iloc[matched_indices].reset_index(drop=True)
        filtered_gene_meta.to_csv(gene_meta_path, index=False)
        print(f"Updated gene_meta.csv: {len(filtered_gene_meta)} genes")

    else:
        # Fill mode
        matched_emb = torch.zeros(len(target_genes), emb_dim, dtype=torch.float32)
        if not zero_fill:
            matched_emb.normal_(0, 0.02)

        matched_count = 0
        for i, g in enumerate(target_genes):
            if g in gene_to_idx:
                matched_emb[i] = emb_layer[gene_to_idx[g]]
                matched_count += 1

        print(f"Matched: {matched_count} ({matched_count/total_genes*100:.1f}%)")
        print(f"Unmatched handling: {'zero-filled' if zero_fill else 'random init'}")

    # Save embeddings
    torch.save(matched_emb, output_path)
    print(f"Embedding shape: {matched_emb.shape} [num_genes, embedding_dim]")
    print(f"Output: {output_path}")
    if drop_unmatched:
        print("All files synchronized: data, metadata, embeddings — 100% aligned, no zero-padding")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gene_meta", default="data/gene_meta.csv", help="Gene metadata CSV path")
    parser.add_argument("--scgpt_dir", default="../scgpt-embedding", help="Local scGPT checkpoint directory")
    parser.add_argument("--output", default="data/gene_embeddings.pt", help="Output embedding path")
    parser.add_argument("--random_fill", action="store_true",
                        help="Random-initialize unmatched genes (default: zero-fill; mutually exclusive with --drop-unmatched)")
    parser.add_argument("--drop-unmatched", action="store_true",
                        help="Drop unmatched genes and synchronously filter all data files — all remaining genes have real embeddings")
    args = parser.parse_args()

    if args.drop_unmatched and args.random_fill:
        raise ValueError("--random-fill and --drop-unmatched are mutually exclusive")

    main(args.gene_meta, args.scgpt_dir, args.output,
         zero_fill=not args.random_fill,
         drop_unmatched=args.drop_unmatched)
