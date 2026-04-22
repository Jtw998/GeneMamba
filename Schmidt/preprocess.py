#!/usr/bin/env python3
"""
Schmidt perturbation dataset preprocessing script.
Converts filter_hvg5000_logNor.h5ad into GeneMamba format.
Only single-gene perturbations are kept. Uses existing logNor layer directly.
Queries real chromosomal coordinates via mygene.
"""
import scanpy as sc
import pandas as pd
import torch
import mygene
import json
import argparse
import os

def main(scgpt_dir="../scgpt-embedding"):
    adata = sc.read_h5ad("filter_hvg5000_logNor.h5ad")
    print(f"Raw data: {adata.n_obs} cells x {adata.n_vars} genes")

    # Only keep single-gene perturbations (exclude combinations with '+')
    single_mask = ~adata.obs["perturbation"].str.contains("\\+", regex=True, na=False)
    adata = adata[single_mask].copy()
    print(f"After filtering combos: {adata.n_obs} cells x {adata.n_vars} genes")
    print(f"Perturbation groups: {adata.obs['perturbation'].nunique()}, control: {(adata.obs['perturbation']=='control').sum()}")

    # Query genomic coordinates
    gene_list = list(adata.var_names)
    print(f"Querying genomic coordinates for {len(gene_list)} genes...")
    mg = mygene.MyGeneInfo()
    results = mg.querymany(gene_list, scopes='symbol', species='human',
                          fields='genomic_pos.chr,genomic_pos.start',
                          returnall=False)

    gene_meta = []
    for entry in results:
        gene = entry['query']
        if 'genomic_pos' in entry:
            pos = entry['genomic_pos']
            pos = pos[0] if isinstance(pos, list) else pos
            if pos.get('chr') and pos.get('start'):
                gene_meta.append({'gene_name': gene, 'chr': str(pos['chr']), 'start': int(pos['start'])})

    gene_meta_df = pd.DataFrame(gene_meta)
    print(f"Coordinates matched: {len(gene_meta_df)}/{len(gene_list)} genes")

    # Sort by chromosome and position
    chr_order = {str(i): i for i in range(1, 23)}
    chr_order.update({'X': 23, 'Y': 24})
    gene_meta_df['chr_order'] = gene_meta_df['chr'].map(lambda x: chr_order.get(x, 999))
    gene_meta_df = gene_meta_df.sort_values(['chr_order', 'start']).reset_index(drop=True)
    gene_meta_df = gene_meta_df.drop(columns=['chr_order'])

    # Reorder expression matrix
    matched_genes = gene_meta_df['gene_name'].tolist()
    adata = adata[:, matched_genes].copy()

    # Use logNor layer directly
    expr_np = adata.layers["logNor"].toarray()
    expr_tensor = torch.tensor(expr_np, dtype=torch.float32)
    pert_labels = list(adata.obs["perturbation"].values)

    # Save
    torch.save({"expression": expr_tensor, "gene_names": matched_genes}, "schmidt_data.pt")
    torch.save({"perturbation": pert_labels}, "schmidt_perturb_labels.pt")
    gene_meta_df.to_csv("schmidt_gene_meta.csv", index=False)

    # Chromosome block boundaries
    chrom_boundaries = []
    current_chr = gene_meta_df['chr'].iloc[0]
    start_idx = 0
    for i, chr in enumerate(gene_meta_df['chr']):
        if chr != current_chr:
            chrom_boundaries.append((start_idx, i))
            start_idx = i
            current_chr = chr
    chrom_boundaries.append((start_idx, len(gene_meta_df)))
    torch.save(chrom_boundaries, "schmidt_chrom_boundaries.pt")

    # scGPT embeddings
    scgpt_dir = os.path.expanduser(scgpt_dir)
    emb_path = os.path.join(scgpt_dir, "vocab.json")
    if os.path.exists(emb_path):
        print(f"\nMatching scGPT embeddings from {scgpt_dir}...")
        with open(emb_path) as f:
            vocab = json.load(f)
        scgpt_model = torch.load(os.path.join(scgpt_dir, "best_model.pt"), map_location='cpu')
        emb_layer = scgpt_model['encoder.embedding.weight']
        emb_dim = emb_layer.shape[1]
        matched_emb = torch.zeros(len(matched_genes), emb_dim, dtype=torch.float32)
        cnt = 0
        for i, g in enumerate(matched_genes):
            if g in vocab:
                matched_emb[i] = emb_layer[vocab[g]]
                cnt += 1
        torch.save(matched_emb, "schmidt_gene_embeddings.pt")
        print(f"scGPT embeddings: {cnt}/{len(matched_genes)} matched ({cnt/len(matched_genes)*100:.1f}%)")
    else:
        print(f"\nscGPT dir not found at {scgpt_dir}, skipping embeddings")

    print(f"\nSaved:")
    print(f"  schmidt_data.pt: {expr_tensor.shape}")
    print(f"  schmidt_perturb_labels.pt: {len(pert_labels)} cells")
    print(f"  schmidt_gene_meta.csv: {len(matched_genes)} genes")
    print(f"  schmidt_chrom_boundaries.pt: {len(chrom_boundaries)} blocks")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scgpt_dir", default="../scgpt-embedding")
    args = parser.parse_args()
    main(scgpt_dir=args.scgpt_dir)
