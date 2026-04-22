#!/usr/bin/env python3
"""
Standalone script to generate chromosome boundary metadata, no modification to original preprocessing pipeline
Reads existing data/gene_meta.csv to generate chromosome block boundaries, saves as data/chrom_boundaries.pt
"""
import pandas as pd
import torch
def main():
    # Read existing gene metadata
    gene_meta = pd.read_csv("data/gene_meta.csv")
    chrom_boundaries = []
    current_chr = gene_meta['chr'].iloc[0]
    start_idx = 0
    for i, chr in enumerate(gene_meta['chr']):
        if chr != current_chr:
            chrom_boundaries.append((start_idx, i))
            start_idx = i
            current_chr = chr
    chrom_boundaries.append((start_idx, len(gene_meta)))
    # Save boundaries
    torch.save(chrom_boundaries, "data/chrom_boundaries.pt")
    print("Chromosome boundary generation completed, total {} chromosome blocks:".format(len(chrom_boundaries)))
    for idx, (start, end) in enumerate(chrom_boundaries):
        chr_name = gene_meta['chr'].iloc[start]
        print("  Chr{}: [{}, {}) {} genes total".format(chr_name, start, end, end-start))
if __name__ == "__main__":
    main()
