"""
Core data preprocessing: converts raw scRNA-seq h5ad files into model-ready tensors.
Reusable: change --input to point to any dataset.
Steps:
1. Read h5ad, keep all input genes (no forced HVG filtering)
2. Batch query genomic coordinates (chromosome + start position) via mygene
3. Sort genes by chromosome + position ascending, reorder expression matrix
4. Apply log1p(CPM/10000) normalization
5. 8:2 train/val split, save as torch tensors
"""
import scanpy as sc
import pandas as pd
import mygene
import torch
import argparse

def main(input_h5ad, output_dir='data'):
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Read data
    adata = sc.read_h5ad(input_h5ad)
    gene_list = list(adata.var_names)
    print(f"Input: {adata.n_obs} cells x {len(gene_list)} genes")

    # Query gene coordinates
    mg = mygene.MyGeneInfo()
    res = mg.querymany(gene_list, scopes='symbol', species='human',
                       fields='genomic_pos.chr,genomic_pos.start',
                       returnall=False)

    # Parse results
    gene_meta = []
    for entry in res:
        gene = entry['query']
        if 'genomic_pos' in entry:
            pos = entry['genomic_pos']
            pos = pos[0] if isinstance(pos, list) else pos
            if pos.get('chr') and pos.get('start'):
                gene_meta.append({
                    'gene_name': gene,
                    'chr': str(pos['chr']),
                    'start': int(pos['start'])
                })

    gene_meta_df = pd.DataFrame(gene_meta)
    print(f"Coordinates matched: {len(gene_meta_df)} genes")

    # Sort genes by chromosome and position
    chr_order = {str(i): i for i in range(1, 23)}
    chr_order.update({'X': 23, 'Y': 24})
    gene_meta_df['chr_order'] = gene_meta_df['chr'].map(lambda x: chr_order.get(x, 999))
    gene_meta_df = gene_meta_df.sort_values(by=['chr_order', 'start']).drop('chr_order', axis=1).reset_index(drop=True)

    # Reorder matrix + normalize
    matched_genes = gene_meta_df['gene_name'].tolist()
    adata_filtered = adata[:, matched_genes].copy()
    sc.pp.normalize_total(adata_filtered, target_sum=10000)
    sc.pp.log1p(adata_filtered)

    # Convert to tensor and split
    expr_tensor = torch.tensor(adata_filtered.X.toarray(), dtype=torch.float32)
    split = int(expr_tensor.shape[0] * 0.8)

    # Save
    torch.save({
        'train': expr_tensor[:split],
        'val': expr_tensor[split:],
        'gene_names': matched_genes
    }, f"{output_dir}/processed_data.pt")
    gene_meta_df.to_csv(f"{output_dir}/gene_meta.csv", index=False)

    print(f"Done! Output: {output_dir}/")
    print(f"Final: {expr_tensor.shape[0]} cells x {expr_tensor.shape[1]} genes")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input h5ad file path")
    parser.add_argument("--output_dir", default="data", help="Output directory")
    args = parser.parse_args()
    main(args.input, args.output_dir)
