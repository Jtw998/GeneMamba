"""
核心数据预处理脚本：处理原始scRNA-seq h5ad文件为模型可用数据集
可复用：换数据集时直接修改输入路径即可
功能：
1. 读取h5ad文件，保留所有输入基因（无强制高变筛选）
2. 批量查询基因基因组坐标（染色体号+起始位置）
3. 按「染色体+坐标」升序排序基因，重排表达矩阵
4. 执行log1p(CPM/10000)标准化
5. 8:2拆分训练/验证集，输出torch张量格式
"""
import scanpy as sc
import pandas as pd
import mygene
import torch
import argparse

def main(input_h5ad, output_dir='data'):
    import os
    os.makedirs(output_dir, exist_ok=True)

    # 读取数据
    adata = sc.read_h5ad(input_h5ad)
    gene_list = list(adata.var_names)
    print(f"输入数据: {adata.n_obs}细胞 × {len(gene_list)}基因")

    # 查询基因坐标
    mg = mygene.MyGeneInfo()
    res = mg.querymany(gene_list, scopes='symbol', species='human',
                       fields='genomic_pos.chr,genomic_pos.start',
                       returnall=False)

    # 解析结果
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
    print(f"坐标匹配成功: {len(gene_meta_df)}基因")

    # 基因排序
    chr_order = {str(i):i for i in range(1,23)}
    chr_order.update({'X':23, 'Y':24})
    gene_meta_df['chr_order'] = gene_meta_df['chr'].map(lambda x: chr_order.get(x, 999))
    gene_meta_df = gene_meta_df.sort_values(by=['chr_order', 'start']).drop('chr_order', axis=1).reset_index(drop=True)

    # 重排矩阵+标准化
    matched_genes = gene_meta_df['gene_name'].tolist()
    adata_filtered = adata[:, matched_genes].copy()
    sc.pp.normalize_total(adata_filtered, target_sum=10000)
    sc.pp.log1p(adata_filtered)

    # 转张量拆分
    expr_tensor = torch.tensor(adata_filtered.X.toarray(), dtype=torch.float32)
    split = int(expr_tensor.shape[0] * 0.8)

    # 保存
    torch.save({
        'train': expr_tensor[:split],
        'val': expr_tensor[split:],
        'gene_names': matched_genes
    }, f"{output_dir}/processed_data.pt")
    gene_meta_df.to_csv(f"{output_dir}/gene_meta.csv", index=False)

    print(f"✅ 处理完成！输出保存到 {output_dir} 目录")
    print(f"最终数据集: {expr_tensor.shape[0]}细胞 × {expr_tensor.shape[1]}基因")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="输入h5ad文件路径")
    parser.add_argument("--output_dir", default="data", help="输出目录")
    args = parser.parse_args()
    main(args.input, args.output_dir)
