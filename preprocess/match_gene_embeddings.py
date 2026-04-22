"""
核心基因嵌入匹配脚本：从本地scGPT预训练模型提取匹配的基因嵌入
可复用：有新的基因列表时直接运行即可
功能：
1. 读取预处理后的基因列表
2. 从本地scGPT预训练模型中匹配提取对应基因的512维嵌入
3. 无对应嵌入的基因默认零填充，可配置为随机初始化/自动剔除
4. 开启--drop-unmatched时自动同步过滤数据集和基因元数据，保证三者100%匹配
5. 输出与基因顺序一一对应的嵌入张量
"""
import torch
import pandas as pd
import json
import argparse
from pathlib import Path

def main(gene_meta_path, scgpt_dir, output_path='data/gene_embeddings.pt', zero_fill=True, drop_unmatched=False):
    # 读取基因列表
    gene_meta_path = Path(gene_meta_path)
    gene_meta = pd.read_csv(gene_meta_path)
    target_genes = gene_meta['gene_name'].tolist()
    total_genes = len(target_genes)
    print(f"目标基因数: {total_genes}")

    # 读取scGPT数据
    scgpt_dir = Path(scgpt_dir)
    gene_to_idx = json.load(open(scgpt_dir / "vocab.json"))
    model = torch.load(scgpt_dir / "best_model.pt", map_location='cpu')
    emb_layer = model['encoder.embedding.weight']
    emb_dim = emb_layer.shape[1]
    print(f"scGPT嵌入维度: {emb_dim}")

    # 匹配嵌入
    if drop_unmatched:
        # 剔除未匹配基因模式
        matched_indices = []
        matched_emb_list = []
        for i, g in enumerate(target_genes):
            if g in gene_to_idx:
                matched_indices.append(i)
                matched_emb_list.append(emb_layer[gene_to_idx[g]])

        matched_count = len(matched_emb_list)
        matched_emb = torch.stack(matched_emb_list)
        print(f"嵌入匹配完成！成功匹配: {matched_count}/{total_genes} ({matched_count/total_genes*100:.1f}%)")
        print(f"自动剔除未匹配基因: {total_genes - matched_count} 个")

        # 同步过滤数据集
        data_path = gene_meta_path.parent / "processed_data.pt"
        if data_path.exists():
            print(f"\n同步过滤数据集: {data_path}")
            data = torch.load(data_path)
            filtered_train = data['train'][:, matched_indices]
            filtered_val = data['val'][:, matched_indices]
            torch.save({
                'train': filtered_train,
                'val': filtered_val
            }, data_path)
            print(f"更新后训练集形状: {filtered_train.shape}")
            print(f"更新后验证集形状: {filtered_val.shape}")
        else:
            print(f"\n未找到数据集文件: {data_path}，跳过数据集过滤")

        # 同步过滤基因元数据
        filtered_gene_meta = gene_meta.iloc[matched_indices].reset_index(drop=True)
        filtered_gene_meta.to_csv(gene_meta_path, index=False)
        print(f"更新后基因元数据已保存: {gene_meta_path}，剩余 {len(filtered_gene_meta)} 个基因")

    else:
        # 原填充模式
        matched_emb = torch.zeros(len(target_genes), emb_dim, dtype=torch.float32)
        if not zero_fill:
            matched_emb.normal_(0, 0.02) # 随机初始化

        matched_count = 0
        for i, g in enumerate(target_genes):
            if g in gene_to_idx:
                matched_emb[i] = emb_layer[gene_to_idx[g]]
                matched_count +=1

        print(f"嵌入匹配完成！成功匹配: {matched_count} ({matched_count/total_genes*100:.1f}%)")
        print(f"未匹配基因处理方式: {'零填充' if zero_fill else '随机初始化'}")

    # 保存嵌入
    torch.save(matched_emb, output_path)
    print(f"\n最终嵌入形状: {matched_emb.shape} [num_genes, embedding_dim]")
    print(f"输出保存到: {output_path}")
    if drop_unmatched:
        print("所有文件已同步更新：数据集、基因元数据、嵌入三者基因顺序100%匹配，无零填充基因")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gene_meta", default="data/gene_meta.csv", help="基因元数据csv路径")
    parser.add_argument("--scgpt_dir", default="../scgpt-embedding", help="本地scGPT预训练模型目录")
    parser.add_argument("--output", default="data/gene_embeddings.pt", help="输出嵌入路径")
    parser.add_argument("--random_fill", action="store_true", help="未匹配基因用随机初始化，默认零填充（与--drop-unmatched互斥）")
    parser.add_argument("--drop-unmatched", action="store_true", help="自动剔除未匹配基因，并同步过滤数据集和基因元数据，保留的基因100%有真实嵌入")
    args = parser.parse_args()

    if args.drop_unmatched and args.random_fill:
        raise ValueError("--random-fill和--drop-unmatched不能同时使用，剔除模式下不会填充任何基因")

    main(args.gene_meta, args.scgpt_dir, args.output, zero_fill=not args.random_fill, drop_unmatched=args.drop_unmatched)
