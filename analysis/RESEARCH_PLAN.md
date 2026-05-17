# Hayat 完整研究方案

## Hayat：基于 cis/trans 分治的基因表达建模及公共配对多组学交叉验证

---

## 一、研究背景与科学问题

基因表达调控本质上并不是单一层面的数值拟合问题，而是由两类不同尺度的生物机制共同决定：

1. **顺式调控（cis）** — 基因所在染色体邻域内的局部染色质上下文，包括增强子-启动子作用、局部拓扑结构、同染色体近邻依赖等；
2. **反式调控（trans）** — 少量全局调控因子，尤其是转录因子和染色质调控蛋白，通过跨染色体、跨通路传播调控信号，影响大量下游基因表达。

现有很多表达建模方法通常把所有基因统一当作等价输入 token，容易忽略两点：
- 基因组在线性染色体上的**局部结构约束**
- 少数关键 regulator 对大规模表达变化的**稀疏支配作用**

因此，本研究提出 **Hayat**，通过显式分解表达生成过程，将基因表达建模为：

> **基因表达 = 顺式染色质上下文 × 反式调控因子调制**

并通过公共表达数据、公共扰动数据、公共 TF 知识库以及公共 10x Multiome PBMC 配对 RNA+ATAC 数据，对该结构假设进行系统验证。

---

## 二、核心假设

### 总体生物学假设

> 基因表达由 cis 与 trans 两类调控机制共同决定；其中 cis 提供局部表达背景，trans 对 cis 状态进行条件性调制。

### 模型假设

- **H1**：按染色体分块建模的 cis 分支能更好学习同染色体局部依赖；
- **H2**：稀疏 regulator gate 的 trans 分支能自动识别少量关键调控因子；
- **H3**：乘性融合 `cis_out * (1 + tanh(trans_out))` 比简单相加更符合"trans 调制 cis"的调控逻辑；
- **H4**：Hayat 的 cis/trans 表征可被独立公共多组学数据支持，即 cis 对应开放染色质，trans 对应调控因子驱动与跨染色体传播。

---

## 三、研究目标

### 目标 1：验证 Hayat 架构的必要性
- 非线性建模是否必要？
- cis/trans 分治是否必要？
- 染色体分块是否必要？

### 目标 2：验证 RegulatorGate 的生物学真实性
- 高 gate 基因是否富集已知 TF？
- 是否富集转录调控相关功能？
- 是否与独立扰动数据中的强效 perturbation gene 一致？

### 目标 3：验证模型内部机制是否得到公共多组学支持
- cis 表征是否与 promoter accessibility 一致？
- trans 表征是否体现跨染色体调控流？
- regulator-target 关系是否得到公共 TF target 先验支持？

---

## 四、模型总体设计

### 4.1 Cis 分支
- 输入为按染色体和坐标排序的基因表达序列；
- 根据 `chrom_boundaries` 对基因序列按染色体块切分；
- 在每个块内使用**双向 Mamba2** 建模局部依赖；
- 输出 `cis_out`，刻画局部顺式表达上下文。

### 4.2 Trans 分支
- 对全基因表达先经过 **RegulatorGate MLP**；
- 选择少量活跃 regulator；
- 用轻量交互模块建模 regulator 对所有基因的全局调制；
- 输出 `trans_out`，刻画反式调控强度。

### 4.3 融合方式

```
total_out = cis_out * (1 + tanh(trans_out))
```

- `cis_out` 提供基础表达背景；
- `trans_out` 负责对 `cis_out` 进行增益/抑制；
- 相较于加法融合，更符合调控因子依赖顺式可及背景发挥作用的生物逻辑。

---

## 五、数据资源

### 5.1 已就绪数据

| 文件 | 大小 | 用途 |
|------|------|------|
| `data/processed_data.pt` | 55 GB | PBS 表达矩阵 train/val |
| `data/gene_embeddings.pt` | 45 MB | scGPT 基因嵌入 |
| `data/gene_meta.csv` | 413 KB | 基因染色体坐标 |
| `data/chrom_boundaries.pt` | 4.5 KB | 染色体块边界 |
| `position_table.pt` | 26 MB | Fourier 位置编码 |
| `Schmidt/schmidt_data.pt` | 1.1 GB | 扰动表达矩阵 |
| `Schmidt/schmidt_perturb_labels.pt` | — | 扰动标签 |
| `Schmidt/schmidt_gene_meta.csv` | — | Schmidt 基因坐标 |
| `Schmidt/schmidt_chrom_boundaries.pt` | — | Schmidt 染色体边界 |
| `Schmidt/schmidt_gene_embeddings.pt` | — | Schmidt 基因嵌入 |

### 5.2 外部公共数据

| 数据 | 来源 | 用途 |
|------|------|------|
| Lambert 2018 human TF list | humantfs.ccbr.utoronto.ca | TF 富集验证 |
| **10x Multiome PBMC 公共 RNA+ATAC 数据** | 10x Genomics / GEO | 配对多组学验证 |
| TRRUST / DoRothEA / ChEA（任选 1–2 个） | 公共数据库 | regulator-target 外部验证 |

### 5.3 为什么用 10x Multiome PBMC 替代 ENCODE PBMC ATAC-seq

1. 同一样本体系下的配对多组学验证更强
2. 可进行细胞类型分层，而不是 bulk 平均
3. 更适合验证 cis/trans 机制与细胞类型特异调控
4. 在不使用 fragments 的前提下，也可通过 peak-by-cell matrix 低成本完成 promoter accessibility 分析

---

## 六、数据预处理与统一框架

### 6.1 基因集统一

所有分析基于统一交集基因集：PBS 表达基因 ∩ Schmidt 数据基因 ∩ 基因 embedding 基因 ∩ 有有效染色体坐标的基因 ∩ Multiome RNA 与 ATAC 可映射的基因。输出统一基因表 `gene_table.csv`。

### 6.2 染色体排序与分块

按染色体编号和 genomic coordinate 排序，构建最终 `chrom_boundaries.pt`，所有 cis 相关分析使用同一排序。

### 6.3 10x Multiome PBMC 预处理

- **输入**：RNA expression matrix, ATAC peak-by-cell matrix, peak coordinates, barcodes
- **Promoter accessibility 定义**：TSS ± 2 kb，将与 promoter 重叠的 peaks 在每个细胞中求和 → `log1p` → library-size normalization
- **细胞层面简化**：保留 5k–10k 高质量细胞，使用 RNA 标记基因做粗细胞类型标注（T cells / B cells / NK cells / Monocytes / others），主分析采用 pseudo-bulk / cell-type average

---

## 七、计算资源与降本策略

### 7.1 分阶段训练

| 阶段 | cells | epochs | 用途 | 设备 |
|------|------|------|------|------|
| Pilot | 2k–5k | 20–30 | 跑通流程、验证趋势 | MPS / 4090 |
| Main | 10k–20k | 30–50 | 生成主结果 | 4090 / A100 |
| Enhanced | 50k+ | 50–100 | 补强结果 | A100 |

主结果以 **10k–20k cells** 为核心，不强依赖 100k cells 全量训练。大多数 study 复用同一个 Hayat 主 checkpoint。

### 7.2 降本原则

1. 只训练两个模型：Hayat 主模型 + LinearAE baseline
2. Ablation 尽量推理级实现（full / cis_only / no_blocking）
3. Attention 不保存全量 tensor，只做在线聚合，输出 `chr_flow_matrix`、`cross_chr_ratio`、`regulator_attention_mass`
4. Multiome 主分析不从 fragments 重算，只使用 processed peak matrix
5. ATAC 主分析用 pseudo-bulk，减少稀疏性

### 7.3 推荐配置

- 设备：RTX 4090 或 A100，显存 ≥ 12 GB
- batch_size = 16，mixed precision
- Pilot 0.5–1 天，Main 1–2 天，分析 1–2 天，主结果约一周

---

## 八、训练方案

### 8.1 Hayat 主模型

- 训练细胞：10k–20k
- epochs：30–50
- batch size：16
- optimizer：AdamW，lr=1e-4
- mixed precision：开启
- early stopping：patience=10
- seeds：3 个独立随机种子

输出：`checkpoints/hayat_main_seed{1,2,3}.pt`

### 8.2 LinearAE baseline

结构：N_genes → 64 → N_genes。使用与 Hayat 相同的训练数据和优化配置，作为最简非结构化线性 baseline。

---

## 九、研究内容与实验设计

---

## Study 1：主性能与结构必要性验证

### 研究目的

验证非线性建模是否必要、cis/trans 分治是否必要、染色体分块是否必要。

### 比较模型

**主文模型**：Hayat Full / Cis-only / No-blocking / LinearAE
**补充模型**：Trans-only（偏调制而非独立生成器，放入补充）

### Ablation 定义

```
ablation="full":        total_out = cis_out * (1 + tanh(trans_out))
ablation="cis_only":    total_out = cis_out
ablation="no_blocking": 跳过染色体分块，整序列进入 Mamba
```

- `cis_only` 可不重训
- `no_blocking` 在主文中先作为 inference-time stress test

### 数据

- 训练/验证：PBS
- 外部泛化：Schmidt perturbation

### 指标

**重建指标**：Global Pearson、Global Spearman、MSE、Intra-chromosome Pearson、Inter-chromosome Pearson
**扰动指标**：PCC-delta、Common-DEGs、Perturbation-wise MSE
**距离分层指标**：same block / same chromosome but far / different chromosome

### 预期结果

- Full 在总体表现、跨染色体建模和扰动泛化上最优
- Cis-only 在同染色体指标上接近 Full，但跨染色体和扰动预测明显下降
- No-blocking 次优，说明分块提供有效结构归纳偏置
- LinearAE 可拟合平均表达，但缺乏调控传播能力

### 论文图

- 柱状图：各模型 × 指标
- 距离分层折线图
- perturbation-wise scatter：Full vs baseline

---

## Study 2：RegulatorGate 生物学真实性验证

### 研究目的

验证模型选出的 regulator 是否具有真实的转录调控意义。

### 2A. TF 富集分析

```python
regulator_idx = model.get_regulator_genes(gene_names, threshold=0.5)
```

同时做 threshold 版本和 top-K 版本（如 top 300）。与 Lambert 2018 TF 列表比较，计算 overlap、Fisher exact / hypergeometric p 值、odds ratio / enrichment fold。

### 2B. GO / Reactome 富集

比较 top 200 regulator vs bottom 200。预期 top regulators 富集 transcription regulation、chromatin remodeling、signal response；low-gate 基因更偏 housekeeping / metabolism。

### 2C. 与扰动数据的一致性

在 Schmidt 中定义每个被 perturb 基因的 effect size（DEGs 数量、平均 |delta expression|、perturbation-induced variance），分析 gate_score 与 perturb_effect_size 的相关性。若高 gate 基因在独立 perturbation 中也更能引发广泛表达改变，则说明 gate 学到的确是有效调控因子。

### 2D. 训练动态

记录每个 epoch 的 active regulator 数量、平均 gate score、gate 稀疏度，展示从全基因到少量 regulator 的收敛过程。

### 预期结果

- 高 gate 基因显著富集 TF
- 高 gate 基因富集调控相关功能
- gate score 与扰动 effect size 显著正相关
- active regulator 数量收敛至约 200–500

### 论文图

- TF enrichment plot
- gate score vs perturb effect size scatter
- GO/Reactome enrichment heatmap
- active regulator 收敛曲线

---

## Study 3：Trans 跨染色体机制验证

### 研究目的

验证 trans 分支是否真正承担了跨染色体调控建模，而不是简单噪声映射。

### 3A. 染色体间注意力流矩阵

在线累计 `Flow(chr_target, chr_regulator)`，得到 chr-to-chr 调控流矩阵。

### 3B. 跨染色体注意力占比

对每个目标基因计算 `cross_chr_ratio(g)` = 该基因投给不同染色体 regulator 的注意力占比。按 chromosome、gene density、mean expression 分层。

### 3C. 已知 TF target 外部验证

对 top gate TF 或 top trans regulator：提取模型高 attention target，与 TRRUST / DoRothEA / ChEA 已知 target 比较，进行富集检验。

### 计算优化

仅分析 active regulators 或 top-K regulators（如 64/128）。不保存全量 attention，只输出 `chr_flow_matrix.pt`、`cross_chr_ratio.csv`、`regulator_attention_mass.csv`。

### 预期结果

- 存在显著跨染色体调控流
- 高 gate regulator 具有更强跨染色体影响
- 部分经典 TF 的预测 target 与已知 regulon 显著重叠

### 论文图

- 22×22 chr flow heatmap
- cross-chr ratio 分布图
- selected TF 的 predicted target enrichment 图

---

## Study 4：基于 10x Multiome PBMC 的配对多组学外部验证

### 研究目的

利用公共 10x Multiome PBMC 配对 RNA+ATAC 数据，验证 Hayat 学到的 cis/trans 表征是否与真实染色质状态和细胞类型特异调控一致。这是整个项目的多组学机制验证核心。

### 核心假设

- **H4.1**：基因的 `cis_score` 与其 promoter accessibility 显著正相关
- **H4.2**：这一关系在不同 PBMC 细胞类型中均成立
- **H4.3**：开放 promoter 基因更偏 cis 主导，关闭 promoter 基因更依赖 trans 调制
- **H4.4**：高 trans regulator 的预测 targets 在相应细胞类型中具有更强的可及性或已知 regulon 支持

### 模型内部表征定义

将 Multiome RNA 输入 Hayat，提取：

```
cis_score(g, c)  = ||cis_hidden[g, c]||_2
trans_score(g, c) = ||trans_hidden[g, c]||_2
```

主分析使用平均形式 `cis_score(g) = E_c[||cis_hidden[g, c]||_2]`，以及按细胞类型版本 `cis_score(g, t) = E_{c∈t}[||cis_hidden[g, c]||_2]`。

### 分析内容

#### 4A. 全局基因层面：cis score 与 promoter accessibility

对每个基因比较 `cis_score(g)` 与 `ATAC(g)`，做 Spearman correlation。预期显著正相关。

#### 4B. 细胞类型分层验证

在各主要 PBMC 细胞类型内分别计算 `corr_t = Spearman(cis_score(·,t), ATAC(·,t))`，排除 bulk 平均伪相关。

#### 4C. 控制表达量后的偏相关

`partial_corr(cis_score, ATAC | RNA)`，若控制表达后仍显著，说明 cis_score 学到的不只是表达强度，而是更接近染色质状态的信息。

#### 4D. 按 promoter openness 分层评估模型行为

根据 Multiome 的 promoter accessibility 将基因分为 Open / Mid / Closed，比较重建 Pearson、重建 MSE、Schmidt PCC-delta、Common-DEGs、perturbation MSE。预期 Open 基因表现更优。

#### 4E. cis/trans dominance 分析

```
ratio(g) = cis_score(g) / (cis_score(g) + trans_score(g) + ε)
```

比较 Open / Mid / Closed 三组基因的 ratio 分布。预期 Open 组更偏 cis 主导，Closed 组相对更偏 trans 调制。

#### 4F. regulator-target 跨模态验证

对 top gate regulator 或 top TF：提取其高 attention / 高影响 target genes，检查这些 target 在相应细胞类型中是否更开放，与 TRRUST / DoRothEA / ChEA target set 比较。形成 "cis × trans × 外部知识" 交互验证。

### 预期结果

- cis_score 与 promoter accessibility 在全局及细胞类型层面均显著正相关
- 控制表达量后相关性仍保留
- Open 基因更偏 cis 主导
- top regulator 的 targets 兼具更高可及性和更强 regulon 支持

### 论文图

- cis_score vs promoter accessibility 散点图
- 细胞类型相关系数柱状图
- Open/Mid/Closed 分层箱线图
- selected regulator target accessibility 图

---

## Supplementary Study：训练规模敏感性分析

### 目的

验证主要结论不依赖超大规模训练。

### 方案

训练 3 个规模（2k / 10k / 50k cells），比较 Global Pearson、PCC-delta、TF enrichment OR、cis-ATAC Spearman。证明 10k–20k cells 已可得到稳定结果。

---

## 十、统计分析方案

| 分析类型 | 方法 |
|----------|------|
| 富集检验 | Fisher exact test / Hypergeometric test |
| 相关分析 | Spearman correlation / Partial correlation（控制表达量） |
| 分组比较 | Wilcoxon rank-sum / Kruskal-Wallis / t-test / ANOVA |
| 多重检验校正 | Benjamini-Hochberg FDR |
| 结果报告 | effect size, 95% CI, p 值或 FDR, mean ± SD, ≥ 3 seeds |

---

## 十一、实现结构

```
analysis/
├── study1_benchmark.py          # Full / Cis-only / No-blocking / LinearAE
├── study2_regulators.py         # TF富集 + GO + 训练动态
├── study3_trans_flow.py         # 跨染色体注意力流
├── study4_multiome.py           # 10x Multiome 配对验证
├── train_linear_ae.py           # 线性AE训练
├── summarize_metrics.py         # 统一指标汇总
└── utils_analysis.py            # 共享工具函数
```

### 建议模型接口扩展

```python
forward(
    x,
    ablation="full",               # full / cis_only
    return_hidden=False,            # 返回 cis_out, trans_out
    return_attention_summary=False  # 返回聚合 attention
)
```

### 建议输出 summary

- `gate_scores.csv`
- `cis_score_per_gene.pt`
- `trans_score_per_gene.pt`
- `chr_flow_matrix.pt`
- `cross_chr_ratio.csv`
- `multiome_celltype_scores.csv`

---

## 十二、实施步骤

### 阶段 1：最小闭环验证

快速确认主线成立：

1. 训练 5k cells 的 Hayat pilot 模型
2. 训练 LinearAE
3. 完成 Study 1 主比较
4. 完成 Study 2 的 TF 富集
5. 完成 Study 4 的 cis_score vs ATAC 验证

### 阶段 2：主结果生成

形成论文主图：

1. 训练 10k–20k cells 主模型
2. 完成 Study 1 全部
3. 完成 Study 2 全部
4. 完成 Study 3 的 chr flow 与 target enrichment
5. 完成 Study 4 全部多组学分析

### 阶段 3：增强验证

1. 规模敏感性分析
2. 选定经典 TF 做 case study
3. 可选 50k+ 训练
4. 可选 genome browser 可视化

---

## 十三、潜在风险与替代方案

| 风险 | 替代方案 |
|------|----------|
| cis 与 ATAC 相关性不够强 | cell-type-specific 分析；控制表达量偏相关；调整 promoter 窗口比较稳健性 |
| gate 对 TF 富集不显著 | 使用 top-K 替代固定 threshold；增加 TRRUST / DoRothEA regulon 支持；连续 gate score 对扰动 effect size 做相关 |
| no-blocking 比较被质疑不公平 | 主文标记为 inference-time stress test；补训 no-blocking 模型做 supplementary |
| Multiome ATAC 稀疏 | pseudo-bulk；必要时 metacell；主分析不依赖单细胞逐点 ATAC |

---

## 十四、预期主要结论

1. Hayat 在表达重建和扰动泛化上优于线性和退化模型
2. cis 分支主要负责局部/同染色体依赖，trans 分支主要承担全局/跨染色体调控
3. RegulatorGate 能自动识别具有真实调控功能的关键基因
4. 模型的 cis 表征与配对 Multiome 中的 promoter accessibility 一致
5. 模型的 trans 表征与 TF regulon、扰动传播和跨染色体调控流一致
6. Hayat 不仅提升预测性能，还学习到了与基因组调控组织相一致的表达生成机制

---

## 十五、论文叙事线

1. **问题提出** — 现有表达模型缺少对 cis/trans 调控分工的结构化建模
2. **方法提出** — Hayat 通过染色体分块双向 Mamba 建模 cis，稀疏 regulator gate 建模 trans，乘性融合表达 "trans 调制 cis"
3. **结果一：性能与必要性** — Hayat 优于 LinearAE、Cis-only、No-blocking
4. **结果二：调控因子真实性** — RegulatorGate 识别出真实 TF 与调控相关基因
5. **结果三：公共多组学交叉验证** — 10x Multiome PBMC 表明 cis 对应真实开放染色质，trans 对应 TF 驱动与跨染色体传播
6. **结论** — Hayat 是一个兼具性能与机制解释力的表达建模框架

---

## 十六、最小可执行版本

若希望先做一个低成本但完整闭环的版本，只完成以下 4 项：

1. 训练 10k cells Hayat
2. 训练 LinearAE
3. Study 1：Full / Cis-only / No-blocking / LinearAE
4. Study 2 TF 富集 + Study 4 Multiome 的 cis-ATAC 验证

这四部分已经足够形成一个完整、可报告、可投稿扩展的核心框架。
