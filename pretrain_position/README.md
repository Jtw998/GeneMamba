# Fourier Position Encoder Pretraining

Pretrain a genomic position encoder on gene coordinate data, then export as a lookup table for injection into GeneMamba.

**Why:** Gene coordinates encode cis-regulatory proximity. Adjacent genes share TADs, co-regulate, and belong to the same chromatin domain. Pre-training a Fourier encoder to capture this structure improves GeneMamba's ability to reason about genomic context — especially useful when scGPT embeddings are unavailable (e.g., new species).

**How it works:**

```
Gene coordinates (chrom, tss, log_tss)
        ↓
Fourier features (sin/cos at geometric-spaced frequencies)
        ↓
Learnable projection → [num_genes, 256] position embeddings
        ↓
Contrastive loss: nearby genes → similar, distant genes → different
```

---

## Execution

```bash
# 1. Prepare: read gene_meta.csv from all datasets, generate positive/negative gene pairs
python pretrain_position/prepare_data.py

# 2. Train: contrastive learning, 50 epochs
python pretrain_position/train.py

# 3. Export: save as position_table.pt (gene_name → [256] tensor)
python pretrain_position/export_table.py
# Output: ../position_table.pt
```

---

## What Gets Generated

| File | Contents |
|------|----------|
| `gene_coords.npy` | `[N, 3]` — chrom_idx, tss_norm, log_tss_norm per gene |
| `gene_to_idx.json` | gene_name → integer index |
| `pairs.json` | 147k gene pairs with labels and genomic distances |
| `checkpoints/best_model.pt` | Trained encoder weights |
| `../position_table.pt` | Final lookup table for GeneMamba |

---

## Configuration

Edit `train.py` → `main()` → `config` dict:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `embed_dim` | 256 | Output embedding dimension |
| `num_frequencies` | 16 | Fourier frequency count |
| `num_epochs` | 50 | Training epochs |
| `batch_size` | 512 | Pairs per batch |
| `lr` | 1e-3 | Learning rate |

---

## Quality Metrics

Monitor these during training:

| Metric | What it means |
|--------|---------------|
| **PosSim** | Mean cosine similarity between positive pairs (nearby genes). Should increase. |
| **NegSim** | Mean cosine similarity between negative pairs (distant genes). Should decrease. |
| **Gap = PosSim − NegSim** | Separation power. Should grow steadily. |

Best model is saved when Gap is maximized.

---

## Loading into GeneMamba

```python
from models import GeneMamba

model = GeneMamba(
    num_genes=...,
    gene_emb_dim=...,
    gene_emb=...,
    gene_names=gene_names_list,
    position_table_path="position_table.pt",  # ← here
)
```

GeneMamba's `PositionEncoder` has two modes:
- **Mode A** (recommended): loads `position_table.pt`, does O(1) lookup per gene
- **Mode B** (fallback): computes Fourier features on-the-fly from gene coordinates — no pretraining needed

The position gate starts suppressed (σ(−2) ≈ 0.12) and gradually opens during training to avoid disrupting learned features.

---

## Data Sources

Coordinates are read from all `gene_meta.csv` files found in the project:
- `data/gene_meta.csv` — original training dataset
- `Schmidt/schmidt_gene_meta.csv` — Schmidt perturbation dataset

Non-standard chromosome names (HSCHR*, HG*_PATCH) are filtered; only chr 1–22, X, Y, MT are used.
