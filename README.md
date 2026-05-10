# GeneMamba: Pure Endogenous Causal State Space Model for Single-Cell Transcriptomics

> Zero-shot counterfactual perturbation prediction from scRNA-seq alone — no multi-modal paired data, no ATAC-seq required. Full-platform compatible: NVIDIA GPU, Apple Silicon MPS, and CPU.

---

## Core Advantages

| Feature | Description |
|---------|-------------|
| **Pure Endogenous Causality** | Trained only on scRNA-seq data. Naturally blocks backdoor paths. Observation distribution is strictly equivalent to intervention distribution. |
| **Zero-shot Perturbation Prediction** | No intervention data during training. Directly performs counterfactual reasoning: gene knockout, overexpression, and pathway-level intervention — with native do-operator support. |
| **Cross-Dataset Inference** | Train on one gene set, infer on another — no hardcoded `num_genes`. Dynamically adapts to any gene list at runtime. |
| **Extreme Efficiency** | Bidirectional Mamba2 architecture. Memory ≤7GB. Training speed 7–8× faster than baseline. Supports simultaneous modeling of the full genome (>20,000 genes). |
| **Full Platform Compatibility** | Pure PyTorch native — no CUDA, no custom operators. Seamless support for NVIDIA GPU, Apple Silicon MPS, and CPU with zero configuration changes. |
| **High Interpretability** | Endogenously decouples chromatin accessibility amplitude Z and transcription switch S. Automatically identifies global regulatory factors. Regulatory network interpretability far exceeds black-box models. |

---

## Quick Start

### 1. Install Dependencies

```bash
pip install torch einops pandas scanpy mygene
```

### 2. Train

```bash
# Default dataset (data/):
python train.py

# Custom dataset:
python train.py --data_dir /path/to/your/dataset
```

The `--data_dir` flag selects the dataset directory. Each dataset should provide:
- `processed_data.pt` — expression tensor with `train`/`val` keys
- `chrom_boundaries.pt` — chromosome block boundaries
- `gene_embeddings.pt` — scGPT embeddings (optional, auto-detected)
- `gene_meta.csv` — gene metadata with chromosome and genomic position (for cross-dataset support)

### 3. Perturbation Prediction

```bash
python inference/inference.py
```

---

## Complete Preprocessing Pipeline

All commands run from the project root. No `cd` required.

The default training data is the PBS vehicle control condition from the **Parse-10M** dataset (629,701 PBMC cells × 40,352 genes, 12 human donors, 18 cell types).

### Step 1: Raw scRNA-seq Preprocessing

Converts a raw `.h5ad` expression matrix into model-ready format.

- Queries gene genomic coordinates automatically (via mygene)
- Sorts genes by chromosome and position (required for Mamba sequence modeling)
- Applies standard normalization: CPM/10000 + log1p
- Splits into 80% training / 20% validation

```bash
python preprocess/preprocess_data.py --input your_data.h5ad --output_dir data/
```

**Outputs under `data/`:**
- `processed_data.pt` — train/val expression tensors `[num_cells, num_genes]` + gene names
- `gene_meta.csv` — gene metadata with chromosome and genomic position

---

### Step 2: scGPT Gene Embedding Matching

Extracts and aligns pre-trained scGPT gene embeddings to your gene list. Supports three modes:

| Mode | Flag | Behavior |
|------|------|---------|
| Zero-fill (default) | none | Unmatched genes filled with zeros |
| Random fill | `--random-fill` | Unmatched genes initialized with small random vectors |
| Drop unmatched (recommended) | `--drop-unmatched` | Automatically removes genes without embeddings; synchronously filters all data files for 100% alignment |

```bash
# Recommended: drop unmatched genes (best performance)
python preprocess/match_gene_embeddings.py \
  --scgpt_dir /path/to/scgpt-embedding/ \
  --drop-unmatched

# Zero-fill unmatched genes
python preprocess/match_gene_embeddings.py \
  --scgpt_dir /path/to/scgpt-embedding/

# Random-fill unmatched genes
python preprocess/match_gene_embeddings.py \
  --scgpt_dir /path/to/scgpt-embedding/ \
  --random-fill
```

**Output under `data/`:**
- `gene_embeddings.pt` — gene embeddings `[num_genes, scgpt_emb_dim]`, perfectly aligned with expression matrix gene order

---

### Step 3: Chromosome Boundary Generation (run once per dataset)

Generates the chromosome block boundary indices required for the cis-regulatory branch.

```bash
python preprocess/generate_chrom_boundaries.py
```

**Output under `data/`:**
- `chrom_boundaries.pt` — list of `(start_idx, end_idx)` tuples, one per chromosome block

---

### Optional Step 4: Fourier Position Encoder Pretraining

Pretrain a genomic position encoder on gene coordinate data with contrastive learning, then export as a lookup table for GeneMamba. This provides position-aware gene representations that complement scGPT embeddings.

**Data:** Gene pairs generated from `gene_meta.csv`. Positive pairs = genes within 50kb; negative pairs = genes over 2Mb apart. ~55k pairs across ~20k genes from the PBS dataset.

**Config** (`pretrain_position/train.py`):
- Fourier encoder: 16 frequencies, embed_dim=256, f0=1.0, sigma=2.0
- Contrastive: temperature=0.1, λ_dist=0.3, λ_uniform=0.1
- Training: batch_size=512, num_epochs=50, lr=1e-3 (CosineAnnealing)

**Expected result:** Validation gap (PosSim − NegSim) ≥ 0.80. Trained model achieves ~0.85.

```bash
cd pretrain_position

# Step 1: prepare gene pairs from gene_meta.csv
python prepare_data.py

# Step 2: pretrain with contrastive learning (~8 min on MPS)
python train.py

# Step 3: export lookup table
python export_table.py
# Output: ../position_table.pt (~26 MB, 20k genes × 256d)
```

See [`pretrain_position/README.md`](pretrain_position/README.md) for details.

---

## Cross-Dataset Inference Architecture

GeneMamba removes the hardcoded `num_genes` dependency from the original design. Two components were modified:

### ① Dynamic Gene Embedding Lookup

Instead of `nn.Embedding(num_genes, 512)` bound to a fixed gene count, embeddings are now looked up dynamically from gene names at runtime. The model accepts a `gene_names_list` parameter — different datasets can pass different gene lists, and the embedding matrix is indexed by name rather than by position.

### ② Dynamic Regulator Gate (MLP)

The original `nn.Parameter(torch.randn(num_genes))` for the regulator gate is replaced by a lightweight MLP that computes gate values from gene embeddings:

```
gate = sigmoid(Linear(gene_embedding)) ∈ (0, 1)
```

This is shape-invariant to gene count — the MLP always produces `[num_genes]` values regardless of how many genes are present.

### Loading Old Checkpoints

```python
from models import load_with_migration
model = load_with_migration(model, "path/to/old_checkpoint.pt")
# Old regulator_gate is skipped (fresh MLP init); all other weights migrate automatically.
```

---

## Perturbation Evaluation Dataset (Schmidt)

See [`Schmidt/`](Schmidt/) for the perturbation evaluation pipeline using the Schmidt perturb-seq dataset. Train on any dataset, evaluate perturbation prediction accuracy with 6 standardized metrics.

---

## Project Structure

```
GeneMamba/
├── train.py                 # Training entry (--data_dir flag)
├── inference/inference.py  # Inference entry
│
├── models/                       # Model definitions
│   └── model.py                  # GeneMamba (cis-trans dual branch)
│                                  # + RegulatorGate MLP (cross-dataset gate)
│                                  # + PositionEncoder (pretrained table or Fourier fallback)
│
├── train/                         # Training logic
│   └── run_training.py      # End-to-end training script (gene_names aware)
│
├── preprocess/                    # Data preprocessing
│   ├── preprocess_data.py         # h5ad → model-ready tensors
│   ├── match_gene_embeddings.py    # scGPT embedding matching
│   └── generate_chrom_boundaries.py # Chromosome block boundaries
│
├── pretrain_position/             # Fourier position encoder pretraining
│   ├── prepare_data.py            # gene_meta.csv → gene pairs
│   ├── fourier_encoder.py         # Fourier position encoder
│   ├── train.py                   # Contrastive learning training
│   └── export_table.py            # Export position_table.pt
│
├── utils/                         # Shared utilities
│   ├── utils.py                   # Config, metrics, checkpoint I/O
│   └── losses.py                  # Loss functions (NB, sparsity, etc.)
│
├── docs/                          # Technical documentation
│   └── V0.1.md                    # Architecture, memory optimization, interface reference
│
├── tests/                         # Debugging / profiling scripts
├── checkpoints/                    # Auto-saved model weights
├── position_table.pt              # Pretrained Fourier position lookup table
├── data/                          # Preprocessed data (auto-created)
│   ├── processed_data.pt          # Train/val expression tensors
│   ├── gene_meta.csv             # Gene metadata
│   ├── gene_embeddings.pt        # scGPT embeddings
│   └── chrom_boundaries.pt       # Chromosome block indices
└── Schmidt/                      # Perturbation evaluation dataset
```

---

## Architecture

Cis-trans dual branch: chromosome-block shared bidirectional Mamba2 + zero-prior global regulatory gating + Fourier position encoder.

- Training memory: **≤7 GB**
- Training speed: **7–8× faster** than baseline
- TF target gene recall: **≥80%**
- Cross-chromosome false positives: **>50% reduction**
- Cross-dataset parameters: `RegulatorGate` MLP ≈ **33k** parameters (<0.2% of total)
- Position encoder: **256d Fourier** (16 frequencies), pretrained with contrastive learning (Gap ≥ 0.80)

---

## Hyperparameter Configuration

All hyperparameters are centralized in `utils/utils.py` — edit there, no need to touch model code.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_dim` | 256 | Mamba / linear layer hidden dimension |
| `num_mamba_layers` | 2 | Number of bidirectional Mamba layers |
| `latent_dim` | 64 | VAE latent space dimension |
| `learning_rate` | 1e-4 | Optimizer learning rate |
| `batch_size` | 16 | Training batch size |
| `max_regulators` | 512 | Active regulators in trans branch (model init) |

---

## Documentation

- [V1 Technical Documentation](docs/V1.md) — architecture, causal proof, loss functions, training config
- [Technical Documentation](docs/V0.1.md) — cis-trans architecture, memory optimization, interface reference
- [Fourier Position Encoder Pretraining](pretrain_position/README.md) — contrastive learning for genomic position embeddings

---

## Frequently Asked Questions

### V2 training runs out of memory (OOM)

V2 is designed for ≤7 GB with default settings. If you hit OOM:

1. Reduce `batch_size` in `utils/utils.py` (default 16 → try 8 or 4)
2. Reduce `max_regulators` at model initialization (default 512 → try 256 or 128)
3. On Apple Silicon, enable MPS fallback: `PYTORCH_ENABLE_MPS_FALLBACK=1 python train.py`

### Gene count mismatch between data and embeddings

Run with `--drop-unmatched` — this synchronously filters all data files so only genes with valid scGPT embeddings remain:

```bash
python preprocess/match_gene_embeddings.py --scgpt_dir /path/to/scgpt --drop-unmatched
```

### Can I run without a GPU?

Yes. GeneMamba falls back automatically: MPS (Apple Silicon) → CUDA → CPU. No code changes needed.

### Can I train on one dataset and infer on another with different genes?

Yes. Pass `gene_names_list` to the model forward call. The `RegulatorGate` MLP and dynamic embedding lookup are shape-invariant to gene count. See "Cross-Dataset Inference Architecture" above.

### What if chromosome boundaries fail to load?

V2 falls back to whole-genome single-block mode (slower, higher memory). Run `python preprocess/generate_chrom_boundaries.py` to generate the boundary file after preprocessing.

### How do I use the pretrained Fourier position encoder?

```bash
python pretrain_position/prepare_data.py   # generates gene pairs
python pretrain_position/train.py            # trains the encoder
python pretrain_position/export_table.py     # exports position_table.pt
```

Then in your training script, pass `position_table_path="position_table.pt"` to `GeneMamba`. See `pretrain_position/README.md` for details.

---

## License

MIT License
