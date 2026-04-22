# GeneMamba: Pure Endogenous Causal State Space Model for Single-Cell Transcriptomics

> Zero-shot counterfactual perturbation prediction from scRNA-seq alone — no multi-modal paired data, no ATAC-seq required. Full-platform compatible: NVIDIA GPU, Apple Silicon MPS, and CPU.

---

## Core Advantages

| Feature | Description |
|---------|-------------|
| **Pure Endogenous Causality** | Trained only on scRNA-seq data. Naturally blocks backdoor paths. Observation distribution is strictly equivalent to intervention distribution. |
| **Zero-shot Perturbation Prediction** | No intervention data during training. Directly performs counterfactual reasoning: gene knockout, overexpression, and pathway-level intervention — with native do-operator support. |
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
python train_v0.1.py

# Custom dataset:
python train_v0.1.py --data_dir /path/to/your/dataset
```

The `--data_dir` flag selects the dataset directory. Each dataset should provide:
- `processed_data.pt` — expression tensor with `train`/`val` keys
- `chrom_boundaries.pt` — chromosome block boundaries
- `gene_embeddings.pt` — scGPT embeddings (optional, auto-detected)

### Perturbation prediction

```bash
python inference/inference_v0.1.py
```

---

## Complete Preprocessing Pipeline

All commands run from the project root. No `cd` required.

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
|------|------|----------|
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

### Perturbation Evaluation Dataset (Schmidt)

See [`Schmidt/`](Schmidt/) for the perturbation evaluation pipeline using the Schmidt perturb-seq dataset. Train on any dataset, evaluate perturbation prediction accuracy with 6 standardized metrics.

---

## Project Structure

```
GeneMamba/
├── train_v0.1.py                # Training entry (--data_dir flag)
├── inference_v0.1.py          # Inference entry
│
├── models/                      # Model definitions
│   └── model.py               # GeneMambaV2 (cis-trans dual branch)
│
├── train/                       # Training logic
│   └── run_training_v0.1.py   # End-to-end training script
│
├── inference/                   # Inference logic
│   └── inference_v0.1.py
│
├── preprocess/                 # Data preprocessing
│   ├── preprocess_data.py       # h5ad → model-ready tensors
│   ├── match_gene_embeddings.py # scGPT embedding matching
│   └── generate_chrom_boundaries.py # Chromosome block boundaries
│
├── utils/                       # Shared utilities
│
├── utils/                       # Shared utilities
│   ├── utils.py                 # Config, metrics, checkpoint I/O
│   └── losses.py                # Loss functions (NB, sparsity, etc.)
│
├── docs/                        # Technical documentation
│   └── V0.1.md               # Architecture, memory optimization, interface reference
│
├── tests/                       # Debugging / profiling scripts
├── checkpoints/                 # Auto-saved model weights
└── data/                        # Preprocessed data (auto-created)
    ├── processed_data.pt       # Train/val expression tensors
    ├── gene_meta.csv           # Gene metadata
    ├── gene_embeddings.pt      # scGPT embeddings
    └── chrom_boundaries.pt     # Chromosome block indices
└── Schmidt/                    # Perturbation evaluation dataset
    ├── preprocess.py            # Preprocessing (h5ad → model format + embeddings)
    ├── evaluate.py             # 6-metric evaluation script
    ├── schmidt_data.pt        # Expression tensor
    ├── schmidt_gene_meta.csv # Gene metadata
    ├── schmidt_chrom_boundaries.pt  # Chromosome block boundaries
    ├── schmidt_perturb_labels.pt     # Perturbation labels
    └── schmidt_gene_embeddings.pt    # scGPT embeddings
```

---

## Architecture

Cis-trans dual branch: chromosome-block shared bidirectional Mamba2 + zero-prior global regulatory gating.

- Training memory: **≤7 GB**
- Training speed: **7–8× faster** than baseline
- TF target gene recall: **≥80%**
- Cross-chromosome false positives: **>50% reduction**
- New parameters: +21,900 (<0.2%, no overfitting risk)

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

---

## Frequently Asked Questions

### V2 training runs out of memory (OOM)

V2 is designed for ≤7 GB with default settings. If you hit OOM:

1. Reduce `batch_size` in `utils/utils.py` (default 16 → try 8 or 4)
2. Reduce `max_regulators` at model initialization (default 512 → try 256 or 128)
3. On Apple Silicon, enable MPS fallback: `PYTORCH_ENABLE_MPS_FALLBACK=1 python train_v0.1.py`

### Gene count mismatch between data and embeddings

Run with `--drop-unmatched` — this synchronously filters all data files so only genes with valid scGPT embeddings remain:

```bash
python preprocess/match_gene_embeddings.py --scgpt_dir /path/to/scgpt --drop-unmatched
```

### Can I run without a GPU?

Yes. GeneMamba falls back automatically: MPS (Apple Silicon) → CUDA → CPU. No code changes needed.

### What if chromosome boundaries fail to load?

V2 falls back to whole-genome single-block mode (slower, higher memory). Run `python preprocess/generate_chrom_boundaries.py` to generate the boundary file after preprocessing.



## License

MIT License
