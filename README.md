# Hayat: Causal State Space Model for Single-Cell Transcriptomics

Zero-shot counterfactual perturbation prediction from scRNA-seq alone — no multi-modal paired data, no ATAC-seq. Full-platform: NVIDIA GPU, Apple Silicon MPS, CPU.

---

## Quick Start

### 1. Install

```bash
pip install torch einops pandas scanpy mygene
```

### 2. Train

```bash
# Default dataset (data/):
python train.py

# Custom dataset:
python train.py --data_dir path/to/dataset
```

Each dataset directory needs:
- `processed_data.pt` — expression tensor with `train`/`val` keys
- `chrom_boundaries.pt` — chromosome block boundaries
- `gene_embeddings.pt` — scGPT embeddings (optional, auto-detected)
- `gene_meta.csv` — gene metadata with chromosome and genomic position

### 3. Perturbation Prediction

```bash
python inference/inference.py
```

---

## Preprocessing Pipeline

Default training data: PBS vehicle control from **Parse-10M** (629,701 PBMC cells x 40,352 genes, 12 donors, 18 cell types).

### Step 1: Raw scRNA-seq → Model-Ready Format

Gene coordinates queried via mygene. Genes sorted by chromosome + position. CPM/10000 + log1p normalization. 80/20 train/val split.

```bash
python preprocess/preprocess_data.py --input your_data.h5ad --output_dir data/
```

Outputs in `data/`: `processed_data.pt`, `gene_meta.csv`.

### Step 2: scGPT Gene Embedding Matching

```bash
# Recommended: drop unmatched genes
python preprocess/match_gene_embeddings.py \
  --scgpt_dir /path/to/scgpt-embedding/ \
  --drop-unmatched

# Zero-fill unmatched (default)
python preprocess/match_gene_embeddings.py \
  --scgpt_dir /path/to/scgpt-embedding/

# Random-fill unmatched
python preprocess/match_gene_embeddings.py \
  --scgpt_dir /path/to/scgpt-embedding/ \
  --random-fill
```

Output: `data/gene_embeddings.pt`.

### Step 3: Chromosome Boundaries (run once per dataset)

```bash
python preprocess/generate_chrom_boundaries.py
```

Output: `data/chrom_boundaries.pt` — one `(start_idx, end_idx)` tuple per chromosome block.

### Step 4: Fourier Position Encoder Pretraining (optional)

Pretrain a genomic position encoder with contrastive learning, export as lookup table.

- 16 frequencies, embed_dim=256, f0=1.0, sigma=2.0
- Contrastive: temperature=0.1, λ_dist=0.3, λ_uniform=0.1
- Positive pairs: genes within 50kb. Negative pairs: genes over 2Mb apart
- Target: validation gap (PosSim − NegSim) ≥ 0.80

```bash
cd pretrain_position
python prepare_data.py      # gene_meta.csv → gene pairs + coords
python train.py             # contrastive pretraining (~8 min on MPS)
python export_table.py      # → ../position_table.pt (~26 MB)
```

The main training script auto-detects `position_table.pt` at the project root and loads it. Falls back to on-the-fly Fourier encoding if not found.

---

## Architecture

Cis-trans dual branch: chromosome-blocked bidirectional Mamba2 + zero-prior global regulatory gating + Fourier position encoder.

```
                    ┌─────────────────────────────────────┐
                    │         Input Expression             │
                    │           [B, N_genes]               │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │       Embedding Fusion               │
                    │   expr_emb + scGPT_emb + pos_emb     │
                    │           [B, N, 256]                │
                    └──────┬───────────────────┬──────────┘
                           │                   │
              ┌────────────▼──────┐   ┌────────▼──────────┐
              │   Cis Branch      │   │   Trans Branch     │
              │                  │   │                   │
              │  Chromosome      │   │  RegulatorGate    │
              │  blocking        │   │  MLP(gene_emb)    │
              │  (326 blocks)    │   │       │           │
              │      │           │   │  Top-K selection  │
              │  Shared BiMamba2 │   │  (K=512 active)   │
              │  (Fwd + Rev)     │   │       │           │
              │      │           │   │  Lightweight      │
              │  Block fusion    │   │  attention        │
              │      │           │   │                   │
              │  cis_out         │   │  trans_out        │
              │  [B, N, 256]     │   │  [B, N, 256]      │
              └────────┬─────────┘   └────────┬──────────┘
                       │                      │
                       └──────────┬───────────┘
                                  │  cis_out * (1 + tanh(trans_out))
                    ┌─────────────▼───────────────────────┐
                    │        VAE Latent Space              │
                    │  latent_mean, latent_log_var         │
                    │  [B, N, 64]                          │
                    │       │                              │
                    │  Reparameterization                  │
                    │  latent_sample [B, N, 64]            │
                    └─────────────┬───────────────────────┘
                                  │
                    ┌─────────────▼───────────────────────┐
                    │     Causal Gating Output             │
                    │     [B, N_genes]                     │
                    └─────────────────────────────────────┘
```

---

## Hyperparameters

All in `utils/utils.py`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_dim` | 256 | Mamba / projection hidden dimension |
| `num_mamba_layers` | 2 | Bidirectional Mamba layers per direction |
| `latent_dim` | 64 | VAE latent dimension |
| `learning_rate` | 1e-4 | AdamW learning rate |
| `weight_decay` | 1e-5 | AdamW weight decay |
| `grad_clip_value` | 1.0 | Gradient clipping |
| `batch_size` | 16 | Training batch size |
| `num_epochs` | 100 | Training epochs |
| `max_train_cells` | 100000 | Cap on training cells (0 = all) |
| `nb_loss_weight` | 1.0 | Negative binomial loss weight |
| `sparsity_loss_weight` | 0.5 | Latent sparsity weight |

Model constructor defaults: `max_regulators=512`, `chunk_size=64`, `d_state=16`, `d_conv=4`, `expand=2`, `dropout=0.1`.

---

## Cross-Dataset Inference

Gene count is not hardcoded. Two components make the model shape-invariant:

1. **Dynamic embedding lookup** — gene embeddings indexed by name, not position. Pass `gene_names_list` at forward time.
2. **RegulatorGate MLP** — a lightweight MLP over gene embeddings replaces the old fixed-size parameter. Always outputs `[num_genes]` regardless of gene count (~33k params, <0.2% of total).

Loading old checkpoints:
```python
from models import load_with_migration
model = load_with_migration(model, "old_checkpoint.pt")
# Old regulator_gate is skipped; all other weights migrate if shapes match.
```

---

## Perturbation Evaluation (Schmidt)

See `Schmidt/` for the perturbation evaluation pipeline using the Schmidt perturb-seq dataset.

```bash
cd Schmidt
python preprocess.py --scgpt_dir /path/to/scgpt-embedding
python evaluate.py --checkpoint ../checkpoints/hayat_checkpoint.pt
```

Six standardized metrics: MSE, E-distance, PCC-delta, Wasserstein, KL-divergence, Common-DEGs.

---

## Project Structure

```
Hayat/
├── train.py                        # Entry point (--data_dir flag)
├── inference/inference.py          # do-operator inference
│
├── models/model.py                 # Hayat, Mamba2, RegulatorGate,
│                                   #   PositionEncoder, load_with_migration
├── train/run_training.py           # Training loop
│
├── utils/
│   ├── utils.py                    # Config, metrics, checkpoint I/O
│   └── losses.py                   # NB loss, sparsity, decoupling, smoothness
│
├── preprocess/
│   ├── preprocess_data.py           # h5ad → processed tensors
│   ├── match_gene_embeddings.py     # scGPT embedding matching
│   └── generate_chrom_boundaries.py # Chromosome block boundaries
│
├── pretrain_position/
│   ├── prepare_data.py              # gene_meta.csv → gene pairs + coords
│   ├── fourier_encoder.py           # Fourier position encoder
│   ├── train.py                     # Contrastive learning
│   └── export_table.py              # Export position_table.pt
│
├── Schmidt/
│   ├── preprocess.py                # Schmidt dataset prep
│   └── evaluate.py                  # 6-metric evaluation
│
├── tests/debug_memory.py            # Memory profiling
├── checkpoints/                     # Saved model weights
├── data/                            # Preprocessed data
│   ├── processed_data.pt
│   ├── gene_meta.csv
│   ├── gene_embeddings.pt
│   └── chrom_boundaries.pt
└── position_table.pt                # Pretrained Fourier lookup table
```

---

## FAQ

### Out of memory

Reduce `batch_size` in `utils/utils.py` (16 → 8 or 4). On Apple Silicon: `PYTORCH_ENABLE_MPS_FALLBACK=1 python train.py`.

### Gene count mismatch

Run with `--drop-unmatched`:
```bash
python preprocess/match_gene_embeddings.py --scgpt_dir /path/to/scgpt --drop-unmatched
```
This synchronously filters all data files so only genes with valid scGPT embeddings remain.

### No GPU

Falls back automatically: MPS (Apple Silicon) → CUDA → CPU. No code changes.

### Cross-dataset inference (different gene sets)

Pass `gene_names_list` to the model forward call. The RegulatorGate MLP and dynamic embedding lookup are shape-invariant.

### Chromosome boundaries fail to load

Model falls back to whole-genome single-block mode. Run `python preprocess/generate_chrom_boundaries.py` to generate boundaries.

### Using the pretrained position encoder

```bash
python pretrain_position/prepare_data.py
python pretrain_position/train.py
python pretrain_position/export_table.py
```
The training script auto-detects `position_table.pt`. No extra flags needed.

---

## License

MIT
