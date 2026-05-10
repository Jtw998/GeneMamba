"""
Pretrain Fourier position encoder with contrastive learning.

Task: Spatial proximity → embedding similarity
  - Positive pairs: genes within 100kb on same chromosome (nearby in genome)
  - Negative pairs: genes >1Mb apart or on different chromosomes (distant)

Loss: L = L_contrastive + λ₁·L_distance + λ₂·L_uniform

  L_contrastive: pull nearby genes together, push distant genes apart
  L_distance:    ||z_i - z_j|| ∝ log(genomic distance)
  L_uniform:     avoid collapse to a single point
"""

import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pathlib import Path

from fourier_encoder import FourierPositionEncoder


# ------------ Dataset ------------

class GenePairDataset(Dataset):
    def __init__(self, data_dir: str):
        self.coords = np.load(Path(data_dir) / "gene_coords.npy")
        with open(Path(data_dir) / "pairs.json") as f:
            pairs = json.load(f)
        self.pairs = pairs
        print(f"Dataset: {len(self.pairs)} gene pairs")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        i, j = pair["i"], pair["j"]
        label = pair["label"]
        dist = pair["dist"]
        coord_i = self.coords[i].astype(np.float32)
        coord_j = self.coords[j].astype(np.float32)
        # Normalize log distance: ln(10Mb) ≈ 16
        log_dist = min(math.log1p(dist) / 15.0, 1.0)
        return {
            "coord_i": torch.from_numpy(coord_i),
            "coord_j": torch.from_numpy(coord_j),
            "label": torch.tensor(label, dtype=torch.float32),
            "log_dist": torch.tensor(log_dist, dtype=torch.float32),
        }


# ------------ Loss ------------

class PositionContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1, lambda_dist=0.3, lambda_uniform=0.1):
        super().__init__()
        self.temperature = temperature
        self.lambda_dist = lambda_dist
        self.lambda_uniform = lambda_uniform

    def contrastive_loss(self, z_i, z_j, labels):
        z_i_n = F.normalize(z_i, dim=-1)
        z_j_n = F.normalize(z_j, dim=-1)
        sim_matrix = torch.mm(z_i_n, z_j_n.T) / self.temperature

        loss = torch.tensor(0.0, device=z_i.device)
        pos_mask = (labels == 1)
        neg_mask = (labels == 0)

        if pos_mask.any():
            logits_pos = sim_matrix[pos_mask]
            labels_pos = torch.arange(logits_pos.shape[0], device=z_i.device)
            loss = loss + F.cross_entropy(logits_pos, labels_pos)

        if neg_mask.any():
            neg_sims = sim_matrix.diag()[neg_mask]
            loss = loss + F.relu(neg_sims + 0.2).mean()

        return loss

    def distance_loss(self, z_i, z_j, log_dists):
        embed_dist = torch.norm(z_i - z_j, dim=-1)
        target = log_dists * embed_dist.detach().mean()
        return F.mse_loss(embed_dist, target)

    def uniform_loss(self, z):
        z_n = F.normalize(z, dim=-1)
        sq_pdist = torch.pdist(z_n, p=2).pow(2)
        return sq_pdist.mul(-2).exp().mean().log()

    def forward(self, z_i, z_j, labels, log_dists):
        l_contrast = self.contrastive_loss(z_i, z_j, labels)
        l_dist = self.distance_loss(z_i, z_j, log_dists)
        z_all = torch.cat([z_i, z_j], dim=0)
        l_uniform = self.uniform_loss(z_all)
        total = l_contrast + self.lambda_dist * l_dist + self.lambda_uniform * l_uniform
        return total, {
            "contrastive": l_contrast.item(),
            "distance": l_dist.item(),
            "uniform": l_uniform.item(),
        }


# ------------ Trainer ------------

class PositionPreTrainer:
    def __init__(self, config: dict):
        self.cfg = config
        self.device = torch.device(
            config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )

        self.encoder = FourierPositionEncoder(
            input_dim=config.get("input_dim", 3),
            embed_dim=config.get("embed_dim", 256),
            num_frequencies=config.get("num_frequencies", 16),
            f0=config.get("f0", 1.0),
            sigma=config.get("sigma", 2.0),
        ).to(self.device)

        self.criterion = PositionContrastiveLoss(
            temperature=config.get("temperature", 0.1),
            lambda_dist=config.get("lambda_dist", 0.3),
            lambda_uniform=config.get("lambda_uniform", 0.1),
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.encoder.parameters(),
            lr=config.get("lr", 1e-3),
            weight_decay=config.get("weight_decay", 1e-4),
        )

        dataset = GenePairDataset(config["data_dir"])
        n = len(dataset)
        n_val = int(n * 0.1)
        train_ds, val_ds = torch.utils.data.random_split(
            dataset, [n - n_val, n_val],
            generator=torch.Generator().manual_seed(42),
        )
        self.train_loader = DataLoader(
            train_ds, batch_size=config["batch_size"], shuffle=True,
            num_workers=0, pin_memory=False,
        )
        self.val_loader = DataLoader(
            val_ds, batch_size=config["batch_size"], shuffle=False,
            num_workers=0, pin_memory=False,
        )

        print(f"Encoder params: {sum(p.numel() for p in self.encoder.parameters()):,}")
        print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

        # Build LR schedule
        total_steps = config.get("num_epochs", 50) * len(self.train_loader)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_steps,
        )

    def train_epoch(self, epoch: int):
        self.encoder.train()
        total_loss = 0.0
        details = {"contrastive": 0.0, "distance": 0.0, "uniform": 0.0}
        n_batches = 0

        for batch in tqdm(self.train_loader, desc=f"Train {epoch+1}", leave=False):
            ci = batch["coord_i"].to(self.device)
            cj = batch["coord_j"].to(self.device)
            labels = batch["label"].to(self.device)
            log_dists = batch["log_dist"].to(self.device)

            zi = self.encoder(ci)
            zj = self.encoder(cj)

            loss, det = self.criterion(zi, zj, labels, log_dists)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            for k, v in det.items():
                details[k] += v
            n_batches += 1

        n = max(n_batches, 1)
        return total_loss / n, {k: v / n for k, v in details.items()}

    @torch.no_grad()
    def validate(self):
        self.encoder.eval()
        pos_sims, neg_sims = [], []

        for batch in self.val_loader:
            ci = batch["coord_i"].to(self.device)
            cj = batch["coord_j"].to(self.device)
            labels = batch["label"].to(self.device)

            zi = self.encoder(ci)
            zj = self.encoder(cj)

            cos = F.cosine_similarity(zi, zj)
            pos_sims.append(cos[labels == 1].cpu())
            neg_sims.append(cos[labels == 0].cpu())

        pos_sim = torch.cat(pos_sims).mean().item()
        neg_sim = torch.cat(neg_sims).mean().item()
        return pos_sim, neg_sim

    def train(self):
        best_val = float("-inf")
        output_dir = Path(self.cfg["output_dir"])
        ckpt_dir = output_dir / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)

        for epoch in range(self.cfg["num_epochs"]):
            train_loss, train_det = self.train_epoch(epoch)
            pos_sim, neg_sim = self.validate()
            gap = pos_sim - neg_sim
            lr = self.optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch+1:3d}/{self.cfg['num_epochs']} | "
                f"TrainLoss {train_loss:.4f} | "
                f"PosSim {pos_sim:.4f} | NegSim {neg_sim:.4f} | "
                f"Gap {gap:.4f} | LR {lr:.6f}"
            )

            if gap > best_val:
                best_val = gap
                torch.save({
                    "epoch": epoch,
                    "encoder_state_dict": self.encoder.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "pos_sim": pos_sim,
                    "neg_sim": neg_sim,
                    "gap": gap,
                    "config": self.cfg,
                }, ckpt_dir / "best_model.pt")
                print(f"  → Saved best (gap={gap:.4f})")

        print(f"\n✅ Done! Best gap={best_val:.4f}")
        print(f"   Model: {ckpt_dir / 'best_model.pt'}")


# ------------ Main ------------

def main():
    config = {
        "data_dir": str(Path(__file__).parent),
        "output_dir": str(Path(__file__).parent),
        "input_dim": 3,
        "embed_dim": 256,
        "num_frequencies": 16,
        "f0": 1.0,
        "sigma": 2.0,
        "batch_size": 512,
        "num_epochs": 50,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "temperature": 0.1,
        "lambda_dist": 0.3,
        "lambda_uniform": 0.1,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    trainer = PositionPreTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
