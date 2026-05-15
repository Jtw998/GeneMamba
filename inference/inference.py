#!/usr/bin/env python3
"""
GeneMamba standalone do-operator inference script
Supports single-gene/multi-gene/pathway-level zero-shot perturbation prediction, automatically considers cis + trans effects
"""
import torch
from typing import List, Dict
from models import GeneMamba
def predict_perturbation(
    model: GeneMamba,
    baseline_expression: torch.Tensor,
    perturb_gene_indices: List[int],
    perturb_type: str = "knockout",
    perturb_value: float = None,
    device: torch.device = torch.device("cpu")
) -> Dict:
    """
    Predict genome-wide expression changes after gene perturbation, supports cis + trans effect propagation
    Args:
        model: GeneMamba model
        baseline_expression: [1, num_genes] Baseline expression profile of a single sample
        perturb_gene_indices: List of gene indices to perturb
        perturb_type: Perturbation type: "knockout" (set to 0), "overexpression", "custom" (custom value)
        perturb_value: Custom perturbation value, required when perturb_type="custom"
        device: Running device
    Returns:
        Dictionary containing baseline prediction, perturbed prediction, expression change and other results
    """
    model.eval()
    baseline_expression = baseline_expression.to(device)
    # --------------------------
    # Baseline prediction (no perturbation)
    # --------------------------
    with torch.no_grad():
        baseline_pred, _, _, _ = model(baseline_expression)
    # --------------------------
    # Construct perturbation input
    # --------------------------
    perturbed_expression = baseline_expression.clone()
    for idx in perturb_gene_indices:
        if perturb_type == "knockout":
            perturbed_expression[:, idx] = 0.0
        elif perturb_type == "overexpression":
            # Default overexpression is 95th percentile
            if perturb_value is None:
                perturb_value = torch.quantile(baseline_expression, 0.95).item()
            perturbed_expression[:, idx] = perturb_value
        elif perturb_type == "custom":
            assert perturb_value is not None, "perturb_value is required for custom type"
            perturbed_expression[:, idx] = perturb_value
        else:
            raise ValueError(f"Unsupported perturbation type: {perturb_type}")
    # --------------------------
    # Post-perturbation prediction: automatically propagate cis + trans effects
    # --------------------------
    with torch.no_grad():
        perturbed_pred, _, _, _ = model(perturbed_expression)
    # --------------------------
    # Calculate changes
    # --------------------------
    expression_change = perturbed_pred - baseline_pred
    # Genes sorted by magnitude of change
    top_upregulated = torch.argsort(expression_change, descending=True)[0, :10].tolist()
    top_downregulated = torch.argsort(expression_change, descending=False)[0, :10].tolist()
    return {
        "baseline_prediction": baseline_pred.cpu().numpy(),
        "perturbed_prediction": perturbed_pred.cpu().numpy(),
        "expression_change": expression_change.cpu().numpy(),
        "perturbed_input": perturbed_expression.cpu().numpy(),
        "top_upregulated_genes": top_upregulated,
        "top_downregulated_genes": top_downregulated,
        "perturb_gene_indices": perturb_gene_indices,
        "perturb_type": perturb_type
    }
def batch_perturbation_analysis(
    model: GeneMamba,
    baseline_expression: torch.Tensor,
    gene_indices: List[int],
    perturb_type: str = "knockout",
    device: torch.device = torch.device("cpu")
) -> Dict[int, torch.Tensor]:
    """Batch single-gene perturbation analysis, returns genome-wide changes after each gene perturbation"""
    results = {}
    for idx in gene_indices:
        res = predict_perturbation(model, baseline_expression, [idx], perturb_type, device=device)
        results[idx] = res["expression_change"]
    return results
if __name__ == "__main__":
    # Example usage
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # Load trained model
    gene_emb = torch.load("data/gene_embeddings.pt")
    model = GeneMamba(
        num_genes=21900,
        gene_emb_dim=gene_emb.shape[1],
        gene_emb=gene_emb,
        freeze_gene_emb=True
    )
    from utils import load_checkpoint
    load_checkpoint(model, "genemamba_checkpoint.pt", device)
    model = model.to(device)
    # Load example cell data (take first validation cell)
    data = torch.load("data/processed_data.pt")
    val_data = data["val"]
    gene_names = data.get("gene_names", None)
    sample = val_data[0:1, :]
    # Example: knockout MYC gene (assume MYC index is 1234, replace with actual index)
    myc_idx = 1234
    res = predict_perturbation(model, sample, [myc_idx], "knockout", device=device)
    print(f"Top 10 upregulated genes after MYC knockout: {res['top_upregulated_genes']}")
    print(f"Top 10 downregulated genes after MYC knockout: {res['top_downregulated_genes']}")
    # Print identified regulatory factors
    regulators = model.get_regulator_genes(gene_names)
    print(f"\nTotal {len(regulators)} regulatory factors identified by model, indices: {regulators.tolist()}")
