#!/usr/bin/env python3
"""
Analyze DINO feature importance in DReX model using grad × activation
=====================================================================

This script computes per-dimension importance of DINO features
for a DReX model that fuses DINO + ResNet features.

Importance is estimated as:
    E[ | ∂(output)/∂(dino_feat) * dino_feat | ]

Outputs:
 - CSV of per-dimension importance values
 - Violin plot with outlier highlights
 - Bar chart of top important dimensions
"""

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# Importance Analysis
# ============================================================
def analyze_dino_importance(model, dataset):
    """
    Compute DINO feature importance using gradient × activation.
    Args:
        model: DReX model (requires both DINO + ResNet features)
        dataset: Iterable of (PIL image, label)
    Returns:
        DataFrame with ['dim', 'importance']
    """
    device = next(model.parameters()).device
    model.eval()

    total_importance = None
    total_count = 0

    for i, (img, label) in enumerate(tqdm(dataset, desc="Computing DINO importance")):
        # Extract DINO + ResNet features
        with torch.no_grad():
            dino_feats = model.extract_dino_features([img]).to(device)
            resnet_feats = model.extract_resnet_features([img]).to(device)

        # Enable gradients on DINO features
        dino_feats.requires_grad_(True)

        # Forward through fusion + head
        fused, _ = model.fusion(dino_feats, resnet_feats)
        output = model.head(fused).sum()  # scalar for backprop

        # Backprop to get gradients wrt DINO embeddings
        output.backward()

        grads = dino_feats.grad.detach().cpu().numpy()
        feats = dino_feats.detach().cpu().numpy()

        # Compute |grad × activation|
        importance = np.abs(grads * feats)

        if total_importance is None:
            total_importance = importance.sum(axis=0)
        else:
            total_importance += importance.sum(axis=0)

        total_count += importance.shape[0]

        # Clear grads for next sample
        model.zero_grad(set_to_none=True)

    avg_importance = total_importance / total_count

    df = pd.DataFrame({
        "dim": np.arange(len(avg_importance)),
        "importance": avg_importance
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return df


# ============================================================
# Visualization utilities
# ============================================================
def plot_importance_violin(df, savepath="results/dino_importance_violin.pdf"):
    """Violin plot of DINO dimension importance."""
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.violinplot([df["importance"]], positions=[0],
                  widths=0.7, showmeans=False, showmedians=False)

    np.random.seed(42)
    x_jitter = np.random.normal(0, 0.04, size=len(df))
    ax.scatter(x_jitter, df["importance"], alpha=0.4, s=20, color='steelblue')

    ax.set_ylabel("Grad × Activation Importance")
    ax.set_xticks([])
    ax.set_title("DINO Dimension Importance Distribution")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(savepath, dpi=300, bbox_inches='tight')
    print(f"Saved violin plot to {savepath}")
    return fig, ax





# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    from models import DReX
    from data import ComplexityDataset
    from scipy.stats import skew, skewtest

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading model...")
    model = DReX(device=device).to(device)
    model.load_state_dict(torch.load("results/DReX.pth", map_location=device))
    model.eval()

    print("Loading dataset...")
    dataset = ComplexityDataset(txt_file="ic9600/test.txt", zip_file="ic9600/images.zip")

    print("Analyzing DINO feature importance...")
    df_importance = analyze_dino_importance(model, dataset)
    print(skew(df_importance["importance"]))
    print(skewtest(df_importance["importance"]))

    # Save results
    df_importance.to_csv("results/dino_dimension_importance.csv", index=False)
    print("Saved results to results/dino_dimension_importance.csv")

    # Visualize
    fig, ax = plot_importance_violin(df_importance)
   