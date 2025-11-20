"""
Attention Weight Analysis for DReX Model

Plots DINO attention weight vs. ground truth complexity
and explores correlations with image-level statistics:
Entropy, Edge Density, and Color Variance.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr

from data import ComplexityDataset
from models import DReX
plt.rcParams["font.family"] = "serif" 




# ------------------------------------------------------------
# Attention extraction
# ------------------------------------------------------------
@torch.inference_mode()
def extract_attention_weights(model, dataset, device):
    """Extract attention weights + image metrics for all images."""
    labels = []
    dino_weights = []

    for img, label in tqdm(dataset, desc="Extracting attention weights"):
        # Extract DINO and ResNet features
        dino_embeddings = model.extract_dino_features([img]).to(dtype=torch.float32)
        resnet_features = model.extract_resnet_features([img]).to(dtype=torch.float32)

        # Get attention weights
        _, weights = model.fusion(dino_embeddings, resnet_features)
        dino_weight = weights[0, 0].cpu().item()

        # Compute metrics

        # Store results
        labels.append(label.item())
        dino_weights.append(dino_weight)

    return {
        "Complexity_GT": np.array(labels),
        "DINO_weight": np.array(dino_weights),
    }


# ------------------------------------------------------------
# Plot: DINO attention vs. complexity
# ------------------------------------------------------------
def plot_attention_vs_complexity(labels, dino_weights, output_path="results/dino_attention_vs_complexity.png"):
    """Create scatter plot of DINO attention weights vs. ground truth complexity with marginal histograms."""
    pearson_r, pearson_p = pearsonr(labels, dino_weights)
    spearman_r, spearman_p = spearmanr(labels, dino_weights)

    # Create figure with grid layout for main plot and marginal histograms
    fig = plt.figure(figsize=(4, 4))
    gs = fig.add_gridspec(3, 3, width_ratios=[1, 4, 0.5], height_ratios=[1, 4, 0.5],
                          hspace=0.05, wspace=0.05)
    
    # Main scatter plot
    ax_main = fig.add_subplot(gs[1, 1])
    ax_main.spines['top'].set_visible(False)
    ax_main.spines['right'].set_visible(False)


    ax_main.scatter(labels, dino_weights, alpha=0.5, s=10, edgecolors='k', linewidth=0.5)
    ax_main.set_xlim(0, 1)
    ax_main.set_ylim(0.25, 0.75)
    

    # ax_main.set_xlabel('Ground Truth Complexity', fontsize=8)
    # ax_main.set_ylabel('DINO Attention Weight', fontsize=8)
    ax_main.grid(False)
    
    # Top histogram (x-axis marginal)
    ax_top = fig.add_subplot(gs[0, 1], sharex=ax_main)
    ax_top.hist(labels, bins=30, color='gold', alpha=0.7, edgecolor='black')
    ax_top.tick_params(left=False, right=False, top=False, bottom=False, 
                       labelleft=False, labelright=False, labeltop=False, labelbottom=False)
    ax_top.spines['top'].set_visible(False)
    ax_top.spines['right'].set_visible(False)
    ax_top.spines['bottom'].set_visible(False)
    ax_top.spines['left'].set_visible(False)
    
    # Right histogram (y-axis marginal)
    ax_right = fig.add_subplot(gs[1, 2], sharey=ax_main)
    ax_right.hist(dino_weights, bins=30, orientation='horizontal', color='steelblue', alpha=0.7, edgecolor='black')
    ax_right.tick_params(left=False, right=False, top=False, bottom=False, 
                         labelleft=False, labelright=False, labeltop=False, labelbottom=False)
    ax_right.spines['top'].set_visible(False)
    ax_right.spines['right'].set_visible(False)
    ax_right.spines['bottom'].set_visible(False)
    ax_right.spines['left'].set_visible(False)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    # print(f"Mean complexity = {mean_complexity:.4f}, Mean weight = {mean_weight:.4f}")
    print(f"Pearson r = {pearson_r:.4f}, Spearman ρ = {spearman_r:.4f}")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print("Loading model...")
    model = DReX(device=device).to(device)
    model.load_state_dict(torch.load("results/DReX.pth", map_location=device))
    model.eval()

    # Load dataset
    print("Loading dataset...")
    dataset = ComplexityDataset(txt_file="ic9600/test.txt", zip_file="ic9600/images.zip")

    metrics_dict = extract_attention_weights(model, dataset, device)

    # Simple scatter (attention vs complexity)
    plot_attention_vs_complexity(metrics_dict["Complexity_GT"], metrics_dict["DINO_weight"])

