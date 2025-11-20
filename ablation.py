"""
Ablation analysis for the DReX model.

This module provides tools for analyzing the contribution of different model components
through complete ablation (removing features from both attention and residual).

Supports ablation of:
1. Entire branches (DINO or ResNet)
2. Specific ResNet layers (1, 2, 3, 4)
3. Specific DINO dimensions (0-383)
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Tuple, Literal, Optional, List, Dict
from scipy.stats import pearsonr, spearmanr

from data import ComplexityDataset
from models import DReX


class AblatedDReX(DReX):
    """
    DReX model with feature ablation support.

    Supports three types of ablation:
    1. Complete branch ablation (entire DINO or ResNet branch)
    2. ResNet layer ablation (specific layers 1-4)
    3. DINO dimension ablation (specific dimensions 0-383)
    """

    def __init__(
        self,
        base_model: DReX,
        ablate_branch: Optional[Literal["dino", "resnet"]] = None,
        ablate_resnet_layers: Optional[List[int]] = None,
        ablate_dino_dims: Optional[List[int]] = None,
    ):
        """
        Initialize an ablated DReX model.

        Args:
            base_model: Original trained DReX model to ablate
            ablate_branch: Which entire branch to ablate - 'dino' or 'resnet' (mutually exclusive with other options)
            ablate_resnet_layers: List of ResNet layer numbers to ablate (0, 1, 2, or 3)
            ablate_dino_dims: List of DINO dimension indices to ablate (0-383)

        Raises:
            ValueError: If invalid combination of ablation parameters
        """
        # Validate mutually exclusive options
        options = [ablate_branch is not None, ablate_resnet_layers is not None, ablate_dino_dims is not None]
        if sum(options) != 1:
            raise ValueError(
                "Exactly one ablation type must be specified: ablate_branch, ablate_resnet_layers, or ablate_dino_dims"
            )

        if ablate_branch is not None and ablate_branch not in ["dino", "resnet"]:
            raise ValueError(f"ablate_branch must be 'dino' or 'resnet', got: {ablate_branch}")

        if ablate_resnet_layers is not None:
            if not all(layer in [0, 1, 2, 3] for layer in ablate_resnet_layers):
                raise ValueError(f"ablate_resnet_layers must contain only values 1-4, got: {ablate_resnet_layers}")

        if ablate_dino_dims is not None:
            if not all(0 <= dim < 384 for dim in ablate_dino_dims):
                raise ValueError(f"ablate_dino_dims must contain values 0-383, got: {ablate_dino_dims}")

        # Don't call super().__init__() - copy from base model instead
        super(DReX, self).__init__()

        # Copy all attributes from base model
        self.device = base_model.device
        self.dino_processor = base_model.dino_processor
        self.dino_model = base_model.dino_model
        self.dino_embed_dim = base_model.dino_embed_dim
        self.resnet_backbone = base_model.resnet_backbone
        self.resnet_embed_dim = base_model.resnet_embed_dim
        self.resnet_transform = base_model.resnet_transform
        self.fusion = base_model.fusion
        self.head = base_model.head

        # Store ablation configuration
        self.ablate_branch = ablate_branch
        self.ablate_resnet_layers = ablate_resnet_layers
        self.ablate_dino_dims = ablate_dino_dims

        # Compute layer dimensions for ResNet (256, 512, 1024, 2048 for ResNet50)
        self._layer_dims = [256, 512, 1024, 2048]

    def extract_resnet_features(self, images):
        """Extract ResNet features with optional layer ablation."""
        resnet_imgs = torch.stack([self.resnet_transform(img) for img in images]).to(self.device)
        
        with torch.no_grad():
            # Manual forward through ResNet backbone to get individual layers
            x = self.resnet_backbone.conv1(resnet_imgs)
            x = self.resnet_backbone.bn1(x)
            x = self.resnet_backbone.relu(x)
            x = self.resnet_backbone.maxpool(x)

            f1 = self.resnet_backbone.layer1(x)
            f2 = self.resnet_backbone.layer2(f1)
            f3 = self.resnet_backbone.layer3(f2)
            f4 = self.resnet_backbone.layer4(f3)

            feats = []
            for i, f in enumerate([f1, f2, f3, f4]):
                pooled = F.adaptive_avg_pool2d(f, (1, 1))
                pooled = pooled.squeeze(-1).squeeze(-1)
                
                # Ablate specific layers if requested
                if self.ablate_resnet_layers is not None and i in self.ablate_resnet_layers:
                    pooled = torch.zeros_like(pooled)
                
                feats.append(pooled)
            
            return torch.cat(feats, dim=-1)

    def extract_dino_features(self, images):
        """Extract DINO features with optional dimension ablation."""
        pixel_values = self.dino_processor(images, return_tensors="pt")['pixel_values'].to(self.device)
        
        with torch.no_grad():
            dino_out = self.dino_model(pixel_values)
            embeddings = dino_out.last_hidden_state[:, 0]
            
            # Ablate specific dimensions if requested
            if self.ablate_dino_dims is not None:
                embeddings = embeddings.clone()
                embeddings[:, self.ablate_dino_dims] = 0.0
            
            return embeddings

    def forward_from_embeddings(
        self, dino_embeddings: torch.Tensor, resnet_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass with ablation applied.

        Args:
            dino_embeddings: DINO feature embeddings [batch_size, dino_embed_dim]
            resnet_features: ResNet feature embeddings [batch_size, resnet_embed_dim]

        Returns:
            Predicted complexity scores [batch_size]
        """
        # Get the projected features
        d_proj = self.fusion.dino_proj(dino_embeddings)
        r_proj = self.fusion.resnet_proj(resnet_features)

        # Create attention weights based on complete branch ablation
        combined = torch.cat([d_proj, r_proj], dim=-1)
        weights = F.softmax(self.fusion.attention(combined) / self.fusion.temperature, dim=-1)
        fused = weights[:, 0:1] * d_proj + weights[:, 1:2] * r_proj
        fused = fused + self.fusion.alpha * (d_proj + r_proj)

        fused = F.layer_norm(fused, fused.shape[-1:])

        # Run through the prediction head
        preds = self.head(fused).squeeze(-1)
        return preds


def load_dataset_images(dataset: ComplexityDataset) -> Tuple[List, np.ndarray]:
    """
    Load all images and labels from a dataset once.

    Args:
        dataset: Dataset to load from

    Returns:
        Tuple of (images list, labels array)
    """
    images = []
    labels = []
    for img, label in tqdm(dataset, desc="Loading dataset"):
        images.append(img)
        labels.append(label.item())
    
    return images, np.array(labels)


@torch.inference_mode()
def extract_and_cache_embeddings(
    model: DReX, images: List, device: torch.device, batch_size: int = 256
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract and cache DINO and ResNet embeddings for all images.
    
    This is done once at the start to avoid recomputing embeddings
    for every ablation experiment.
    
    Args:
        model: DReX model to extract features with
        images: List of PIL images
        device: Device to run on
        batch_size: Batch size for extraction
        
    Returns:
        Tuple of (dino_embeddings, resnet_embeddings) tensors
    """
    all_dino_embeddings = []
    all_resnet_embeddings = []
    
    num_batches = (len(images) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(images), batch_size), total=num_batches, desc="Caching embeddings"):
        batch_images = images[i:i + batch_size]
        
        # Extract DINO embeddings
        dino_emb = model.extract_dino_features(batch_images)
        all_dino_embeddings.append(dino_emb.cpu())
        
        # Extract ResNet embeddings
        resnet_emb = model.extract_resnet_features(batch_images)
        all_resnet_embeddings.append(resnet_emb.cpu())
    
    # Concatenate all batches
    dino_embeddings = torch.cat(all_dino_embeddings, dim=0)
    resnet_embeddings = torch.cat(all_resnet_embeddings, dim=0)
    
    return dino_embeddings, resnet_embeddings


@torch.inference_mode()
@torch.inference_mode()
def predict_from_cached_embeddings(
    model: AblatedDReX,
    dino_embeddings: torch.Tensor,
    resnet_embeddings: torch.Tensor,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """
    Run model predictions using pre-cached embeddings.
    
    This is much faster than re-extracting features since it skips the expensive
    DINO and ResNet forward passes.
    
    Args:
        model: Ablated DReX model
        dino_embeddings: Pre-computed DINO embeddings [N, 384]
        resnet_embeddings: Pre-computed ResNet embeddings [N, 3840]
        device: Device to run on
        batch_size: Batch size for inference
        
    Returns:
        Predictions as numpy array
    """
    all_preds = []
    for i in range(0, len(dino_embeddings), batch_size):
        # Get batch and move to device
        batch_dino = dino_embeddings[i:i + batch_size].to(device)
        batch_resnet = resnet_embeddings[i:i + batch_size].to(device)
        
        # Apply ablations to embeddings BEFORE forward pass
        if model.ablate_branch == "dino":
            batch_dino = torch.zeros_like(batch_dino)
        elif model.ablate_branch == "resnet":
            batch_resnet = torch.zeros_like(batch_resnet)
        elif model.ablate_dino_dims is not None:
            batch_dino = batch_dino.clone()
            batch_dino[:, model.ablate_dino_dims] = 0.0
        elif model.ablate_resnet_layers is not None:
            # Zero out specific ResNet layers
            batch_resnet = batch_resnet.clone()
            layer_dims = [256, 512, 1024, 2048]
            start_idx = 0
            for layer_idx, layer_dim in enumerate(layer_dims):
                if layer_idx in model.ablate_resnet_layers:
                    batch_resnet[:, start_idx:start_idx + layer_dim] = 0.0
                start_idx += layer_dim
        
        # Forward pass with ablated embeddings (attention will adapt naturally)
        batch_preds = model.forward_from_embeddings(batch_dino, batch_resnet).detach().cpu().numpy()
        all_preds.append(batch_preds)
    
    return np.concatenate(all_preds)


def bootstrap_corr_diff(
    labels: np.ndarray,
    preds_full: np.ndarray,
    preds_ablated: np.ndarray,
    n_boot: int = 10000,
    n_perm: int = 10000,
    seed: int = 42
) -> Dict:
    """
    Bootstrap test for correlation difference between full and ablated models.

    Uses bootstrap resampling to estimate the distribution of the correlation
    difference and compute confidence intervals for both Pearson and Spearman.

    Args:
        labels: Ground truth labels
        preds_full: Predictions from full model
        preds_ablated: Predictions from ablated model
        n_boot: Number of bootstrap iterations
        n_perm: Number of permutation iterations
        seed: Random seed for reproducibility

    Returns:
        Dictionary with bootstrap statistics including p-values
    """
    rng = np.random.default_rng(seed)
    n = len(labels)

    # Observed correlations (on original, non-resampled data)
    r_full_p_obs = pearsonr(labels, preds_full)[0]
    r_ablated_p_obs = pearsonr(labels, preds_ablated)[0]
    obs_diff_p = r_ablated_p_obs - r_full_p_obs

    r_full_s_obs = spearmanr(labels, preds_full)[0]
    r_ablated_s_obs = spearmanr(labels, preds_ablated)[0]
    obs_diff_s = r_ablated_s_obs - r_full_s_obs

    # Paired bootstrap (resample indices with replacement)
    pearson_diffs = np.zeros(n_boot)
    spearman_diffs = np.zeros(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = labels[idx]
        pf = preds_full[idx]
        pa = preds_ablated[idx]

        # Pearson
        r_full_p = pearsonr(yb, pf)[0]
        r_ablated_p = pearsonr(yb, pa)[0]
        pearson_diffs[b] = r_ablated_p - r_full_p

        # Spearman
        r_full_s = spearmanr(yb, pf)[0]
        r_ablated_s = spearmanr(yb, pa)[0]
        spearman_diffs[b] = r_ablated_s - r_full_s

    # Bootstrap summary (mean, CI)
    mean_diff_p = np.mean(pearson_diffs)
    ci_lower_p, ci_upper_p = np.percentile(pearson_diffs, [0.5, 99.5])  # 95% CI
    # Approximate bootstrap two-sided p-value (center the bootstrap under null)
    # center the boot distribution so its mean is zero (null: mean diff = 0)
    centered_p = pearson_diffs - mean_diff_p
    # compute how many centered bootstrap values are >= |obs - mean_boot|
    p_boot_p = np.mean(np.abs(centered_p) >= abs(obs_diff_p - mean_diff_p))

    mean_diff_s = np.mean(spearman_diffs)
    ci_lower_s, ci_upper_s = np.percentile(spearman_diffs, [0.5, 99.5])
    centered_s = spearman_diffs - mean_diff_s
    p_boot_s = np.mean(np.abs(centered_s) >= abs(obs_diff_s - mean_diff_s))

    # Paired permutation test (recommended for p-value)
    # For each index randomly swap pf and pa with probability 0.5
    perm_diffs_p = np.zeros(n_perm)
    perm_diffs_s = np.zeros(n_perm)
    for i in range(n_perm):
        # Boolean mask: True = swap predictions at that position
        swap_mask = rng.random(n) < 0.5
        pf_perm = preds_full.copy()
        pa_perm = preds_ablated.copy()
        # swap where mask True
        pf_perm[swap_mask], pa_perm[swap_mask] = pa_perm[swap_mask], pf_perm[swap_mask]
        # recompute correlations on swapped data
        perm_diffs_p[i] = pearsonr(labels, pa_perm)[0] - pearsonr(labels, pf_perm)[0]
        perm_diffs_s[i] = spearmanr(labels, pa_perm)[0] - spearmanr(labels, pf_perm)[0]

    # Two-sided permutation p-value: fraction of |perm_diffs| >= |observed_diff|
    p_perm_p = np.mean(np.abs(perm_diffs_p) >= abs(obs_diff_p))
    p_perm_s = np.mean(np.abs(perm_diffs_s) >= abs(obs_diff_s))

    return {
        "observed_pearson_diff": obs_diff_p,
        "observed_spearman_diff": obs_diff_s,

        "pearson_mean_diff_boot": mean_diff_p,
        "pearson_ci_95": (ci_lower_p, ci_upper_p),
        "pearson_p_value_boot": p_boot_p,      # approx bootstrap two-sided p
        "pearson_p_value_perm": p_perm_p,      # permutation two-sided p 

        "spearman_mean_diff_boot": mean_diff_s,
        "spearman_ci_95": (ci_lower_s, ci_upper_s),
        "spearman_p_value_boot": p_boot_s,     # approx bootstrap two-sided p
        "spearman_p_value_perm": p_perm_s,     # permutation two-sided p 
    }


def run_ablation(
    ablated_model: AblatedDReX,
    dino_embeddings: torch.Tensor,
    resnet_embeddings: torch.Tensor,
    labels: np.ndarray,
    preds_full: np.ndarray,
    ablation_type: str,
    ablation_details: str,
) -> Dict:
    """
    Run a single ablation experiment and compute statistics.
    
    Args:
        ablated_model: Ablated DReX model
        dino_embeddings: Cached DINO embeddings
        resnet_embeddings: Cached ResNet embeddings
        labels: Ground truth labels
        preds_full: Full model predictions
        ablation_type: Type of ablation (branch, resnet_layer, dino_dim)
        ablation_details: Details of what was ablated
        
    Returns:
        Dictionary with ablation statistics
    """
    preds_ablated = predict_from_cached_embeddings(
        ablated_model, dino_embeddings, resnet_embeddings, ablated_model.device
    )
    r_p = pearsonr(labels, preds_ablated)[0]
    r_s = spearmanr(labels, preds_ablated)[0]
    r_full_p = pearsonr(labels, preds_full)[0]
    r_full_s = spearmanr(labels, preds_full)[0]
    
    bootstrap_stats = bootstrap_corr_diff(labels, preds_full, preds_ablated)

    return {
        "ablation_type": ablation_type,
        "ablation_details": ablation_details,
        "pearson_corr": r_p,
        "spearman_corr": r_s,
        "pearson_delta": r_p - r_full_p,
        "spearman_delta": r_s - r_full_s,
        **bootstrap_stats,
    }


def full_ablation_analysis(
    model_path: str = "results/DReX.pth",
    ic9600_txt: str = "ic9600/test.txt",
    ic9600_zip: str = "ic9600/images.zip",
) -> pd.DataFrame:
    """
    Run complete ablation analysis: branches, ResNet layers, and DINO dimensions.
    
    Uses cached DINO and ResNet embeddings to drastically speed up the ablation loops.
    Embeddings are extracted once and reused for all 390+ ablation experiments.
    
    Returns:
        DataFrame with all ablation results including bootstrapped p-values
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Load model and dataset once
    print("Loading model...")
    base = DReX(device=device).to(device)
    base.load_state_dict(torch.load(model_path, map_location=device))
    base.eval()
    
    # Load all images and labels once (will be reused for all ablations)
    dataset = ComplexityDataset(txt_file=ic9600_txt, zip_file=ic9600_zip)
    images, labels = load_dataset_images(dataset)
    print(f"Loaded {len(images)} images from dataset\n")
    
    # Cache embeddings once (this is the key optimization!)
    print("Extracting and caching DINO/ResNet embeddings...")
    dino_embeddings, resnet_embeddings = extract_and_cache_embeddings(base, images, device)
    print(f"✓ Cached embeddings: DINO {dino_embeddings.shape}, ResNet {resnet_embeddings.shape}\n")
    
    # Get baseline predictions from cached embeddings (no ablation)
    print("Computing full model predictions from cached embeddings...")
    preds_full = []
    for i in range(0, len(dino_embeddings), 256):
        batch_preds = base.forward_from_embeddings(
            dino_embeddings[i:i + 256].to(device),
            resnet_embeddings[i:i + 256].to(device)
        ).detach().cpu().numpy()
        preds_full.append(batch_preds)
    preds_full = np.concatenate(preds_full)
    
    r_full_p = pearsonr(labels, preds_full)[0]
    r_full_s = spearmanr(labels, preds_full)[0]
    print(f"✓ Full model: Pearson={r_full_p:.4f}, Spearman={r_full_s:.4f}\n")
    
    results = [{
        "ablation_type": "full",
        "ablation_details": "none",
        "pearson_corr": r_full_p,
        "spearman_corr": r_full_s,
        "pearson_delta": 0.0,
        "spearman_delta": 0.0,
        "pearson_p_value": np.nan,
        "spearman_p_value": np.nan,
        "pearson_ci_lower": np.nan,
        "pearson_ci_upper": np.nan,
        "spearman_ci_lower": np.nan,
        "spearman_ci_upper": np.nan,
        "pearson_mean_diff": np.nan,
        "spearman_mean_diff": np.nan,
    }]
    
    # Branch ablations
    print("\n" + "="*70)
    print("BRANCH ABLATIONS")
    print("="*70)
    for branch in ["dino", "resnet"]:
        print(f"\nAblating {branch.upper()} branch...")
        model = AblatedDReX(base, ablate_branch=branch).eval()
        results.append(run_ablation(
            model, dino_embeddings, resnet_embeddings, labels, preds_full, 
            "branch", f"{branch}_ablated"
        ))
    
    # ResNet layer ablations
    print("\n" + "="*70)
    print("RESNET LAYER ABLATIONS")
    print("="*70)
    for layer in range(4):
        print(f"\nAblating ResNet block {layer}...")
        model = AblatedDReX(base, ablate_resnet_layers=[layer]).eval()
        results.append(run_ablation(
            model, dino_embeddings, resnet_embeddings, labels, preds_full,
            "resnet_layer", f"layer_{layer+1}"
        ))
    
    # DINO dimension ablations
    print("\n" + "="*70)
    print("DINO DIMENSION ABLATIONS")
    print("="*70)
    for dim in tqdm(range(384), desc="Ablating DINO dimensions"):
        model = AblatedDReX(base, ablate_dino_dims=[dim]).eval()
        results.append(run_ablation(
            model, dino_embeddings, resnet_embeddings, labels, preds_full,
            "dino_dim", f"dim_{dim+1}"
        ))
    
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    """Run all ablation experiments and save results."""
    import matplotlib.pyplot as plt
    from scipy.stats import false_discovery_control
    plt.rcParams["font.family"] = "serif" 
    
    # uncomment to re-run the full ablation analysis
    df = full_ablation_analysis()
    output_file = "results/ablation_results.csv"
    df.to_csv(output_file, index=False)

    plt.figure(figsize=(4, 2))
    df = pd.read_csv("results/ablation_results.csv")  
    df_dino_dim = df[df['ablation_type'] == 'dino_dim']
    fdr = false_discovery_control(df_dino_dim['pearson_p_value_perm'].values)
    plt.bar(df_dino_dim['ablation_details'], df_dino_dim['pearson_delta'], color=np.where(fdr < 0.01, 'blue', 'black'), width=2)
    print(df_dino_dim.sort_values(by='pearson_delta', ascending=True).head(20)[['ablation_details', 'pearson_p_value_perm', 'pearson_delta']])
    print(np.where(fdr < 0.01))
    plt.xlabel('DINO Dim')
    plt.xticks([], [])
    plt.ylabel('Pearson Delta')
    plt.title('Pearson Delta vs. DINO Dim')
    plt.savefig('results/dino_dim_pearson_delta.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(2, 2))
    df_resnet_layer = df[df['ablation_type'] == 'resnet_layer']
    fdr = false_discovery_control(df_resnet_layer['pearson_p_value_perm'].values)
    plt.bar(df_resnet_layer['ablation_details'], df_resnet_layer['pearson_delta'], color=np.where(fdr < 0.01, 'red', 'black'))
    plt.xlabel('ResNet Layer')
    plt.ylabel('Pearson Delta')
    plt.title('Pearson Delta vs. ResNet Layer')
    plt.savefig('results/resnet_layer_pearson_delta.pdf', dpi=300, bbox_inches='tight')
    plt.close()

  
    