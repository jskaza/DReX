import torch
import numpy as np
import pandas as pd
import os
from data import ComplexityDataset
from models import DReX
from scipy.stats import pearsonr, spearmanr

def calculate_rmse(y_true, y_pred):
    """Calculate Root Mean Square Error."""
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))

def calculate_rmae(y_true, y_pred):
    """Calculate Root Mean Absolute Error."""
    return np.sqrt(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))

def evaluate(model_class, model_ckpt_path,labels_txt,zip_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model_class(device=device).to(device)

    # Load trained model
    checkpoint = torch.load(model_ckpt_path, map_location=device)
        
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    dataset = ComplexityDataset(txt_file=labels_txt, zip_file=zip_file)

    labels = []
    preds = []
    with torch.inference_mode():
        for img, label in dataset:
            pred = model.forward_from_images([img]).cpu().item()
            labels.append(label.item())
            preds.append(pred)
    pearson, _ = pearsonr(labels, preds)
    spearman, _ = spearmanr(labels, preds)
    rmse = calculate_rmse(labels, preds)
    rmae = calculate_rmae(labels, preds)
    
    return {"pearson": pearson, "spearman": spearman, "rmse": rmse, "rmae": rmae}
    
if __name__ == "__main__":
    # IC9600 evaluation
    ic_results = evaluate(model_class=DReX,
                          model_ckpt_path="DReX.pth",
                          labels_txt="test.txt",
                          zip_file="images.zip",
                          )
    print(f"IC9600 Results: Pearson: {ic_results['pearson']:.4f}, Spearman: {ic_results['spearman']:.4f}, RMSE: {ic_results['rmse']:.4f}, RMAE: {ic_results['rmae']:.4f}")
    
    