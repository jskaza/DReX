import torch
import numpy as np
import pandas as pd
import os
from data import ComplexityDataset
from models import DReX
from ic9600.ICNet import ICNet
from d2s.D2S import D2S_R18, D2S_R50
from scipy.stats import pearsonr, spearmanr
from time import time

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
    # Create results folder if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # List to store all results
    all_results = []
    
    models = {DReX: "results/DReX.pth", ICNet: "ic9600/ICNet_ckpt.pth", D2S_R18: "d2s/D2S_R18.pth", D2S_R50: "d2s/D2S_R50.pth"}
    for model, model_ckpt_path in models.items():
        print(f"{'='*60}")
        print(f"{model.__name__} Evaluation")
        print(f"{'='*60}")
        
        # IC9600 evaluation
        ic_results = evaluate(model_class=model,
                              model_ckpt_path=model_ckpt_path,
                              labels_txt="ic9600/test.txt",
                              zip_file="ic9600/images.zip",
                              )
        print(f"IC9600 Results: Pearson: {ic_results['pearson']:.4f}, Spearman: {ic_results['spearman']:.4f}, RMSE: {ic_results['rmse']:.4f}, RMAE: {ic_results['rmae']:.4f}")
        
        # Add IC9600 results to list
        all_results.append({
            "model": model.__name__,
            "dataset": "IC9600",
            "category": "all",
            "pearson": ic_results['pearson'],
            "spearman": ic_results['spearman'],
            "rmse": ic_results['rmse'],
            "rmae": ic_results['rmae'],
        })
        
        # SAVOIAS evaluation
        savoias_results = {}
        for category in ["ad", "art", "interior_design", "objects", "scenes", "sup", "vis"]:
            savoias_results[category] = evaluate(model_class=model,
                                     model_ckpt_path=model_ckpt_path,
                                     labels_txt=f"savoias/{category}.txt",
                                     zip_file=f"savoias/{category}.zip")
            # Add individual category results
            all_results.append({
                "model": model.__name__,
                "dataset": "SAVOIAS",
                "category": category,
                "pearson": savoias_results[category]['pearson'],
                "spearman": savoias_results[category]['spearman'],
                "rmse": savoias_results[category]['rmse'],
                "rmae": savoias_results[category]['rmae'],
            })
        
        # Calculate and print SAVOIAS averages
        savoias_avg_pearson = np.mean([savoias_results[category]['pearson'] for category in savoias_results])
        savoias_avg_spearman = np.mean([savoias_results[category]['spearman'] for category in savoias_results])
        savoias_avg_rmse = np.mean([savoias_results[category]['rmse'] for category in savoias_results])
        savoias_avg_rmae = np.mean([savoias_results[category]['rmae'] for category in savoias_results])
        print(f"SAVOIAS Results: Pearson: {savoias_avg_pearson:.4f} Spearman: {savoias_avg_spearman:.4f} RMSE: {savoias_avg_rmse:.4f} RMAE: {savoias_avg_rmae:.4f}")
        
        # Add SAVOIAS average results
        all_results.append({
            "model": model.__name__,
            "dataset": "SAVOIAS",
            "category": "average",
            "pearson": savoias_avg_pearson,
            "spearman": savoias_avg_spearman,
            "rmse": savoias_avg_rmse,
            "rmae": savoias_avg_rmae,
        })
        
        # PASCAL VOC evaluation
        pascal_voc_results = evaluate(model_class=model,
                                      model_ckpt_path=model_ckpt_path,
                                      labels_txt="pascal_voc/pascal_voc.txt",
                                      zip_file="pascal_voc/images.zip")
        print(f"PASCAL VOC Results: Pearson: {pascal_voc_results['pearson']:.4f}, Spearman: {pascal_voc_results['spearman']:.4f}")
        
        # Add PASCAL VOC results to list
        all_results.append({
            "model": model.__name__,
            "dataset": "PASCAL_VOC",
            "category": "all",
            "pearson": pascal_voc_results['pearson'],
            "spearman": pascal_voc_results['spearman'],
            "rmse": pascal_voc_results['rmse'],
            "rmae": pascal_voc_results['rmae'],
        })
        
        print(f"{'='*60}")
    
    # Convert results to DataFrame and save to CSV
    df = pd.DataFrame(all_results)
    csv_path = "results/eval_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    print(f"\nDataFrame shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())