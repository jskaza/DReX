import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt
from data import ComplexityDataset
from utils import set_seed
from models import DReX
import time

def train_step(model, batch, optimizer, device, scheduler, ema_fusion=None, ema_head=None, ema_decay=0.999):
    """Single training step using precomputed embeddings."""
    dino_inputs, resnet_inputs, scores = batch
    dino_inputs = dino_inputs.to(device)
    resnet_inputs = resnet_inputs.to(device)
    scores = scores.to(device).view(-1)

    optimizer.zero_grad()
    preds = model.forward_from_embeddings(dino_inputs, resnet_inputs)
    loss = F.huber_loss(preds, scores)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

    # EMA update
    if ema_fusion is not None and ema_head is not None:
        with torch.no_grad():
            for ema_param, model_param in zip(ema_fusion.parameters(), model.fusion.parameters()):
                ema_param.data.mul_(ema_decay).add_(model_param.data, alpha=1 - ema_decay)
            for ema_param, model_param in zip(ema_head.parameters(), model.head.parameters()):
                ema_param.data.mul_(ema_decay).add_(model_param.data, alpha=1 - ema_decay)

    return loss.item()


def main():
    set_seed()

    num_epochs = 10
    batch_size = 16
    learning_rate = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}\nUsing device: {device}\n{'='*60}")

    # ------------------------------
    # Load dataset
    # ------------------------------
    print("Loading dataset...")
    train_dataset = ComplexityDataset(
        txt_file="train.txt",
        zip_file="images.zip"
    )
    print(f"✓ Loaded {len(train_dataset)} samples")

    # ------------------------------
    # Initialize model
    # ------------------------------
    print("Initializing model...")
    model = DReX(device=device).to(device)
    model.eval()  # No gradient for feature extraction
    print("✓ Model initialized")

    # ------------------------------
    # Precompute all embeddings
    # ------------------------------
    print("\nPrecomputing embeddings (this may take a few minutes)...")
    all_dino, all_resnet, all_scores = [], [], []

    for img, score in tqdm(train_dataset, desc="Extracting"):
        # Model expects a list of PIL images
        dino_emb = model.extract_dino_features([img])[0].cpu()
        resnet_emb = model.extract_resnet_features([img])[0].cpu()
        all_dino.append(dino_emb)
        all_resnet.append(resnet_emb)
        all_scores.append(score)

    all_dino = torch.stack(all_dino)
    all_resnet = torch.stack(all_resnet)
    all_scores = torch.tensor(all_scores, dtype=torch.float32)

    print(f"✓ Precomputed {len(all_scores)} embeddings")
    print(f"DINO shape: {all_dino.shape}, ResNet shape: {all_resnet.shape}")

    # ------------------------------
    # Build DataLoader from embeddings
    # ------------------------------
    emb_dataset = TensorDataset(all_dino, all_resnet, all_scores)
    emb_loader = DataLoader(emb_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    # ------------------------------
    # Train only the fusion + head
    # ------------------------------
    print("Switching to training mode for fusion+head...")
    model.train()
    for p in model.dino_model.parameters():
        p.requires_grad = False
    for p in model.resnet_backbone.parameters():
        p.requires_grad = False

    ema_fusion = copy.deepcopy(model.fusion).eval()
    ema_head = copy.deepcopy(model.head).eval()

    trainable_params = list(model.fusion.parameters()) + list(model.head.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)
    total_steps = len(emb_loader) * num_epochs
    scheduler = OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        total_steps=total_steps
    )

    print(f"\n{'='*60}\nTraining\n{'='*60}")
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}\n{'='*60}\n")

    start_time = time.time()
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        model.train()
        total_loss = 0.0

        for batch in tqdm(emb_loader, desc="Batches", leave=False):
            loss = train_step(
                model, batch, optimizer, device,
                scheduler=scheduler,
                ema_fusion=ema_fusion,
                ema_head=ema_head
            )
            total_loss += loss

        avg_loss = total_loss / len(emb_loader)
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"Epoch {epoch+1:02d} | Loss: {avg_loss:.4f} | LR: {current_lr:.2e}")
    end_time = time.time()
    print(f"Training time: {end_time - start_time:.2f} seconds ({((end_time - start_time)/60):.2f} minutes)")
    # ------------------------------
    # Save EMA model
    # ------------------------------
    model.fusion.load_state_dict(ema_fusion.state_dict())
    model.head.load_state_dict(ema_head.state_dict())
    torch.save(model.state_dict(), "DReX.pth")


    
    print(f"\n✓ Training complete. Model saved to DReX.pth\n{'='*60}")


if __name__ == "__main__":
    main()