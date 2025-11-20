import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModel
from utils import get_hf_token
import torchvision



# ============================================================
# Multi-Scale ResNet Backbone
# ============================================================
class MultiScaleResNet(nn.Module):
    def __init__(self, freeze=True):
        super().__init__()
        resnet = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)

        # Split ResNet into stages
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        if freeze:
            for p in self.parameters():
                p.requires_grad = False

    def get_output_dim(self):
        with torch.no_grad():
            device = next(self.parameters()).device
            dummy = torch.randn(1, 3, 224, 224, device=device)
            out = self.forward(dummy)
            return out.shape[-1]

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)

        feats = []
        for f in [f1, f2, f3, f4]:
            pooled = F.adaptive_avg_pool2d(f, (1, 1))
            feats.append(pooled.squeeze(-1).squeeze(-1))
        return torch.cat(feats, dim=-1)


# ============================================================
# Attention-Based Fusion
# ============================================================
class AttentionFusion(nn.Module):
    def __init__(self, dino_dim, resnet_dim, hidden_dim=384):
        super().__init__()
        self.dino_proj = nn.Sequential(
            nn.Linear(dino_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        self.resnet_proj = nn.Sequential(
            nn.Linear(resnet_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.Tanh(),
            nn.Linear(128, 2)
        )
        self.temperature = nn.Parameter(torch.tensor(0.7))
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, dino_feat, resnet_feat):
        d_proj = self.dino_proj(dino_feat)
        r_proj = self.resnet_proj(resnet_feat)
        combined = torch.cat([d_proj, r_proj], dim=-1)
        weights = F.softmax(self.attention(combined) / self.temperature, dim=-1)
        fused = weights[:, 0:1] * d_proj + weights[:, 1:2] * r_proj
        fused = fused + self.alpha * (d_proj + r_proj)
        fused = F.layer_norm(fused, fused.shape[-1:])
        return fused, weights


# ============================================================
# DReX Main Model
# ============================================================
class DReX(nn.Module):
    def __init__(
        self,
        dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        fusion_dim=384,
        device=None
    ):
        super().__init__()
        self.device = device if device is not None else torch.device("cpu")
    
        # DINO
        self.dino_processor = AutoImageProcessor.from_pretrained(
            dino_model_name, token=get_hf_token()
        )
        self.dino_model = AutoModel.from_pretrained(
            dino_model_name, token=get_hf_token()
        ).eval()
        for p in self.dino_model.parameters():
            p.requires_grad = False
        self.dino_embed_dim = self.dino_model.config.hidden_size

        # ResNet backbone
        self.resnet_backbone = MultiScaleResNet(freeze=True).eval()
        self.resnet_embed_dim = self.resnet_backbone.get_output_dim()

        # ResNet preprocessing
        self.resnet_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # Fusion + Head
        self.fusion = AttentionFusion(
            self.dino_embed_dim,
            self.resnet_embed_dim,
            hidden_dim=fusion_dim
        )
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )

    # -------------------------------
    # Feature extraction helpers
    # -------------------------------
    def extract_dino_features(self, images):
        pixel_values = self.dino_processor(images, return_tensors="pt")['pixel_values'].to(self.device)
        with torch.no_grad():
            dino_out = self.dino_model(pixel_values)
            return dino_out.last_hidden_state[:, 0]

    def extract_resnet_features(self, images):
        resnet_imgs = torch.stack([self.resnet_transform(img) for img in images]).to(self.device)
        with torch.no_grad():
            return self.resnet_backbone(resnet_imgs)

    # -------------------------------
    # Forward variants
    # -------------------------------
    def forward_from_images(self, images):
        """Run full forward pass starting from PIL images."""
        dino_embeddings = self.extract_dino_features(images).to(dtype=torch.float32)
        resnet_features = self.extract_resnet_features(images).to(dtype=torch.float32)
        return self.forward_from_embeddings(dino_embeddings, resnet_features)

    def forward_from_embeddings(self, dino_embeddings, resnet_features):
        """Run forward pass from precomputed embeddings."""
        fused, _ = self.fusion(dino_embeddings, resnet_features)
        preds = self.head(fused).squeeze(-1)
        return preds
    
    