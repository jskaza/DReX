from models import DReX
from ic9600.ICNet import ICNet
from d2s.D2S import D2S_R18, D2S_R50

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_all_parameters(model):
    return sum(p.numel() for p in model.parameters())

# DReX overall
drex_model = DReX()
print(f"DReX: {count_parameters(drex_model)/1e6:.2f}M parameters (trainable)")

# DReX module breakdown
print(f"\nDReX Module Breakdown:")
print(f"  DINO (frozen): {count_all_parameters(drex_model.dino_model)/1e6:.2f}M parameters")
print(f"  ResNet extraction (frozen): {count_all_parameters(drex_model.resnet_backbone)/1e6:.2f}M parameters")
print(f"  Attention fusion: {count_parameters(drex_model.fusion)/1e6:.2f}M parameters")
print(f"  MLP head: {count_parameters(drex_model.head)/1e6:.2f}M parameters")

# Other models
print(f"\nICNet: {count_parameters(ICNet())/1e6:.2f}M parameters")
print(f"D2S_R18: {count_parameters(D2S_R18())/1e6:.2f}M parameters")
print(f"D2S_R50: {count_parameters(D2S_R50())/1e6:.2f}M parameters")