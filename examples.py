from models import DReX
from data import ComplexityDataset
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DReX(device=device).to(device)

# Load trained model
checkpoint = torch.load("results/DReX.pth", map_location=device)
    
# Handle different checkpoint formats
if isinstance(checkpoint, dict) and "model_state" in checkpoint:
    model.load_state_dict(checkpoint["model_state"])
else:
    model.load_state_dict(checkpoint)
model.eval()

dataset = ComplexityDataset(txt_file="examples/images.txt", zip_file="examples/images.zip")

for image, label in dataset:
    pred = model.forward_from_images([image])
    print(pred)
