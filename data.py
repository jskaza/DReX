from torch.utils.data import Dataset
from PIL import Image
import io
import zipfile
import os
import numpy as np
import torch

class ComplexityDataset(Dataset):
    """
    Dataset that reads images from a .zip file and their labels from a .txt file.
    The .txt file should have lines like:
        image_name.jpg  <two spaces>  label_value
    """

    def __init__(self, txt_file: str, zip_file: str, transform=None):
        super().__init__()
        self.zip_file = zip_file
        self.entries = []

        # Parse labels
        with open(txt_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("  ")  # two spaces between name and label
                if len(parts) != 2:
                    raise ValueError(f"Invalid line in {txt_file}: {line}")
                img_name, label_str = parts
                self.entries.append((img_name.strip(), float(label_str.strip())))

        # Open zip for reading
        self.zf = zipfile.ZipFile(zip_file, "r")
        self.zip_names = set(self.zf.namelist())

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        img_name, label = self.entries[idx]
        try:
            if "images/" + img_name not in self.zip_names and "images/" + img_name.replace(".jpg", ".png") not in self.zip_names:
                raise FileNotFoundError(f"{img_name} not found in {self.zip_file}")
            else:
                img_path = "images/" + img_name if "images/" + img_name in self.zip_names else "images/" + img_name.replace(".jpg", ".png")
                with self.zf.open(img_path) as img_file:
                    img_bytes = img_file.read()
                    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                return img, torch.tensor(label, dtype=torch.float32)
        except Exception as e:
            raise FileNotFoundError(f"{img_name} not found in {self.zip_file}")

    def __del__(self):
        # Ensure zip file is closed
        try:
            self.zf.close()
        except Exception:
            pass

