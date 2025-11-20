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

class SavoiasDataset(Dataset):
    """
    Dataset for loading SAVOIAS image complexity data from directory structure.
    Expects:
      - images under `images_dir/<Category>/...`
      - ground truth as a .npy dict at `ground_truth_npy` with entries: {idx: (filename, rank_score)}
    """
    def __init__(self, images_dir: str, ground_truth_npy: str, category_subdir: str, transform=None):
        super().__init__()
        self.transform = transform
        self.images_dir = os.path.join(images_dir, category_subdir)
        self.data = []

        # Load ground truth dict: {idx: (filename, rank_score)}
        gt_dict = np.load(ground_truth_npy, allow_pickle=True).item()

        # Build data list: (image_path, rank_score)
        for idx in sorted(gt_dict.keys()):
            filename, rank_score = gt_dict[idx]
            image_path = os.path.join(self.images_dir, filename)

            if not os.path.exists(image_path):
                base_name, _ = os.path.splitext(filename)
                for alt_ext in [".jpg", ".png", ".jpeg", ".JPG", ".PNG"]:
                    alt_path = os.path.join(self.images_dir, base_name + alt_ext)
                    if os.path.exists(alt_path):
                        image_path = alt_path
                        break

            if os.path.exists(image_path):
                self.data.append((image_path, float(rank_score)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, score = self.data[idx]
        img = Image.open(image_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(score/100.0, dtype=torch.float32)

def load_savoias_dataset(categories=None, images_dir: str = "savoias/Images", ground_truth_dir: str = "savoias/Ground truth/npy", transform=None):
    """
    Create a Dataset (or ConcatDataset) for the SAVOIAS benchmark.

    Args:
        categories: list of category names as they appear in `images_dir`.
                    If None, uses all available categories found in the folder.
        images_dir: root images directory (default: "savoias/Images")
        ground_truth_dir: directory with ground-truth npy files (default: "savoias/Ground truth/npy")
        transform: optional transform to apply to PIL images

    Returns:
        torch.utils.data.Dataset. If multiple categories are provided, a ConcatDataset is returned.
    """
    # Map folder category -> ground truth file suffix
    category_to_gt_key = {
        "Advertisement": "ad",
        "Art": "art",
        "Interior Design": "interior_design",
        "Objects": "objects",
        "Scenes": "scenes",
        "Suprematism": "sup",
        "Visualizations": "vis",
    }

    # Determine categories
    if categories is None:
        # Use present subdirectories intersected with known mapping to be robust
        try:
            present = [d for d in os.listdir(images_dir) if os.path.isdir(os.path.join(images_dir, d))]
        except FileNotFoundError:
            present = []
        categories = [c for c in category_to_gt_key.keys() if c in present]

    datasets = []
    for category in categories:
        if category not in category_to_gt_key:
            raise ValueError(f"Unknown SAVOIAS category: {category}")
        gt_key = category_to_gt_key[category]
        gt_path = os.path.join(ground_truth_dir, f"global_ranking_{gt_key}.npy")
        ds = SavoiasDataset(images_dir=images_dir, ground_truth_npy=gt_path, category_subdir=category, transform=transform)
        datasets.append(ds)

    if not datasets:
        raise RuntimeError("No SAVOIAS datasets constructed. Check directories and categories.")

    if len(datasets) == 1:
        return datasets[0]

    from torch.utils.data import ConcatDataset
    return ConcatDataset(datasets)

def create_drex_collate_fn(dino_processor, resnet_transform):
    """
    Collate function that:
      - applies DINO processor and ResNet transforms
      - returns tensors ready for DReX.forward()
    """
    def collate_fn(batch):
        images, scores = zip(*batch)
        scores = torch.stack(scores)

        # DINO pixel values (processor expects PIL images)
        dino_inputs = dino_processor(images, return_tensors="pt")["pixel_values"]

        # ResNet preprocessed tensors
        resnet_inputs = torch.stack([resnet_transform(img) for img in images])

        return dino_inputs, resnet_inputs, scores
    return collate_fn

