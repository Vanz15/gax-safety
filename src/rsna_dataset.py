# this class is to tell PyTorch how to read the dataset
import os
from glob import glob
from typing import List, Tuple

import numpy as np
import pydicom
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

CLASSES = ["normal", "pneumonia"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASSES)}


def load_dicom_as_pil(path: str) -> Image.Image:
    """
    Load a DICOM file and return a 3‑channel PIL image.
    """
    ds = pydicom.dcmread(path)
    pixel_array = ds.pixel_array.astype(np.float32)

    # Simple normalization to [0, 1] based on min/max in the image
    min_val = pixel_array.min()
    max_val = pixel_array.max()
    if max_val > min_val:
        pixel_array = (pixel_array - min_val) / (max_val - min_val)
    else:
        pixel_array = np.zeros_like(pixel_array, dtype=np.float32)

    # Convert to 0‑255 uint8
    pixel_array = (pixel_array * 255.0).clip(0, 255).astype(np.uint8)

    # Create single‑channel PIL image
    img = Image.fromarray(pixel_array)  # mode "L"

    # Convert to 3‑channel (RGB) by duplicating the channel
    img = img.convert("RGB")
    return img


class DicomRSNADataset(Dataset):
    def __init__(self, root_dir: str, split: str, transform=None):
        """
        root_dir: path to 'dataset' (which has train/val/test)
        split: 'train', 'val', or 'test'
        transform: torchvision transforms to apply to each image
        """
        assert split in {"train", "val", "test"}, f"Invalid split: {split}"
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        self.samples: List[Tuple[str, int]] = []
        split_dir = os.path.join(root_dir, split)
        for class_name in CLASSES:
            class_dir = os.path.join(split_dir, class_name)
            # All .dcm files in this class directory
            pattern = os.path.join(class_dir, "*.dcm")
            for path in glob(pattern):
                label_idx = CLASS_TO_IDX[class_name]
                self.samples.append((path, label_idx))

        if not self.samples:
            raise RuntimeError(f"No DICOM files found under {split_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]

        img = load_dicom_as_pil(path)

        if self.transform is not None:
            img = self.transform(img)

        # label as integer (0 for normal, 1 for pneumonia)
        label = torch.tensor(label, dtype=torch.long)
        return img, label

class JpegRSNADataset(Dataset):
    def __init__(self, root_dir: str, split: str, transform=None):
        assert split in {"train", "val", "test"}
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        self.samples: List[Tuple[str, int]] = []
        split_dir = os.path.join(root_dir, split)
        for class_name in CLASSES:
            class_dir = os.path.join(split_dir, class_name)
            pattern = os.path.join(class_dir, "*.jpg")
            for path in glob(pattern):
                label_idx = CLASS_TO_IDX[class_name]
                self.samples.append((path, label_idx))

        if not self.samples:
            raise RuntimeError(f"No JPEG files found under {split_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        img = Image.open(path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        label_tensor = torch.tensor(label, dtype=torch.long)
        return img, label_tensor

# ImageNet normalization values
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transforms = T.Compose([
    T.Resize(256),
    T.RandomResizedCrop(224, scale=(0.8, 1.0)),
    T.RandomHorizontalFlip(p=0.5),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

val_transforms = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])