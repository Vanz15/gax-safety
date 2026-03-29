import os
from glob import glob

import numpy as np
import pydicom
from PIL import Image

INPUT_ROOT = "dicom_dataset"       # has train/val/test
OUTPUT_ROOT = "jpeg_dataset" # new root with JPEGs

CLASSES = ["normal", "pneumonia"]
SPLITS = ["train", "val", "test"]


def dicom_to_rgb_image(path: str) -> Image.Image:
    ds = pydicom.dcmread(path)
    arr = ds.pixel_array.astype(np.float32)

    # min-max normalize to [0, 1]
    min_val, max_val = arr.min(), arr.max()
    if max_val > min_val:
        arr = (arr - min_val) / (max_val - min_val)
    else:
        arr = np.zeros_like(arr, dtype=np.float32)

    # scale to [0, 255] uint8
    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)

    img = Image.fromarray(arr)    # "L"
    img = img.convert("RGB")      # 3-channel
    return img


def main():
    for split in SPLITS:
        for cls in CLASSES:
            in_dir = os.path.join(INPUT_ROOT, split, cls)
            out_dir = os.path.join(OUTPUT_ROOT, split, cls)
            os.makedirs(out_dir, exist_ok=True)

            pattern = os.path.join(in_dir, "*.dcm")
            for dcm_path in glob(pattern):
                fname = os.path.splitext(os.path.basename(dcm_path))[0] + ".jpg"
                out_path = os.path.join(out_dir, fname)

                if os.path.exists(out_path):
                    continue  # already converted

                img = dicom_to_rgb_image(dcm_path)
                img.save(out_path, format="JPEG", quality=95)


if __name__ == "__main__":
    main()