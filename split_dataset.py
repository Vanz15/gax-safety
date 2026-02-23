import os
import random
import shutil

# Base directory where your current folders live
BASE_DIR = "dataset"
CLASSES = ["normal", "pneumonia"]

# Split ratios
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1  # test gets the remaining 0.1

random.seed(42)  # for reproducibility


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def main():
    for cls in CLASSES:
        src_dir = os.path.join(BASE_DIR, cls)
        # Destination dirs: dataset/train/cls, dataset/val/cls, dataset/test/cls
        train_dir = os.path.join(BASE_DIR, "train", cls)
        val_dir = os.path.join(BASE_DIR, "val", cls)
        test_dir = os.path.join(BASE_DIR, "test", cls)

        ensure_dir(train_dir)
        ensure_dir(val_dir)
        ensure_dir(test_dir)

        # List all DICOM files for this class
        files = [
            f for f in os.listdir(src_dir)
            if os.path.isfile(os.path.join(src_dir, f)) and f.lower().endswith(".dcm")
        ]

        random.shuffle(files)
        n_total = len(files)
        n_train = int(n_total * TRAIN_RATIO)
        n_val = int(n_total * VAL_RATIO)
        n_test = n_total - n_train - n_val

        train_files = files[:n_train]
        val_files = files[n_train:n_train + n_val]
        test_files = files[n_train + n_val:]

        print(f"{cls}: total={n_total}, "
              f"train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")

        # Move files into splits (change to shutil.copy2 if you prefer copying)
        for fname, dst_root in (
            *[(f, train_dir) for f in train_files],
            *[(f, val_dir) for f in val_files],
            *[(f, test_dir) for f in test_files],
        ):
            src_path = os.path.join(src_dir, fname)
            dst_path = os.path.join(dst_root, fname)

            # Skip if already moved (useful if you re-run)
            if not os.path.exists(src_path):
                continue

            shutil.move(src_path, dst_path)


if __name__ == "__main__":
    main()