import pandas as pd

# Load detailed class info (Normal, Lung Opacity, No Lung Opacity / Not Normal)
df = pd.read_csv("stage_2_detailed_class_info.csv")

def assign_label(classes):
    classes = set(classes)
    if "Lung Opacity" in classes:
        return "pneumonia"
    if classes == {"Normal"}:
        return "normal"
    # No Lung Opacity / Not Normal → drop
    return None

# One row per patient; drop "No Lung Opacity / Not Normal" patients
binary_labels = (
    df.groupby("patientId")["class"]
    .apply(lambda x: assign_label(x.values))
    .reset_index()
)
binary_labels.columns = ["patientId", "label"]
binary_labels = binary_labels[binary_labels["label"].notna()]

binary_labels.to_csv("rsna_binary_labels.csv", index=False)

print(binary_labels["label"].value_counts())

import os
import shutil
import pandas as pd

labels = pd.read_csv("rsna_binary_labels.csv")

image_dir = "stage_2_train_images"
output_dir = "dataset"

os.makedirs(f"{output_dir}/pneumonia", exist_ok=True)
os.makedirs(f"{output_dir}/normal", exist_ok=True)

for _, row in labels.iterrows():
    pid = row["patientId"]
    label = row["label"]

    src = f"{image_dir}/{pid}.dcm"
    dst = f"{output_dir}/{label}/{pid}.dcm"

    # Skip if destination file already exists (saves time on reruns)
    if not os.path.exists(src):
        continue
    if os.path.exists(dst):
        continue

    shutil.copy(src, dst)
