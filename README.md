# Medical AI Safety - Shortcut Learning Evaluation

This repository contains a pipeline for generating attributions and scoring them to evaluate if a medical imaging model is taking shortcuts (i.e., looking at background artifacts instead of the lungs).

## Methodology

### Phase One: Dataset Preparation

1. **Data Gathering**
   - **Dataset Used:** [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/data)
   - **Files:** 26,684 original files referenced from `stage_2_detailed_class_info.csv`.
     - 11,821 No Lung Opacity / Not Normal
     - 8,851 Normal
     - 6,012 Lung Opacity

2. **Dataset Filtering & Sorting**
   - The dataset was filtered for binary classification (Normal vs. Pneumonia). Images under "No Lung Opacity / Not Normal" were dropped.
   - **Final Dataset:** 14,863 Unique Chest X-ray Images (8,851 Normal, 6,012 Lung Opacity).
   - The DICOM images were sorted into respective class folders using `sort.py`.
   - The dataset was split into train, val, and test subsets (80:10:10 split) using `split_dataset.py`:
     - **TRAIN (11,890):** NORMAL: 7,081 | PNEUMONIA: 4,809
     - **VAL (1,486):** NORMAL: 885 | PNEUMONIA: 601
     - **TEST (1,487):** NORMAL: 885 | PNEUMONIA: 602

3. **Dataset Preprocessing**
   - `rsna_dataset.py` was created to allow PyTorch to easily load the dataset. It also applies standard preprocessing transforms (resizing, cropping, flipping, and normalization).

4. **Conversion from DICOM to JPEG**
   - Since JPEG is faster and easier to work with for standard vision pipelines, the images were converted from DICOM format using `convert_dicom_to_jpeg.py`.

### Phase Two: Model Training

- Model training was primarily conducted via Google Colab using a T4 GPU.
- The main architecture evaluated is **ResNet34**. Multiple variations of this model were trained for comparison.
- The notebook and codes used for training can be found here: Google Colab Training Link

### Phase Three: Safety-Evaluation

1. **Applying GAX Process**
   - *Optional:* Applying Deconvolution and computing XAI CO Score.
   - Applying the GAX Method.
   - Computing GAX CO Scores.
   - Saving GAX heatmaps.
2. **Applying Lung Segmentation on the Test Dataset**
   - Using a deep-learning segmentation model to isolate lung pixels from the background.
3. **Computing the Cheating Score for Each Image**
   - Measuring how much of the model's positive attribution falls outside the lung mask.
4. **Computing the Average Cheating Score**
5. **Classifying Model Safety**
   - **Current Threshold:** > 50% cheating score indicates shortcut learning.
   - *Observation:* Out of 100 tested pneumonia samples, 96 images exhibited a cheating score > 50%.

---

## How to Run the Evaluation Pipeline

The core scripts accept command-line arguments, making it easy to test different models and keep their results organized in separate folders.

### 1. Generate GAX Heatmaps
Run the Generative Attribution eXplanation (GAX) optimization for a specific model. This creates the raw heatmap `.npy` files.

```bash
python generate_gax.py --model_path checkpoints/best_resnet34_v4.pth --output_dir results/resnet34_v4/gax_images
```

### 2. Compute Cheating Scores
Calculate the shortcut learning "cheating score" by analyzing how much attribution falls outside the ground truth lung masks.

```bash
python compute_cheating_score.py --gax_dir results/resnet34_v4/gax_images --output_csv results/resnet34_v4/cheating_scores.csv
```

### 3. Visualize the Results
Generate 3-panel visual proofs for the worst and best-scoring images to manually verify the shortcut learning.

```bash
python visualize_cheating.py --csv_path results/resnet34_v4/cheating_scores.csv --gax_dir results/resnet34_v4/gax_images --output_dir results/resnet34_v4/visualizations
```

*Note: If you run any of these scripts without arguments, they will default to evaluating `best_resnet34_v3.pth` and saving to the `results/resnet34_v3/` directory.*
