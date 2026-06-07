# Quantifying the Reliability of a Pneumonia-Diagnosing ResNet34 Model via GAX-based Interpretable Confidence Maps

An Explainable AI (XAI) verification framework designed to bridge the gap between high predictive accuracy and clinical trust in medical deep learning. This repository implements a pre-deployment auditing tool that detects **shortcut learning** and **simplicity bias** by evaluating whether classification models track true biological markers or spurious image background artifacts.

---

## Project Overview & Core Contribution
While deep learning models achieve outstanding statistical performance on medical benchmarks, their deployment introduces critical safety concerns due to the **black-box problem**. High accuracy scores often mask unstable reasoning pathways, where models anchor predictions onto equipment borders, text annotations, or peripheral artifacts rather than the actual pathology.

This project extends the **Generative Augmentative Explanation (GAX)** paradigm from a purely descriptive visual tool into an **automated quantitative audit pipeline** using a formalized **Cheating Score ($S_{cheat}$)**. 

### Key Features
* **Backbone Architecture:** Fine-tuned variations of a 34-layer Convolutional Neural Network (**ResNet34**) optimized using ImageNet-pretrained initialization weights.
* **Functional Explanations:** Employs an iterative optimization generator loop to dynamically scale attribution patterns to maximize prediction confidence.
* **Output-Stage Anatomical Segmentation:** Integrates a deep-learning segmentation model to isolate precise biological regions of interest (ROI) at the output stage for visualization and evaluation.
* **Reliability Metrics:** Implements a localized tracking layer measuring the **Cheating Rate (CR)** and **Average Cheating Score** to flag unsafe models prior to clinical deployment.

---

## Methodological Framework

### Phase 1: Dataset Engineering
* **Data Sourcing:** Grayscale Chest X-ray (CXR) imagery from the RSNA Pneumonia Detection Challenge.
* **Clinical Filtering:** The original distribution (26,684 files) was converted into a rigorous binary cohort by dropping the ambiguous "No Lung Opacity / Not Normal" category. 
    * **Final Dataset:** 14,863 unique images (8,851 Normal, 6,012 Pneumonia).
* **Partitioning & Preprocessing:** Data was sorted and split via an 80:10:10 ratio (Train: 11,890 | Val: 1,486 | Test: 1,487). Images were converted from DICOM to JPEG to optimize standard vision pipeline I/O bottlenecks.

### Phase 2: Backbone Training
Eight variations of the **ResNet34** architecture were trained to evaluate the impact of different hyperparameter configurations (learning rate, momentum, weight decay) on both performance and reasoning reliability. Training was hardware-accelerated using an NVIDIA Tesla T4 GPU.

### Phase 3: Quantitative Safety Evaluation
1. **GAX Paradigm Execution:** Optimizes an attribution mask to maximize the classifier's predictive confidence. A Confidence Optimization (CO) score of ~0.9 signals highly optimized, positive alignment.
2. **Anatomical Ground Truth Segmentation:** Isolates the true biological lung fields. Medically irrelevant extra-pulmonary regions are explicitly mapped.
3. **Cheating Score ($S_{cheat}$) Quantification:** Computes the mathematical proportion of positive model attribution falling completely outside the segmented lung mask.

$$S_{cheat} = \frac{\sum_{i=1}^{N_{p}}(H_{i}^{+} \odot R_{irr,i})}{\sum_{i=1}^{N_{p}}H_{i}^{+}}$$

---

## Experimental Results & The Performance-Reliability Paradox

Despite exceptional baseline classification scores across all eight trained architectures (achieving test accuracies up to 93.14%), the quantitative reliability audit revealed a severe structural reliance on non-clinical signals.

* **The Paradox Confirmed:** Correlation analysis tracked a strong positive relationship between baseline model accuracy and average cheating values. As models became more statistically accurate, they relied more heavily on medically irrelevant thoracic background markers.
* **Systemic Failure:** Applying the definitive safety threshold ($S_{cheat} > 0.50$) flagged the vast majority of predictions as exhibiting shortcut-learning behavior. 
* **Observation:** Out of 100 tested pneumonia samples, 96 images exhibited a cheating score greater than 50%, completely invalidating their clinical readiness despite "correct" predictions.

---

## Execution Pipeline & Usage

### Installation
Ensure you are running Python 3.10+ alongside a CUDA-supported execution frame:

```bash
git clone [https://github.com/Vanz15/gax-safety.git](https://github.com/Vanz15/gax-safety.git)
cd gax-safety
pip install -r requirements.txt
```

### Evaluation Workflow
The core scripts accept command-line arguments, making it easy to test different models and keep results isolated. *(Note: Running scripts without arguments defaults to evaluating `best_resnet34_v3.pth`).*

**1. Generate GAX Heatmaps** Run the Generative Attribution eXplanation optimization to create raw heatmap `.npy` files.

```bash
python generate_gax.py \
  --model_path checkpoints/best_resnet34_v4.pth \
  --output_dir results/resnet34_v4/gax_images
```

**2. Compute Cheating Scores** Calculate the $S_{cheat}$ metric by analyzing spatial attribution against ground-truth lung masks.

```bash
python compute_cheating_score.py \
  --model resnet34_v4 \
  --gax_dir results/resnet34_v4/gax_images \
  --output_csv results/resnet34_v4/cheating_scores.csv
```

**3. Visualize the Results** Generate 3-panel visual proofs (Original X-Ray, Segmented Lung Mask, GAX Heatmap Overlay) to manually verify shortcut learning severity.

```bash
python visualize_cheating.py \
  --csv_path results/resnet34_v4/cheating_scores.csv \
  --gax_dir results/resnet34_v4/gax_images \
  --output_dir results/resnet34_v4/visualizations
```

---

## 🎓 Academic Context
This framework was developed as part of an Undergraduate Special Problem for the degree of **Bachelor of Science in Computer Science** at **The University of the Philippines Baguio**.
* **Authors:** Aivann Herald P. Martinez & Aleja Jeremiah V. Talaue
* **Adviser:** Joseph Ludwin D.C. Marigmen, M. Sc.
