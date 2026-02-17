# MammoFusionNet

**Study-Level Multimodal Mammography Classification with Structured Fusion**

---

## Overview

MammoFusionNet is a deep learning framework for **study-level mammography classification (normal vs abnormal)** using:

- Multi-view mammogram images  
- Clinical metadata  
- Attention-based aggregation  
- Structured breast-aware fusion  

The project progresses from strong baselines to an advanced structured multimodal model (FVNet) using EfficientNet backbones.

---

## Problem Statement

Given a mammography study containing multiple views (e.g., CC and MLO for left and right breasts), predict whether the study is:

- **Normal**  
- **Abnormal**  

The system operates at **study level**, not image level, which better reflects clinical decision workflows.

---

## Repository Structure

```

├── Data_Preprocessing.ipynb
├── Baseline.ipynb
├── Baseline_Attention.ipynb
├── Baseline_Metadata.ipynb
├── Final_Train.ipynb
└── README.md

```

---

## Methodology

### 1. Data Preprocessing

- Study-level grouping of mammograms  
- Metadata cleaning and alignment  
- Class balance verification  
- Split validation  
- Missing file detection  
- CSV correction  

**Outputs:**

- Cleaned image CSV  
- Study-level metadata CSV  

---

### 2. Image-Only Baseline

**Backbone:**  

- ResNet-50 (pretrained)  

**Aggregation:**  

- Mean pooling across views  

**Loss:**  

- Binary Cross Entropy (BCEWithLogitsLoss)  

**Metrics:**  

- ROC-AUC  
- PR-AUC  
- Accuracy  
- F1-score  
- Brier score  

---

### 3. Attention-Based Model

**Improvement over baseline:**  

- Learns attention weights over views  
- Identifies most informative mammogram projections  
- Weighted feature aggregation  

**Purpose:**  
Evaluate whether view-aware aggregation improves performance.

---

### 4. Metadata Baseline

**Models:**  

- Logistic Regression  
- XGBoost  

**Features:**  

- Age  
- Breast density  
- Other clinical variables  

**Purpose:**  
Measure predictive strength of metadata alone.

---

### 5. Multimodal Early Fusion

**Architecture:**  

- ResNet backbone for images  
- MLP for metadata  
- Feature concatenation  
- Joint classifier  

This evaluates whether combining modalities improves prediction.

---

### 6. Final Model — FVNet (Structured Fusion)

**Backbone:**  

- EfficientNet  

**Architecture:**  

- Left breast branch  
- Right breast branch  
- Study-level aggregation  
- Multimodal fusion  

**Outputs:**  

- Study prediction  
- Left breast prediction  
- Right breast prediction  
- Feature embeddings  

**Loss:**  

- Multi-task objective  
- Study-level BCE  
- Auxiliary breast-level supervision  

This model enforces anatomical consistency and structured learning.

---

## Key Contributions

- Study-level multi-view aggregation  
- Attention-based view weighting  
- Metadata-only benchmark  
- Early multimodal fusion  
- Structured breast-aware fusion (FVNet)  
- Multi-task learning setup  
- Comprehensive evaluation metrics  

---

## Training Details

**Optimizer:**  

- AdamW  

**Loss:**  

- BCEWithLogitsLoss  
- Custom FVNet multi-task loss  

**Evaluation:**  

- ROC curves  
- Precision-Recall curves  
- Confusion matrix  
- Validation checkpointing  

---

## Metrics Reported

- ROC-AUC  
- Average Precision (PR-AUC)  
- Accuracy  
- F1-score  
- Brier score  

---

## Dataset Assumptions

**Directory format:**

```

birads_preprocessed_dataset/
training/
normal/
abnormal/
test/
normal/
abnormal/

```

Each study contains multiple mammogram views.

---

## Reproducibility

To reproduce experiments:

1. Run `Data_Preprocessing.ipynb`  
2. Train image baseline (`Baseline.ipynb`)  
3. Train attention model (`Baseline_Attention.ipynb`)  
4. Train metadata + multimodal models (`Baseline_Metadata.ipynb`)  
5. Train final structured FVNet (`Final_Train.ipynb`)  

---

## Future Work

- Cross-validation  
- Grad-CAM interpretability  
- External dataset validation  
- Self-supervised pretraining  
- Uncertainty estimation  
- Clinical calibration analysis  

---

## Potential Applications

- Computer-Aided Detection (CAD)  
- Screening prioritization  
- Risk stratification support systems  

---


## Citation

If used in academic work, please cite:

```

Author Name. Yash Jaiswal
MammoFusionNet: Study-Level Multimodal Mammography Classification with Structured Fusion.
Year. 2025

```

