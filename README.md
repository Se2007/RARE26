# RARE26 — Early Cancer Detection in Low-Prevalence Settings

A PyTorch-based solution for the [RARE26 Grand Challenge](https://rare26.grand-challenge.org/), focused on automated detection of early-stage neoplasia in Barrett's Esophagus (BE) endoscopy images - a setting where positive cases represent less than 1% of clinical data.

---

## 🩺 Problem Statement

Early-stage esophageal cancer in BE patients is extremely rare and visually subtle. Standard classification models trained on balanced datasets fail in real clinical settings where the class imbalance is severe. This project addresses that gap by combining imbalance-aware loss functions, proper evaluation metrics, and a ResNet-50 backbone fine-tuned for binary anomaly detection.

---

## 🏗️ Architecture

- **Backbone:** ResNet-50 (pretrained on ImageNet), fine-tuned for binary classification
- **Training:** k-fold cross-validation with SGD optimizer (momentum=0.9, Nesterov optional)
- **Experiment tracking:** Weights & Biases (W&B) — ROC curve, PR curve, confusion matrix

---

## ⚖️ Loss Functions (`loss.py`)

Two custom losses designed for severe class imbalance:

- **`WeightedFocalLoss`**  
- **`HybridBCEMedicalLoss`** 

---

## 📊 Evaluation Metrics (`metrics.py`)

| Metric | Description |
|--------|-------------|
| **AUROC** | Area under ROC curve |
| **AUPRC** | Area under Precision-Recall curve |
| **PPV@90% Recall** | Positive Predictive Value at ≥90% sensitivity (challenge metric) |
| Accuracy / Sensitivity / Specificity | At fixed threshold (0.5) |
| T90 Threshold | Decision threshold achieving 90% recall |

> The challenge's key metric is **PPV@90% Recall**, which directly reflects clinical utility in low-prevalence settings.

---


## 🚀 Quick Start

### Training

Open `train.ipynb` and run all cells. The notebook handles:
- Data loading from `./Datasets/RARE25-train-data`
- k-fold cross-validation
- W&B logging of metrics and curves
- Checkpoint saving

### Hyperparameter Search

```bash
python HP.py
```

### Evaluation

Open `evaluate.ipynb` with a saved checkpoint path.

---

## 📦 Dependencies

```
torch
torchvision
torchmetrics
scikit-learn
numpy
matplotlib
wandb
tqdm
prettytable
colorama
```

---

## 🔗 Links

- **Challenge page:** [rare26.grand-challenge.org](https://rare26.grand-challenge.org/)
