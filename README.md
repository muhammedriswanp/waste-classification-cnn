# Waste Classification CNN

## Dataset
- 4650 images, 6 classes: battery, glass, metal, organic, paper, plastic
- Train: 3720 | Val: 930

## Results
| Epoch | Train Acc | Val Acc | Val Loss |
|-------|-----------|---------|----------|
| 01/10 | 45.62% | 64.19% | 0.9523 |
| 02/10 | 55.46% | 64.09% | 0.9265 |
| 03/10 | 56.72% | 64.73% | 0.9310 |
| 04/10 | 57.18% | 65.81% | 0.8700 |
| 05/10 | 59.01% | 66.45% | 0.8621 |
| 06/10 | 59.57% | 66.24% | 0.8822 |
| 07/10 | 60.94% | 68.82% | 0.8443 |
| 08/10 | 61.94% | 67.96% | 0.8073 |
| **09/10** | **63.25%** | **73.87%** | **0.7752** |
| 10/10 | 62.90% | 64.30% | 0.8880 |

**Best:** Val Acc 73.87% | Val Loss 0.7752 (Epoch 09)

## Quick Start
```bash
pip install -r requirements.txt
python train.py
```

# src/evaluate.py

import torch
import mlflow
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from src.config import *
from src.data import get_dataloaders
from src.model import WasteCNN


def evaluate(model_path):
    _, val_loader, class_names, _, val_size = get_dataloaders(DATA_DIR)

    model = WasteCNN().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            output = model(images)
            preds  = output.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    # --- Classification report ---
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print(report)

    # --- Confusion matrix ---
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    plt.title("Confusion Matrix — Waste CNN")
    plt.tight_layout()

    cm_path = "evaluation/confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved → {cm_path}")

    # --- Log to MLflow ---
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="evaluation"):
        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(model_path)
        print("Logged to MLflow")


if __name__ == "__main__":
    import sys
    # Usage: python -m src.evaluate models/best_model_XXXXXXXX.pth
    if len(sys.argv) < 2:
        # Auto-pick latest model if no path given
        import os
        models = sorted(os.listdir("models"))
        if not models:
            print("No models found in models/")
            sys.exit(1)
        model_path = f"models/{models[-1]}"
        print(f"Auto-selected: {model_path}")
    else:
        model_path = sys.argv[1]

    evaluate(model_path)