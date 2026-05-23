import torch
import numpy as np
import os
import mlflow
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import src.data as data
from src.model  import WasteCNN
import src.config as cfg 

def evaluate(model_path):
    _, val_loader, class_names, _, _ = data.get_dataloaders("data")

    model = WasteCNN().to(cfg.DEVICE)
    model.load_state_dict(torch.load(model_path))

    predicted, actual_labels = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(cfg.DEVICE)
            output = model(images)
            preds = output.argmax(1).cpu().numpy()  #NumPy works only on CPU memory.
            predicted.extend(preds)
            actual_labels.extend(labels.numpy())

    report = classification_report(actual_labels, predicted, target_names=class_names)
    print(report)

    cm = confusion_matrix(actual_labels, predicted)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues")
    plt.title("Confusion Matrix — Waste CNN")
    plt.tight_layout()

    os.makedirs("evaluation", exist_ok=True)
    cm_path = "evaluation/confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()

    mlflow.set_experiment(cfg.EXPERIMENT_NAME)
    with mlflow.start_run(run_name=cfg.RUN_NAME):
        mlflow.log_artifact(cm_path)
        print("Logged to MLflow")


if __name__ == "__main__":

    model_path = "models/best_model.pth"

    if not os.path.exists(model_path):
        print("Best model not found.")
    else:
        evaluate(model_path)