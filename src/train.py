import mlflow
import torch
import torch.nn as nn
from datetime import datetime 
import os
import mlflow.pytorch

from src.data import *
from src.config import *
from src.model import WasteCNN

def train():
    train_loader, val_loader, class_names, train_size, val_size = get_dataloaders("data")
    print(f"Train: {train_size} | Val: {val_size} | Classes: {class_names}")

    model = WasteCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    mlflow.set_experiment(EXPERIMENT_NAME)

    os.makedirs("models", exist_ok=True)

    best_acc = float('-inf')
    patience_counter = 0

    with mlflow.start_run(run_name=f"wasteCNN_lr{LR}_dropout{DROPOUT}"):

        mlflow.log_params({
            "epochs":        EPOCHS,
            "learning_rate": LR,
            "dropout":       DROPOUT
        })

        for epoch in range(EPOCHS):

            model.train()
            running_loss, correct = 0, 0

            for images, labels in train_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                output = model(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                correct += (output.argmax(1) == labels).sum().item()

            train_acc = correct / train_size * 100
            train_loss = running_loss / len(train_loader)

            model.eval()
            val_loss_total, val_correct = 0, 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(DEVICE), labels.to(DEVICE)
                    output = model(images)
                    loss = criterion(output, labels)
                    val_loss_total += loss.item()
                    val_correct    += (output.argmax(1) == labels).sum().item()

            val_acc  = val_correct / val_size * 100
            val_loss = val_loss_total / len(val_loader)

            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_acc":  train_acc,
                "val_loss":   val_loss,
                "val_acc":    val_acc,
            }, step=epoch)

            print(f"Epoch {epoch+1:02d}/{EPOCHS} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            if val_acc > best_acc:
                best_acc = val_acc
                patience_counter = 0
                torch.save(model.state_dict(), "models/best_model.pth")
                print(f"New best model saved! Val Acc: {best_acc:.2f}%")
            else:
                patience_counter += 1
            
            if patience_counter >= PATIENCE:
                print(f"\n  Early stopping at epoch {epoch+1}")
                break

        model.load_state_dict(torch.load("models/best_model.pth"))
        mlflow.pytorch.log_model(model, name="best_model")
        mlflow.log_metric("best_val_accuracy",best_acc)

        print("Train Completed")
if __name__ == "__main__":
    train()