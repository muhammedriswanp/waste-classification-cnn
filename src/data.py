import torch
from torchvision import transforms, datasets
from torch.utils.data import random_split, DataLoader
from src.config import *
def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform

def get_dataloaders(data_dir, val_split=0.2, batch_size=BATCH_SIZE, seed=42):
    train_transform, val_transform = get_transforms()

    full_dataset = datasets.ImageFolder(data_dir, transform=train_transform)

    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=generator
        )
    
    val_dataset.dataset.transform = val_transform       #change the original dataset transform into validation transform

    train_loader  = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    class_name = full_dataset.classes

    return train_loader, val_loader, class_name, train_size, val_size

if __name__ == "__main__":
    train_loader, val_loader, class_names, train_size, val_size = get_dataloaders("data")

    print(f"Classes   : {class_names}")
    print(f"Train size: {train_size}")
    print(f"Val size  : {val_size}")

    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")   # expect [32, 3, 128, 128]
    print(f"Labels     : {labels}")
