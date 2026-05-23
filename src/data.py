import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
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

    train_dataset = datasets.ImageFolder(data_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(data_dir, transform=val_transform)

    total_size = len(train_dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total_size, generator=generator).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    train_loader = DataLoader(train_subset, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(val_subset, shuffle=False, batch_size=batch_size)
    class_names = train_dataset.classes
    return train_loader, val_loader, class_names, train_size, val_size


if __name__ == "__main__":
    train_loader, val_loader, class_names, train_size, val_size = get_dataloaders("data")

    print(f"Classes   : {class_names}")
    print(f"Train size: {train_size}")
    print(f"Val size  : {val_size}")

    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")   
    print(f"Labels     : {labels}")
