# === AI504 Project 1: Single-file trainer that saves raw logits ===
# Rename this file to your exact student ID, e.g., 202412345.py
# Usage examples:
#   python 202412345.py --student_id 202412345 --dataset fashion_mnist -- epochs 8
#   python 202412345.py --student_id 202412345 --dataset cifar10 --epochs 10

import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms, models

# --------------------- Utils ---------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# ----------------- Data & Transforms -----------------
def build_transforms(dataset: str, image_size: int, grayscale_to_rgb: bool):
    # ImageNet normalization tends to work well with pretrained backbones
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std  = (0.229, 0.224, 0.225)

    to_rgb = []
    if grayscale_to_rgb and dataset in {"fashion_mnist", "mnist"}:
        # Convert 1-channel images to 3-channel for pretrained CNNs
        to_rgb = [transforms.Grayscale(num_output_channels=3)]

    train_tf = transforms.Compose(
        to_rgb + [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ]
    )
    test_tf = transforms.Compose(
        to_rgb + [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ]
    )
    return train_tf, test_tf

def load_dataset(name: str, data_dir: str, train_tf, test_tf):
    name = name.lower()
    if name == "fashion_mnist":
        train = datasets.FashionMNIST(root=data_dir, train=True,  download=True, transform=train_tf)
        test  = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=test_tf)
        num_classes = 10
    elif name == "mnist":
        train = datasets.MNIST(root=data_dir, train=True,  download=True, transform=train_tf)
        test  = datasets.MNIST(root=data_dir, train=False, download=True, transform=test_tf)
        num_classes = 10
    elif name == "cifar10":
        train = datasets.CIFAR10(root=data_dir, train=True,  download=True, transform=train_tf)
        test  = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_tf)
        num_classes = 10
    elif name == "cifar100":
        train = datasets.CIFAR100(root=data_dir, train=True,  download=True, transform=train_tf)
        test  = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=test_tf)
        num_classes = 100
    else:
        raise ValueError(f"Unsupported dataset: {name}")
    return train, test, num_classes

def build_loaders(train_ds, test_ds, batch_size: int, val_split: float = 0.1, num_workers: int = 2, seed: int = 42):
    n_total = len(train_ds)
    n_val = max(1, int(val_split * n_total))
    n_train = n_total - n_val
    set_seed(seed)
    train_subset, val_subset = random_split(train_ds, [n_train, n_val])
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_subset,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,      batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader

# ----------------- Model & Training -----------------
def build_model(arch: str, num_classes: int, pretrained: bool = True):
    arch = arch.lower()
    if arch == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        in_feats = m.fc.in_features
        m.fc = nn.Linear(in_feats, num_classes)
    elif arch == "mobilenet_v2":
        m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None)
        in_feats = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_feats, num_classes)
    elif arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        in_feats = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_feats, num_classes)
    else:
        # Simple fallback CNN (from-scratch) if needed
        m = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, num_classes)
        )
    return m

def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        if scaler is None:
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        else:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        running_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total

@torch.no_grad()
def collect_test_logits(model, loader, device):
    model.eval()
    logits = []
    for imgs, _ in loader:              # DO NOT TOUCH TEST LABELS
        imgs = imgs.to(device)
        outputs = model(imgs)           # RAW logits (NO softmax)
        logits.append(outputs.detach().cpu().numpy())
    return np.concatenate(logits, axis=0)

# ----------------- Main -----------------
def main():
    # Hard-coded student ID
    student_id = "20254086"
    npy_name = f"{student_id}.npy"

    # You can still override these manually later if the brief changes
    dataset = "fashion_mnist"
    arch = "resnet18"
    pretrained = True
    image_size = 224
    batch_size = 64
    epochs = 8
    lr = 1e-3
    weight_decay = 1e-4
    val_split = 0.1
    num_workers = 2
    seed = 42
    grayscale_to_rgb = True

    # ---------------------------------------
    set_seed(seed)
    device = get_device()
    print(f"Using device: {device}")

    train_tf, test_tf = build_transforms(dataset, image_size, grayscale_to_rgb)
    data_dir = "./data"
    train_ds, test_ds, num_classes = load_dataset(dataset, data_dir, train_tf, test_tf)
    print(f"Dataset: {dataset} | Train: {len(train_ds)} | Test: {len(test_ds)} | Classes: {num_classes}")

    train_loader, val_loader, test_loader = build_loaders(
        train_ds, test_ds,
        batch_size=batch_size,
        val_split=val_split,
        num_workers=num_workers,
        seed=seed
    )

    model = build_model(arch, num_classes, pretrained=pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler('cuda') if device.type == "cuda" else None

    best_val_acc, best_state = -1.0, None
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        print(f"Epoch {epoch:02d}/{epochs} | train_loss={tr_loss:.4f} acc={tr_acc:.4f} | val_loss={va_loss:.4f} acc={va_acc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    # Optional local check (not graded)
    te_loss, te_acc = evaluate(model, test_loader, criterion, device)
    print(f"[INFO] Test accuracy (for your reference): {te_acc*100:.2f}%")

    # Collect RAW logits on test set
    logits = collect_test_logits(model, test_loader, device)
    expected_rows = len(test_ds)
    assert logits.ndim == 2 and logits.shape[0] == expected_rows and logits.shape[1] == num_classes, \
        f"logits shape must be ({expected_rows}, {num_classes}), got {logits.shape}"
    np.save(npy_name, logits)
    print(f"[OK] Saved raw logits to ./{npy_name} with shape {logits.shape}. Do NOT rename this file.")

if __name__ == "__main__":
    main()
