# AI504 Project 1
# Make sure file is named afert StudentID, e.g., 202412345.py

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm

# --------------------- Utils ---------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------- Data & Transforms -----------------
def build_transforms(dataset: str, image_size: int, grayscale_to_rgb: bool):
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)

    # Enhanced training augmentation
    train_base = []
    if grayscale_to_rgb and dataset.lower() in {"fashion_mnist", "mnist"}:
        train_base.append(transforms.Grayscale(num_output_channels=3))
    
    train_tf = transforms.Compose(
        train_base + [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
        ]
    )
    
    # Test transform (no augmentation)
    test_base = []
    if grayscale_to_rgb and dataset.lower() in {"fashion_mnist", "mnist"}:
        test_base.append(transforms.Grayscale(num_output_channels=3))
    
    test_tf = transforms.Compose(
        test_base + [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ]
    )
    
    return train_tf, test_tf

def load_dataset(name: str, data_dir: str, train_tf, test_tf):
    dataset_map = {
        "fashion_mnist": (datasets.FashionMNIST, 10),
        "mnist": (datasets.MNIST, 10),
        "cifar10": (datasets.CIFAR10, 10),
        "cifar100": (datasets.CIFAR100, 100),
    }
    
    name_lower = name.lower()
    if name_lower not in dataset_map:
        raise ValueError(f"Unsupported dataset: {name}")
    
    dataset_class, num_classes = dataset_map[name_lower]
    train = dataset_class(root=data_dir, train=True, download=True, transform=train_tf)
    test = dataset_class(root=data_dir, train=False, download=True, transform=test_tf)
    
    return train, test, num_classes

def build_loaders(train_ds, test_ds, batch_size: int, val_split: float = 0.1, 
                  num_workers: int = 2, seed: int = 42):
    n_total = len(train_ds)
    n_val = max(1, int(val_split * n_total))
    n_train = n_total - n_val
    
    set_seed(seed)
    train_subset, val_subset = random_split(train_ds, [n_train, n_val])
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, 
                             num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, 
                           num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader

# ----------------- Model -----------------
def build_model(arch: str, num_classes: int, pretrained: bool = True, dropout: float = 0.3):
    arch_lower = arch.lower()
    
    model_map = {
        "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1, "fc"),
        "mobilenet_v2": (models.mobilenet_v2, models.MobileNet_V2_Weights.IMAGENET1K_V1, "classifier"),
        "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1, "classifier"),
    }
    
    if arch_lower in model_map:
        model_fn, weights, classifier_attr = model_map[arch_lower]
        m = model_fn(weights=weights if pretrained else None)
        
        # Replace final layer with dropout for regularization
        classifier = getattr(m, classifier_attr)
        if isinstance(classifier, nn.Sequential):
            in_feats = classifier[-1].in_features
            classifier[-1] = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_feats, num_classes)
            )
        else:
            in_feats = classifier.in_features
            setattr(m, classifier_attr, nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_feats, num_classes)
            ))
    else:
        # Simple fallback CNN
        m = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    return m

# ----------------- Training & Evaluation -----------------
def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None, epoch=1, total_epochs=1):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [Train]", leave=False)
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
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
        
        pbar.set_postfix(loss=f"{running_loss/total:.4f}", acc=f"{correct/total:.4f}")
    
    return running_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, device, desc="Eval"):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    
    pbar = tqdm(loader, desc=desc, leave=False)
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        
        running_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix(loss=f"{running_loss/total:.4f}", acc=f"{correct/total:.4f}")
    
    return running_loss / total, correct / total

@torch.no_grad()
def collect_test_logits(model, loader, device):
    model.eval()
    logits = []
    
    pbar = tqdm(loader, desc="Collecting logits", leave=False)
    for imgs, _ in pbar:
        imgs = imgs.to(device, non_blocking=True)
        outputs = model(imgs)
        logits.append(outputs.cpu().numpy())
    
    return np.concatenate(logits, axis=0)

# ----------------- Main -----------------
def main():
    # Configuration
    student_id = "20254086"
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
    dropout = 0.3                  # Added dropout
    label_smoothing = 0.1          # Added label smoothing
    patience = 4                   # Early stopping patience

    # Setup
    set_seed(seed)
    device = get_device()
    print(f"Device: {device}")

    # Data
    train_tf, test_tf = build_transforms(dataset, image_size, grayscale_to_rgb)
    train_ds, test_ds, num_classes = load_dataset(dataset, "./data", train_tf, test_tf)
    print(f"Dataset: {dataset} | Train: {len(train_ds)} | Test: {len(test_ds)} | Classes: {num_classes}")

    train_loader, val_loader, test_loader = build_loaders(
        train_ds, test_ds, batch_size, val_split, num_workers, seed
    )

    # Model with dropout
    model = build_model(arch, num_classes, pretrained, dropout).to(device)
    
    # Label smoothing for better generalization
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Cosine annealing learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    scaler = torch.amp.GradScaler('cuda') if device.type == "cuda" else None

    # Training loop with early stopping
    best_val_acc, best_state = -1.0, None
    epochs_no_improve = 0
    print(f"\nTraining {arch} for {epochs} epochs (patience={patience})...")
    
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, 
                                         device, scaler, epoch, epochs)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device, 
                                   desc=f"Epoch {epoch}/{epochs} [Val]")
        
        # Learning rate scheduling
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Track best model
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            print(f"Epoch {epoch:02d}/{epochs} | "
                  f"Train: loss={tr_loss:.4f} acc={tr_acc:.4f} | "
                  f"Val: loss={va_loss:.4f} acc={va_acc:.4f} | "
                  f"LR: {current_lr:.2e} ✓ NEW BEST")
        else:
            epochs_no_improve += 1
            print(f"Epoch {epoch:02d}/{epochs} | "
                  f"Train: loss={tr_loss:.4f} acc={tr_acc:.4f} | "
                  f"Val: loss={va_loss:.4f} acc={va_acc:.4f} | "
                  f"LR: {current_lr:.2e}")
        
        # Early stopping
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered after {epoch} epochs (no improvement for {patience} epochs)")
            break

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\nLoaded best model (val_acc={best_val_acc:.4f})")

    # Test evaluation
    te_loss, te_acc = evaluate(model, test_loader, criterion, device, desc="Test")
    print(f"Test accuracy: {te_acc*100:.2f}%")

    # Save logits
    logits = collect_test_logits(model, test_loader, device)
    npy_name = f"{student_id}.npy"
    
    assert logits.ndim == 2 and logits.shape == (len(test_ds), num_classes), \
        f"Expected shape ({len(test_ds)}, {num_classes}), got {logits.shape}"
    
    np.save(npy_name, logits)
    print(f"\n✓ Saved logits to {npy_name} with shape {logits.shape}")

if __name__ == "__main__":
    main()