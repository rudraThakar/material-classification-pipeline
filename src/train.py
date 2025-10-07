import argparse
import json
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from torchvision import models
from tqdm import tqdm

from data import build_loaders


def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


@torch.no_grad()
def evaluate(model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_preds = []
    all_targets = []
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(targets.cpu().numpy())
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    return acc, precision, recall, cm


def set_requires_grad(model: nn.Module, requires_grad: bool) -> None:
    for p in model.parameters():
        p.requires_grad = requires_grad


def train(args: argparse.Namespace) -> None:
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    train_loader, val_loader, class_index = build_loaders(
        args.train_dir,
        args.val_dir,
        args.img_size,
        args.batch_size,
        args.num_workers,
        args.split_ratio,
        use_autoaugment=not args.no_autoaugment,
        use_randresizedcrop=not args.no_randresizedcrop,
        random_erasing_p=args.random_erasing_p,
    )
    with open(args.class_index, "w", encoding="utf-8") as f:
        json.dump(class_index, f, indent=2)

    num_classes = len(class_index)
    model = build_model(num_classes=num_classes, pretrained=not args.no_pretrain).to(device)

    # Freeze all but head for a few warmup epochs, then unfreeze
    if args.freeze_epochs > 0:
        set_requires_grad(model, False)
        set_requires_grad(model.fc, True)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)
    # Simple warmup then cosine
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - args.warmup_epochs))
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    best_acc = 0.0
    history = []
    for epoch in range(1, args.epochs + 1):
        # Unfreeze after freeze_epochs
        if args.freeze_epochs > 0 and epoch == args.freeze_epochs + 1:
            set_requires_grad(model, True)
            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - epoch))

        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for images, targets in pbar:
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        # Warmup: simple linear warmup on lr
        if epoch <= args.warmup_epochs:
            for g in optimizer.param_groups:
                g["lr"] = args.lr * epoch / max(1, args.warmup_epochs)
        else:
            scheduler.step()
        train_loss = running_loss / len(train_loader.dataset)
        acc, precision, recall, cm = evaluate(model, val_loader, device)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_accuracy": acc,
            "val_precision": precision,
            "val_recall": recall,
        })
        print(f"Epoch {epoch}: loss={train_loss:.4f} acc={acc:.4f} precision={precision:.4f} recall={recall:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save({
                "model_state": model.state_dict(),
                "num_classes": num_classes,
                "img_size": args.img_size,
            }, os.path.join("models", "best_model.pt"))

        # Save confusion matrix each epoch
        cm_path = os.path.join("results", f"confusion_matrix_epoch_{epoch}.csv")
        pd.DataFrame(cm).to_csv(cm_path, index=False)

    pd.DataFrame(history).to_csv(os.path.join("results", "training_history.csv"), index=False)
    print(f"Best val accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--val_dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--split_ratio", type=float, default=0.8)
    parser.add_argument("--class_index", type=str, default="models/class_index.json")
    parser.add_argument("--no_pretrain", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--freeze_epochs", type=int, default=1)
    parser.add_argument("--warmup_epochs", type=int, default=1)
    parser.add_argument("--no_autoaugment", action="store_true")
    parser.add_argument("--no_randresizedcrop", action="store_true")
    parser.add_argument("--random_erasing_p", type=float, default=0.1)
    args = parser.parse_args()
    train(args)


