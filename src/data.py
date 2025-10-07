import argparse
import json
import os
from typing import Dict, Tuple, Optional

from PIL import Image
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def build_transforms(
    img_size: int,
    use_autoaugment: bool = True,
    use_randresizedcrop: bool = True,
    random_erasing_p: float = 0.1,
) -> Tuple[transforms.Compose, transforms.Compose]:
    train_ops = []
    if use_randresizedcrop:
        train_ops.append(transforms.RandomResizedCrop(size=img_size, scale=(0.7, 1.0), ratio=(0.9, 1.1)))
    else:
        train_ops.append(transforms.Resize((img_size, img_size)))
    train_ops.extend([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    ])
    if use_autoaugment:
        try:
            train_ops.append(transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET))
        except Exception:
            pass
    train_ops.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    if random_erasing_p > 0:
        train_ops.append(transforms.RandomErasing(p=random_erasing_p))

    train_tf = transforms.Compose(train_ops)

    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


def prepare_datasets(
    train_dir: Optional[str],
    val_dir: Optional[str],
    img_size: int,
    split_ratio: float = 0.8,
    use_autoaugment: bool = True,
    use_randresizedcrop: bool = True,
    random_erasing_p: float = 0.1,
) -> Tuple[datasets.ImageFolder, datasets.ImageFolder]:
    train_tf, val_tf = build_transforms(img_size, use_autoaugment, use_randresizedcrop, random_erasing_p)

    if train_dir and val_dir and os.path.isdir(train_dir) and os.path.isdir(val_dir):
        train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
        val_ds = datasets.ImageFolder(val_dir, transform=val_tf)
        return train_ds, val_ds

    # Single folder auto split
    if not train_dir or not os.path.isdir(train_dir):
        raise ValueError("When --val_dir is not provided, --train_dir must point to a folder with subfolders per class.")

    full_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    num_train = int(len(full_ds) * split_ratio)
    num_val = len(full_ds) - num_train
    train_subset, val_subset = random_split(full_ds, [num_train, num_val])

    # For validation we want deterministic transforms
    val_subset.dataset = datasets.ImageFolder(train_dir, transform=val_tf)
    return train_subset, val_subset


def build_loaders(
    train_dir: Optional[str],
    val_dir: Optional[str],
    img_size: int,
    batch_size: int,
    num_workers: int,
    split_ratio: float = 0.8,
    use_autoaugment: bool = True,
    use_randresizedcrop: bool = True,
    random_erasing_p: float = 0.1,
) -> Tuple[DataLoader, DataLoader, Dict[int, str]]:
    train_ds, val_ds = prepare_datasets(
        train_dir, val_dir, img_size, split_ratio, use_autoaugment, use_randresizedcrop, random_erasing_p
    )
    if hasattr(train_ds, "dataset") and hasattr(train_ds.dataset, "classes"):
        classes = train_ds.dataset.classes
    else:
        classes = train_ds.classes  # type: ignore[attr-defined]

    class_index = {i: cls for i, cls in enumerate(classes)}
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, class_index


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--val_dir", type=str, default=None)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--split_ratio", type=float, default=0.8)
    parser.add_argument("--out_class_index", type=str, default="models/class_index.json")
    parser.add_argument("--no_autoaugment", action="store_true")
    parser.add_argument("--no_randresizedcrop", action="store_true")
    parser.add_argument("--random_erasing_p", type=float, default=0.1)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_class_index), exist_ok=True)
    _, _, class_index = build_loaders(
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
    with open(args.out_class_index, "w", encoding="utf-8") as f:
        json.dump(class_index, f, indent=2)
    print(f"Saved class index to {args.out_class_index}")


