import sys,os,time
import argparse
from typing import List
import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

import segmentation_models_pytorch as smp



# =========================
# Dataset (binary segmentation)
# =========================
class ImageDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.transform = transform
        self.df = dataframe

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Load image via path from our Master Index CSV
        image = np.array(Image.open(row['image_path']).convert('RGB'))
        mask  = np.array(Image.open(row['label_path']).convert('L'))

        mask = (mask > 0).astype(np.uint8)

        if self.transform is not None:
            aug = self.transform(image=image, mask=mask)
            image = aug['image']  # image: [C,H,W] float, mask: [H,W] (ToTensorV2 后为 torch.uint8/long)
            mask = aug['mask']

        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        return {"image":image, "mask":mask} #{'A': image, 'B': mask}



# =========================
# Metrics
# =========================
def dice_coeff(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    pred = (pred > 0.5).float()
    iflat = pred.view(-1)
    tflat = target.view(-1)
    inter = (iflat * tflat).sum()
    return (2. * inter + smooth) / (iflat.sum() + tflat.sum() + smooth)


def iou(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    pred = (pred > 0.5).float()
    iflat = pred.view(-1)
    tflat = target.view(-1)
    inter = (iflat * tflat).sum()
    union = iflat.sum() + tflat.sum() - inter
    return (inter + smooth) / (union + smooth)


def pixel_accuracy(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = (pred > 0.5).float()
    correct = (pred == target).float()
    return correct.sum() / correct.numel()


# =========================
# Build model (binary)
# =========================
def build_model(name: str, n_classes: int = 1):
    print(f"Building model: {name} with {n_classes} output classes", file=sys.stderr)
    name = name.lower()
    if name in ['unet', 'u-net', 'unet_resnet34']:
        model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet",
                         in_channels=3, classes=1, activation='sigmoid')
        print("UNet model created with ResNet34 encoder.", file=sys.stderr)

    elif name in ['unet++', 'unetplusplus', 'unetpp']:
        model = smp.UnetPlusPlus(encoder_name="resnet34", encoder_weights="imagenet",
                                 in_channels=3, classes=1, activation='sigmoid')
    else:
        raise ValueError(f"Unsupported model_name '{name}'. Choose from: unet, unet++")
    print(f"Model {name} built successfully.", file=sys.stderr)
    return model


# =========================
# Train / Val
# =========================
def train_val_one_model(model_name: str,
                        train_loader: DataLoader,
                        val_loader: DataLoader,
                        device: torch.device,
                        args: argparse.Namespace,
                        save_root: str):
    print(f"\n===== Training model: {model_name} =====", file=sys.stderr)
    model = build_model(model_name, n_classes=1)

    model = model.to(device)

    # Optimizer
    if args.optimizer.lower() == 'adam':
        optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'sgd':
        optimizer = SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    else:
        raise ValueError("Optimizer not recognized (use 'Adam' or 'SGD').")

    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-7)

    # Binary segmentation: sigmoid + BCE
    criterion = nn.BCELoss()

    best_dice = 0.0
    patience = 0

    save_dir = os.path.join(save_root, 'saved_model')
    os.makedirs(save_dir, exist_ok=True)
    best_path = os.path.join(save_dir, f'best_{model_name}.pth')

    for epoch in range(args.num_epochs):
        sys.stderr.write(f"epoch {epoch + 1}/{args.num_epochs} start!\n")
        model.train()
       
        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True).float()

            optimizer.zero_grad()
            preds = model(images) # [B,1,H,W], sigmoid already applied by smp
            loss = criterion(preds, masks)
            loss.backward()
            optimizer.step()


        # ---- Validation ----
        model.eval()
        val_dice, val_iou, val_acc = 0.0, 0.0, 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device, non_blocking=True)
                masks = batch["mask"].to(device, non_blocking=True).float()
                preds = model(images)

                val_dice += dice_coeff(preds, masks).item()
                val_iou  += iou(preds, masks).item()
                val_acc  += pixel_accuracy(preds, masks).item()

        val_dice /= max(1, len(val_loader))
        val_iou  /= max(1, len(val_loader))
        val_acc  /= max(1, len(val_loader))
        print(f"[{model_name}] Val  Dice: {val_dice:.4f} | IoU: {val_iou:.4f} | Acc: {val_acc:.4f}", file=sys.stderr)

        # Save best
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), best_path)
            patience = 0
        else:
            patience += 1

        if patience > args.early_stopping:
            print(f"[{model_name}] Early stopping triggered.", file=sys.stderr)
            break

        scheduler.step()

    return best_path


# =========================
# Test
# =========================
def test_one_model(model_name: str,
                   best_path: str,
                   test_loader: DataLoader,
                   device: torch.device,
                   save_root: str):
    print(f"\n===== Testing model: {model_name} =====", file=sys.stderr)
    model = build_model(model_name, n_classes=1).to(device)
    if not os.path.exists(best_path):
        raise FileNotFoundError(f"Best model not found: {best_path}")
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()

    test_dice, test_iou, test_acc = 0.0, 0.0, 0.0
    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True).float()
            preds = model(images)

            test_dice += dice_coeff(preds, masks).item()
            test_iou  += iou(preds, masks).item()
            test_acc  += pixel_accuracy(preds, masks).item()

    n = max(1, len(test_loader))
    test_dice /= n
    test_iou  /= n
    test_acc  /= n

    # real output goes to stdout
    print(f"{test_dice:.4f},{test_iou:.4f},{test_acc:.4f}")


# =========================
# Main (argparse)
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cvidx', type = str)
    parser.add_argument('--trans', type = int, default = 1)
    parser.add_argument('--nfolds', type = int)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--model_name', type=str, default='unet',
                        help="Single or comma-separated models: 'unet', 'unet++' or 'unet,unet++'")
    parser.add_argument('--n_classes', type=int, default=2,
                        help="Kept for compatibility; this script assumes BINARY segmentation (foreground/background).")
    parser.add_argument('--early_stopping', type=int, default=20)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'SGD'])
    parser.add_argument('--num_epochs', type=int, default=20) # 20 is probably good
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    args = parser.parse_args()


    #cv index
    df = pd.read_csv(args.cvidx)

    # Paths
    save_root = "./models_temp/"
    os.makedirs(save_root, exist_ok=True)

    # randomize it
    myseed = int (time.time())
    torch.manual_seed(myseed)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device, file=sys.stderr)

    # preprocessing transformations
    if args.trans == 1:
        train_tf = A.Compose([
            A.Resize(args.image_size, args.image_size, interpolation=3), # this may cause big problem to encrypted data
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2()
        ], is_check_shapes=False)

        val_tf = A.Compose([
            A.Resize(args.image_size, args.image_size, interpolation=3),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2()
        ], is_check_shapes=False)
    elif args.trans == 2: # no resizing, need to work on size unified images
        train_tf = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2()
        ], is_check_shapes=False)

        val_tf = A.Compose([
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2()
        ], is_check_shapes=False)


    # Support comma-separated models
    models: List[str] = [m.strip() for m in args.model_name.split(',')]
    for mname in models:
        print(args.cvidx + " " + args.model_name + " trans" +str(args.trans))
        for fold_idx in range(args.nfolds):
            sys.stderr.write(f"\n--- STARTING FOLD {fold_idx} ---")

            # Split the DataFrame based on the 'fold' column
            test_df = df[df["fold"] == fold_idx]
            val_df = df[df["fold"] == (fold_idx + 1) % args.nfolds]
            train_df = df[(df["fold"] != fold_idx) & (df["fold"] != (fold_idx + 1) % args.nfolds)]

            # Create Dataset objects with their respective transforms
            train_ds = ImageDataset(train_df, transform=train_tf)
            val_ds = ImageDataset(val_df, transform=val_tf)
            test_ds = ImageDataset(test_df, transform=val_tf)

            # Create DataLoaders
            train_loader = DataLoader(train_ds, batch_size=args.batch_size,           shuffle=True, num_workers=2)
            val_loader = DataLoader(val_ds, batch_size=args.batch_size,               shuffle=False, num_workers=2)
            test_loader = DataLoader(test_ds, batch_size=args.batch_size,             shuffle=False, num_workers=2)

            best_model = train_val_one_model(mname, train_loader, val_loader, device, args, save_root)
            test_one_model(mname, best_model, test_loader, device, save_root)


if __name__ == '__main__':
    main()
