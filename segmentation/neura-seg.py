import torch
import argparse
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms.functional as TF
from PIL import Image
import pandas as pd
import sys
import numpy as np

sys.path.insert(0, "../classification/")
from neura import PrivateEncoder, NeuraCryptViT

import torch
import torch.nn as nn
from einops import repeat,rearrange

class NeuraCryptSeg(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.vit = NeuraCryptViT() # Your existing NeuraCryptViT
        dim = 768 # Based on your dim=768

        # 1. Segmentation Head (The Decoder)
        self.decoder = nn.Sequential(
            # Input: [B, 768, 14, 14]
            nn.Conv2d(dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Step 1: 14x14 -> 56x56
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Step 2: 56x56 -> 224x224
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )

    def forward(self, x):
        # x: [Batch, 196, 768] (Pre-encoded patches)
        b = x.shape[0]

        # Add CLS token and run transformer just like your original code
        cls_tokens = repeat(self.vit.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)

        # Pass through DeepViT Transformer blocks
        # Output: [B, 197, 768]
        x = self.vit.transformer(x)

        # --- THE SEGMENTATION STEP ---
        # 1. Drop CLS (index 0), keep the 196 patches
        patches = x[:, 1:, :] # [B, 196, 768]

        # 2. Reshape to spatial grid (sqrt(196) = 14)
        # [B, 196, 768] -> [B, 768, 14, 14]
        grid = rearrange(patches, 'b (h w) c -> b c h w', h=14, w=14)

        # 3. Upsample to pixel mask
        mask = self.decoder(grid) # [B, 1, 224, 224]
        return mask


class NeuraCryptSegDataset(Dataset):
    def __init__(self, csv_path,is_train=True):
        self.df = pd.read_csv(csv_path)
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['image_path']).convert('RGB')
        # Assuming masks are in a separate folder with the same relative path
        mask = Image.open(row['label_path']).convert('L')

        # 1. Mandatory Resize for NeuraCrypt (224x224)
        img = TF.resize(img, (224, 224))
        mask = TF.resize(mask, (224, 224), interpolation=TF.InterpolationMode.NEAREST)

        # 2. Augmentations (Only for Training)
        if self.is_train:
            if torch.rand(1) > 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)
            if torch.rand(1) > 0.5:
                angle = torch.randint(-15, 15, (1,)).item()
                img = TF.rotate(img, angle)
                mask = TF.rotate(mask, angle)

        # 3. Final Tensor Conversion
        img = TF.to_tensor(img)
        img = TF.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        # Binary mask: 1 for foreground, 0 for background
        mask = torch.from_numpy(np.array(mask)).float().unsqueeze(0) / 255.0
        return img, mask


def dice_coeff(preds, targets, threshold=0.5):
    preds = (torch.sigmoid(preds) > threshold).float()
    inter = (preds * targets).sum()
    return (2. * inter) / (preds.sum() + targets.sum() + 1e-7)

def iou(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.  Tensor:
    pred = (pred > 0.5).float()
    iflat = pred.view(-1)
    tflat = target.view(-1)
    inter = (iflat * tflat).sum()
    union = iflat.sum() + tflat.sum() - inter
    return (inter + smooth) / (union + smooth)


def run_segmentation_cv(csv_path, n_folds=5, n_epochs = 20, batch_size = 16, nc_shuffle = 0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nc_shuffle = False if nc_shuffle == 0 else True

    # Fixed Secret Key (PrivateEncoder)
    encoder = PrivateEncoder(img_size=224, patch_size=16).to(device)
    encoder.eval()
    for p in encoder.parameters(): p.requires_grad = False

    df = pd.read_csv(csv_path)

    for fold_id in range(n_folds):
        sys.stderr.write(f"\n--- STARTING FOLD {fold_id} ---\n")

        # Index splits
        test_idx = df[df['fold'] == fold_id].index.tolist()
        val_idx = df[df['fold'] == (fold_id + 1) % n_folds].index.tolist()
        train_idx = df[~df['fold'].isin([fold_id, (fold_id + 1) % n_folds])].index.tolist()

        # Create separate dataset instances for different transforms
        train_ds = NeuraCryptSegDataset(csv_path, is_train=True)
        val_ds = NeuraCryptSegDataset(csv_path, is_train=False)

        train_loader = DataLoader(Subset(train_ds, train_idx), batch_size, shuffle=True)
        val_loader = DataLoader(Subset(val_ds, val_idx), batch_size)
        test_loader = DataLoader(Subset(val_ds, test_idx), batch_size)

        # Initialize Seg Model
        model = NeuraCryptSeg(num_classes=1).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        criterion = torch.nn.BCEWithLogitsLoss()

        best_val_dice = 0.0
        for epoch in range(n_epochs):
            model.train()
            for images, masks in train_loader:
                images, masks = images.to(device), masks.to(device)

                # Apply PrivateEncoder on-the-fly
                with torch.no_grad():
                    encoded = encoder(images, shuffle=nc_shuffle)

                optimizer.zero_grad()
                outputs = model(encoded)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

            # Validation Round
            val_dice, val_iou = evaluate_seg(model, encoder, nc_shuffle, val_loader, device)
            if val_dice > best_val_dice:
                best_val_dice = val_dice
                torch.save(model.state_dict(), f"./models_temp/best_seg_fold_{fold_id}.pth")
            print(f"Epoch {epoch} | Val Dice: {val_dice:.4f}", file=sys.stderr)

        # --- TEST FOLDER EVALUATION ---
        model.load_state_dict(torch.load(f"./models_temp/best_seg_fold_{fold_id}.pth"))
        test_dice, test_iou = evaluate_seg(model, encoder, nc_shuffle, test_loader, device)
        print(f"{fold_id},{test_dice:.4f},{test_iou:.4f}")

def evaluate_seg(model, encoder, shuffle, loader, device):
    model.eval()
    total_dice = 0
    total_iou =0
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            encoded = encoder(images, shuffle)
            outputs = model(encoded)
            total_dice += dice_coeff(outputs, masks).item()
            total_iou += iou(outputs, masks).item()
    return total_dice / len(loader), total_iou/ len(loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cvidx', type = str)
    parser.add_argument('--nfolds', type = int)
    parser.add_argument('--shuffle', type = int, default =0)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=20) # 20 is probably    good
    args = parser.parse_args()
    run_segmentation_cv(csv_path = args.cvidx,
                        n_folds  = args.nfolds,
                        n_epochs = args.num_epochs,
                        batch_size = args.batch_size,
                        nc_shuffle = args.shuffle)

if __name__ == '__main__':
    main()
