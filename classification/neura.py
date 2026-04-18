import os,sys
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# --- CORE 1: PRIVATE ENCODER (The Shredder) ---
class PrivateEncoder(nn.Module):
    def __init__(self, in_channels=3, img_size=224, patch_size=16, hidden_dim=768, depth=3):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2

        # Patch Projection: Initial pixel-to-vector conversion
        layers = [
            nn.Conv2d(in_channels, hidden_dim, kernel_size=patch_size, dilation=1, stride=patch_size),
            nn.ReLU()
        ]

        # Transformation Layers: Convolutional "Secret Key" layers
        for _ in range(depth):
            layers.extend([
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, dilation=1, stride=1),
                nn.BatchNorm2d(hidden_dim, track_running_stats=False),
                nn.ReLU()
            ])

        self.image_encoder = nn.Sequential(*layers)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, hidden_dim))
        self.mixer = nn.Sequential(*[
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
            ])

    def forward(self, x, shuffle=False):
        # Input x: [Batch, 3, 224, 224] -> Output: [Batch, 196, 768]
        x = self.image_encoder(x)
        B, C, H, W = x.size()
        x = x.view([B, -1, H * W]).transpose(1, 2)
        x += self.pos_embedding
        x = self.mixer(x)

        if shuffle:
            shuffled = torch.zeros_like(x)
            for i in range(B):
                idx = torch.randperm(H * W, device=x.device)
                shuffled[i] = x[i, idx]
            x = shuffled
        return x

# --- CORE 2: DEEP VIT BACKBONE (The Transformer) ---
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.reattn_weights = nn.Parameter(torch.randn(heads, heads))
        self.reattn_norm = nn.Sequential(
            Rearrange('b h i j -> b i j h'), nn.LayerNorm(heads), Rearrange('b i j h -> b h i j')
        )
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        h = self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        attn = einsum('b h i j, h g -> b g i j', attn, self.reattn_weights)
        attn = self.reattn_norm(attn)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class TransformerBlock(nn.Module):
    def __init__(self, dim,  heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads=heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim), nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.attn(self.norm1(x)) + x
        x = self.mlp(self.norm2(x)) + x
        return x

class NeuraCryptViT(nn.Module):
    def __init__(self, num_classes=2, dim=768, depth=12, heads=12, mlp_dim=1536):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = nn.Sequential(*[TransformerBlock(dim, heads, mlp_dim) for _ in range(depth)])
        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, x):
        # x: [Batch, 196, 768] (Pre-encoded patches)
        b = x.shape[0]
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.transformer(x)
        return self.mlp_head(x[:, 0])


import torch
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
import pandas as pd
import os
from torchvision import transforms

class NeuraCryptRawDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path =  row['path']
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        y = torch.tensor(row['label'], dtype=torch.long)
        return img, y


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import os
from sklearn.metrics import precision_score, recall_score, f1_score

# --- 1. EVALUATION FUNCTION ---
def evaluate(model, encoder, shuffle, loader, device):
    """
    Standard evaluation loop for Validation or Test sets.
    """
    model.eval()
    encoder.eval()
    correct = 0
    total = 0

    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            # Shred images into patches
            encoded = encoder(images, shuffle = shuffle )
            # Forward pass through Transformer backbone
            outputs = model(encoded)
            preds = torch.argmax(outputs, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    return correct / (total + 1e-7), precision_score(all_labels, all_preds, average='macro', zero_division=0), recall_score(all_labels, all_preds, average='macro', zero_division=0), f1_score(all_labels, all_preds, average='macro',zero_division=0)

# --- 2. MAIN CROSS-VALIDATION LOOP ---
def run_full_experiment(csv_path, n_classes = 2, n_folds=5, shuffle=0, l_rate =1e-4, n_epochs = 20, batch_size = 16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nc_shuffle = False if shuffle == 0 else True

    # 1. Training Augmentations (Active)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 2. Validation/Test Transforms (Deterministic)
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Initialize the Static Secret Key (Frozen PrivateEncoder)
    encoder = PrivateEncoder(img_size=224, patch_size=16).to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # Save the encoder (Key) so it can be reused later
    torch.save(encoder.state_dict(), "neuracrypt_secret_key.pth")


    ds = NeuraCryptRawDataset(csv_path, transform=train_transform)
    df = ds.df

    for fold_id in range(n_folds):
        print(f"\n{'='*20} STARTING FOLD {fold_id} {'='*20}", file=sys.stderr)

        # Get indices for this fold
        test_idx = df[df['fold'] == fold_id].index.tolist()
        val_idx = df[df['fold'] == (fold_id + 1) % n_folds].index.tolist()
        train_idx = df[~df['fold'].isin([fold_id, (fold_id + 1) % n_folds])].index.tolist()

        # --- KEY MODIFICATION: Create three separate dataset instances ---

        # 1. Training Dataset (with Augmentation)
        train_loader = DataLoader(Subset(ds, train_idx), batch_size=batch_size, shuffle=True)

        # 2. Validation Dataset (Clean/No Augmentation)
        val_loader = DataLoader(Subset(ds, val_idx), batch_size=batch_size)

        # 3. Test Dataset (Clean/No Augmentation)
        test_loader = DataLoader(Subset(ds, test_idx), batch_size=batch_size)

        # Initialize fresh ViT for this fold
        model = NeuraCryptViT(num_classes=n_classes).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=l_rate)
        criterion = nn.CrossEntropyLoss()

        # Training Phase
        best_val_f1 = 0.0
        for epoch in range(n_epochs):
            model.train()
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                # On-the-fly encoding (shredding)
                with torch.no_grad():
                    encoded = encoder(images, nc_shuffle)

                optimizer.zero_grad()
                loss = criterion(model(encoded), labels)
                loss.backward()
                optimizer.step()

            # Validation Checkpointing
            val_acc,val_prec,val_recall,val_f1 = evaluate(model, encoder, nc_shuffle, val_loader, device)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(model.state_dict(), f"best_backbone_fold_{fold_id}.pth")
            print(f"Epoch {epoch} | Val Acc: {val_acc:.4f} | f1: {val_f1:.4f}", file=sys.stderr)

        # --- TEST FOLDER EVALUATION ---
        # Load the best weights for this fold's backbone
        model.load_state_dict(torch.load(f"best_backbone_fold_{fold_id}.pth"))
        test_acc,test_prec,test_recall,test_f1 = evaluate(model, encoder, nc_shuffle, test_loader, device)
        print(f"{fold_id},{test_acc:.4f},{test_f1:.4f},{test_prec:.4f},{test_recall:.4f}")

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cvidx', type = str)
    parser.add_argument('--nfolds', type = int)
    parser.add_argument('--batch_size', type = int, default = 16)
    parser.add_argument('--n_classes', type = int)
    parser.add_argument('--num_epochs', type = int, default = 20)

    parser.add_argument('--learning_rate', type = float, default = 1e-4)
    parser.add_argument('--trans', type = int, default = 1)
    parser.add_argument('--shuffle', type = int, default = 0)
    args = parser.parse_args()
    print(f"{args.cvidx},shuffle={args.shuffle}")
    run_full_experiment(args.cvidx, n_classes = args.n_classes, n_folds = args.nfolds, shuffle=args.shuffle, l_rate = args.learning_rate, n_epochs = args.num_epochs)


if __name__ == "__main__":
    main()

