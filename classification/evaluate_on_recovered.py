import pandas as pd
import argparse
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from PIL import Image
import sys

class OfflineDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row['path']).convert('RGB')
        label = row['label']

        if self.transform:
            image = self.transform(image)

        return image, label


def get_three_way_splits(master_csv_path, test_fold_idx, num_folds=5):
    """
    Returns (train_df, val_df, test_df) for a specific fold index.
    """
    df = pd.read_csv(master_csv_path)

    # 1. The Test set is the current fold
    test_df = df[df['fold'] == test_fold_idx].reset_index(drop=True)

    # 2. The Validation set is the "next" fold (using modulo to wrap around)
    val_fold_idx = (test_fold_idx + 1) % num_folds
    val_df = df[df['fold'] == val_fold_idx].reset_index(drop=True)

    # 3. The Training set is everything else
    train_df = df[
        (df['fold'] != test_fold_idx) &
        (df['fold'] != val_fold_idx)
    ].reset_index(drop=True)

    return train_df, val_df, test_df


def get_model(model_name, n_classes):
    if model_name == 'vgg16':
        model = models.vgg16(pretrained=True)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, n_classes)
    elif model_name == 'vgg19':
        model = models.vgg19(pretrained=True)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, n_classes)
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, n_classes)
    elif model_name == 'resnet34':
        model = models.resnet34(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, n_classes)
    else:
        raise ValueError('Model name not recognized')

    return model


def get_test_transform(image_size, trans):
    """Get the test transform based on trans type"""
    if trans == 1:
        test_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif trans == 2:
        test_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif trans == 3:
        test_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
    elif trans == 4:
        test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    elif trans == 5:
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        raise ValueError('Transform type not recognized')
    
    return test_transform


def evaluate_on_fold(model, test_loader, device):
    """
    Evaluate model on test set and return metrics
    """
    model.eval()
    correct, total, all_labels, all_preds = 0, 0, [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device).long()

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # Compute metrics
    test_accuracy = correct / total
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return {
        'accuracy': test_accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'num_samples': total
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--recovered_cvidx', type=str, required=True, help='Path to recovered dataset CSV')
    parser.add_argument('--checkpoint_dir', type=str, default='./models_temp/', help='Directory containing model checkpoints')
    parser.add_argument('--model_name', type=str, default='vgg16', help='Model name (vgg16, vgg19, resnet50, resnet34)')
    parser.add_argument('--n_classes', type=int, required=True, help='Number of classes')
    parser.add_argument('--nfolds', type=int, default=5, help='Number of folds')
    parser.add_argument('--image_size', type=int, default=256, help='Image size')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--trans', type=int, default=1, help='Transform type (1-5)')
    parser.add_argument('--base_cvidx', type=str, required=True, help='Path to original training CSV (to extract fold info)')
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sys.stderr.write(f"Using device: {device}\n")

    # Verify recovered CSV exists
    if not os.path.exists(args.recovered_cvidx):
        raise FileNotFoundError(f"Recovered CSV not found: {args.recovered_cvidx}")

    if not os.path.exists(args.base_cvidx):
        raise FileNotFoundError(f"Base CSV not found: {args.base_cvidx}")

    # Get test transform
    test_transform = get_test_transform(args.image_size, args.trans)

    # Store results
    fold_results = []
    all_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

    # Iterate through folds
    for fold_idx in range(args.nfolds):
        sys.stderr.write(f"\n--- EVALUATING FOLD {fold_idx} ---\n")

        # Get test set for this fold (using base CSV for fold assignments)
        train_df, val_df, test_df = get_three_way_splits(args.recovered_cvidx, fold_idx, args.nfolds)

        # Create dataset and loader
        test_ds = OfflineDataset(test_df, transform=test_transform)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

        # Load model checkpoint
        checkpoint_path = args.checkpoint_path + f"/{args.base_cvidx}_{args.model_name}_{fold_idx}"# had to change this line

        # Load model
        model = get_model(args.model_name, args.n_classes).to(device)
        model.load_state_dict(torch.load(args.checkpoint_path, weights_only=True))
        sys.stderr.write(f"Loaded model from {checkpoint_path}\n")

        # Evaluate
        metrics = evaluate_on_fold(model, test_loader, device)

        fold_results.append({
            'fold': fold_idx,
            'num_samples': metrics['num_samples'],
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1']
        })

        # Store for averaging
        all_metrics['accuracy'].append(metrics['accuracy'])
        all_metrics['precision'].append(metrics['precision'])
        all_metrics['recall'].append(metrics['recall'])
        all_metrics['f1'].append(metrics['f1'])

        # Print fold results
        sys.stderr.write(f"Fold {fold_idx} Results:\n")
        sys.stderr.write(f"  Accuracy:  {metrics['accuracy']:.4f}\n")
        sys.stderr.write(f"  Precision: {metrics['precision']:.4f}\n")
        sys.stderr.write(f"  Recall:    {metrics['recall']:.4f}\n")
        sys.stderr.write(f"  F1:        {metrics['f1']:.4f}\n")

        # Print in CSV format for easy parsing
        print(f"{fold_idx},{metrics['accuracy']:.6f},{metrics['f1']:.6f},{metrics['precision']:.6f},{metrics['recall']:.6f}")

    # Print summary
    sys.stderr.write(f"\n\n=== SUMMARY ACROSS ALL FOLDS ===\n")
    avg_accuracy = sum(all_metrics['accuracy']) / len(all_metrics['accuracy'])
    avg_precision = sum(all_metrics['precision']) / len(all_metrics['precision'])
    avg_recall = sum(all_metrics['recall']) / len(all_metrics['recall'])
    avg_f1 = sum(all_metrics['f1']) / len(all_metrics['f1'])

    sys.stderr.write(f"Average Accuracy:  {avg_accuracy:.4f}\n")
    sys.stderr.write(f"Average Precision: {avg_precision:.4f}\n")
    sys.stderr.write(f"Average Recall:    {avg_recall:.4f}\n")
    sys.stderr.write(f"Average F1:        {avg_f1:.4f}\n")

    # Print average in CSV format
    print(f"Average,{args.recovered_cvidx},{avg_accuracy:.6f},{avg_f1:.6f},{avg_precision:.6f},{avg_recall:.6f}\n")


if __name__ == '__main__':
    main()
