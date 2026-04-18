
import pandas as pd
import argparse
import os,time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
from PIL import Image
import sys

def train_model(model, train_loader, val_loader, criterion, optimizer,
                device, args, fold):

    best_accuracy, best_loss, early_stopping_counter, best_f1 = 0, float('inf'), 0, 0.0
    train_losses = []
    val_losses = []
    for epoch in range(args.num_epochs):
        sys.stderr.write(f"epoch {epoch + 1}/{args.num_epochs} start!\n")
        model.train()
        running_loss,correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).long()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predicted = torch.argmax(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_accuracy = correct / total
        sys.stderr.write(f"train accuracy {train_accuracy}\n")
        avg_train_loss = running_loss / len(train_loader)
        sys.stderr.write(f"train loss {avg_train_loss}\n")

        train_losses.append(avg_train_loss)

        #validation phase
        model.eval()
        val_loss, correct, total, all_labels, all_preds = 0.0, 0, 0, [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device).long()

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                predicted = torch.argmax(outputs, dim=1)

                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

        val_accuracy = correct / total
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        #evaluation metric
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        sys.stderr.write(f"train accuracy: {train_accuracy} | val accuracy: {val_accuracy}\n")
        sys.stderr.write(f"precision: {precision: 4f} | recall: {recall: 4f} | f1: {f1: 4f}\n")

        #if val_f1 > best_f1: Best to use F1 when we have class imbalance
        if f1 > best_f1:
            best_f1 = f1
            save_path = f"./models_temp/"
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torch.save(model.state_dict(), save_path + args.cvidx + f"_{args.model_name}")

        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter > args.early_stopping:
            sys.stderr.write(f"early stopping trigger, stopping training!\n")
            break
    return save_path + args.cvidx + f"_{args.model_name}_{fold}"

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
    # We exclude both the test fold and the validation fold
    train_df = df[
        (df['fold'] != test_fold_idx) &
        (df['fold'] != val_fold_idx)
    ].reset_index(drop=True)

    return train_df, val_df, test_df

class OfflineDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Load image via path from our Master Index CSV
        image = Image.open(row['path']).convert('RGB')
        label = row['label']

        if self.transform:
            image = self.transform(image)

        return image, label


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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cvidx', type = str)
    parser.add_argument('--nfolds', type = int)
    parser.add_argument('--image_size', type = int, default = 256)
    parser.add_argument('--batch_size', type = int, default = 16)
    parser.add_argument('--model_name', type = str, default = 'vgg16')
    parser.add_argument('--n_classes', type = int)
    parser.add_argument('--early_stopping', type = int, default = 10)
    parser.add_argument('--weight_decay', type = float, default = 1e-4)
    parser.add_argument('--optimizer', type = str, default = 'Adam')
    parser.add_argument('--num_epochs', type = int, default = 20)

    parser.add_argument('--learning_rate', type = float, default = 1e-4)
    parser.add_argument('--trans', type = int, default = 1)
    args = parser.parse_args()

    # randomize it
    myseed = int (time.time())
    torch.manual_seed(myseed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform = None
    val_transform = None
    test_transform = None

    if args.trans == 1:
        train_transform = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),  # Standard for pre-trained models like VGG16
            transforms.RandomHorizontalFlip(p=0.5),  # Breast tissue is mostly symmetric
            transforms.RandomVerticalFlip(p=0.3),  # Less likely but useful for some mammograms
            transforms.RandomRotation(degrees=10),  # Reduce rotation to ≤10° (15° may distort key structures)
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.01),
            transforms.RandomAffine(degrees=0, translate=(0.02, 0.02)),  #Small translations for position invariance
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),  #Reduce blur intensity (0.1 to 0.5)
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
        ])

        # both val and test transforms cannot use augmentation
        val_transform = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        test_transform = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif args.trans == 2:
        train_transform = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        val_transform = train_transform
        test_transform = train_transform
    elif args.trans == 3:
        train_transform = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
        ])
        val_transform = train_transform
        test_transform = train_transform
    elif args.trans == 4: # Apply augmentations but no normalization and no resizing (keep original image size)
        train_transform = transforms.Compose([ 
            transforms.RandomHorizontalFlip(p=0.5),  # Breast tissue is mostly symmetric
            transforms.RandomVerticalFlip(p=0.3),  # Less likely but useful for some mammograms
            transforms.RandomRotation(degrees=10),  # Reduce rotation to ≤10° (15° may distort key structures)
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.01),
            transforms.RandomAffine(degrees=0, translate=(0.02, 0.02)),  #Small translations for position invariance
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),  #Reduce blur intensity (0.1 to 0.5)
            transforms.ToTensor()
        ])

        # both val and test transforms cannot use augmentation
        val_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        test_transform = val_transform

    elif args.trans == 5: # with normalization
        train_transform = transforms.Compose([
            #transforms.Resize((args.image_size, args.image_size)),  # Standard for pre-trained models like VGG16
            transforms.RandomHorizontalFlip(p=0.5),  # Breast tissue is mostly symmetric
            transforms.RandomVerticalFlip(p=0.3),  # Less likely but useful for some mammograms
            transforms.RandomRotation(degrees=10),  # Reduce rotation to ≤10° (15° may distort key structures)
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.01),
            transforms.RandomAffine(degrees=0, translate=(0.02, 0.02)),  #Small translations for position invariance
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),  #Reduce blur intensity (0.1 to 0.5)
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
        ])

        # both val and test transforms cannot use augmentation
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        test_transform = val_transform

    num_folds = args.nfolds

    print(args.cvidx + " " + args.model_name + " trans" +str(args.trans))
    for fold_idx in range(num_folds):
        sys.stderr.write(f"\n--- STARTING FOLD {fold_idx} ---")

        # Split the DataFrame based on the 'fold' column
        train_df, val_df, test_df = get_three_way_splits(args.cvidx, fold_idx)

        # Create Dataset objects with their respective transforms
        train_ds = OfflineDataset(train_df, transform=train_transform)
        val_ds = OfflineDataset(val_df, transform=val_transform)
        test_ds = OfflineDataset(test_df, transform=test_transform)

        # Create DataLoaders
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

        # need to reinitialize the model every time
        model = get_model(args.model_name, args.n_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        if args.optimizer == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr = args.learning_rate, weight_decay=args.weight_decay)
        elif args.optimizer == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr = args.learning_rate, momentum = 0.9, weight_decay = args.weight_decay)
        else:
            raise ValueError('Optimizer not recognized')

        best_model = train_model(model, train_loader, val_loader, criterion, optimizer, device, args, fold_idx)


        #load best_model
        model.load_state_dict(torch.load(best_model, weights_only=True))
        model.eval()

        correct, total, all_labels, all_preds= 0, 0, [], []

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device).long()

                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

            # Compute test metrics
            test_acc = correct / total
            precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
            recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
            f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)


            #save to csv format
            print(str(fold_idx) + "," + ",".join([ str(r) for r in [test_acc, f1, precision, recall]]))


if __name__ == '__main__':
    main()











