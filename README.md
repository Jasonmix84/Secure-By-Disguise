# Secure-By-Disguise: Clinical Validation of Image Disguising Methods for Confidential Medical Image Modeling

The integration of Deep Learning (DL) into clinical workflows offers transformative potential for medical image analysis, but it often requires high-performance cloud computing that poses significant risks to Protected Health Information (PHI). This repository implements and evaluates a privacy-preserving framework designed to bridge the gap between high-performance cloud training and strict data confidentiality.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Quick Start](#quick-start)
  - [Dataset Setup](#dataset-setup)
  - [Environment Configuration](#environment-configuration)
- [Workflow Guide](#workflow-guide)
  - [Step 1: Data Cleanup](#step-1-data-cleanup)
  - [Step 2: Image Encryption](#step-2-image-encryption)
  - [Step 3: Classification](#step-3-classification)
  - [Step 4: Segmentation](#step-4-segmentation)
  - [Step 5: NeuraCrypt Architecture](#step-5-neuracrypt-architecture)
  - [Step 6: Cryptographic Attacks](#step-6-cryptographic-attacks)
  - [Step 7: Recovery and Evaluation](#step-7-recovery-and-evaluation)
- [Detailed Command Reference](#detailed-command-reference)
- [Project Structure](#project-structure)
- [References](#references)

## Project Overview

This framework investigates the security of image encryption methods by:

1. **Encrypting** images using three distinct methods (Random Matrix Tranformations (RMT), Advanved Encryption Standard (AES), and NeuraCrypt).
2. **Training** deep learning models (classification and segmentation) on encrypted images.
3. **Testing** utility of RMT and AES with different parameters against NeuraCrypt. 
4. **Executing attacks** using matrix estimation and codebook attacks.
5. **Evaluating robustness** by comparing model performance on recovered vs. encrypted images.


## Features

- **Multiple Encryption Methods**:
  - **RMT**: Block-based encryption with configurable block size, noise, and shuffling.
  - **AES**: Byte-level symmetric encryption.
  - **NeuraCrypt**: Random neural networks to encode data, hiding the actual data.

- **Computer Vision Tasks**:
  - Image classification using pre-trained models (VGG16, VGG19, ResNet32, ResNet50).
  - Semantic segmentation using pretrained models (Unet, Unet++).
  - 5-fold cross-validation for robust evaluation.

- **Attack Methodologies**:
  - Encryption Matrix Estimation for RMT. 
  - Codebook attacks for AES.

- **Comprehensive Evaluation**: Metrics including accuracy, precision, recall, and F1-score for classification. Dice and IoU for segmentation.

---

## Quick Start

### Dataset Setup

This project used 4 datasets. Download them from the URLs below and place them in appropriate directories:

```bash
# Dataset 1: Breast Dataset 400x magnification Dataset
# URL: https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/
# Extract to: ./Secure-By-Disguise/Data/Breast

# Dataset 2: Wound Classification Dataset
# URL: https://github.com/uwm-bigdata/wound_classification
# Extract to: ./Secure-By-Disguise/Data/MData

# Dataset 3: Endoscopic Colonoscopy Frames for Polyp Dectection Dataset
# URL: https://www.kaggle.com/datasets/balraj98/cvcclinicdb
# Extract to: ./Secure-By-Disguise/Data/CVC

# Dataset 4: DFUTissue Dataset
# URL: https://github.com/uwm-bigdata/DFUTissueSegNet/tree/main
# Note: This dataset is not yet publically available. The dataset was provided to us directly from the owners.
# Extract to: ./Secure-By-Disguise/Data/DFUT
```


Dataset Directory Structure (example for classification):
```
Data/
├── dataset1/
│   ├── class_1/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── class_2/
│       ├── image1.jpg
│       └── image2.jpg
```

### Environment Configuration

#### 1. Create Virtual Environment

```bash
# Navigate to project root
cd Secure-By-Disguise

# Create Python virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On Linux/macOS:
source venv/bin/activate
```

#### 2. Install Dependencies

```bash
cd encryption
pip install -r requirements.txt
cd ..
```

The `requirements.txt` includes:
- PyTorch with CUDA support
- TorchVision for image processing
- Pandas for data management
- OpenCV, PIL for image operations
- ImageHash for duplicate detection
- Cryptography libraries (PyCryptodome)

---

## Workflow Guide

### Step 1: Data Cleanup

Before encryption, remove duplicate images from your dataset to ensure data quality.

#### Command:
```bash
cd encryption

python find-dupes.py \
  --target_dir ../Data/dataset1 \
  --output_dir ../Data/dataset1_cleaned \
  --output_log duplicate_report.txt \
  --copy_labels 0
```

#### Parameters:
- `--target_dir`: Source directory containing raw images
- `--output_dir`: Destination for deduplicated images
- `--output_log`: Report file listing removed duplicates
- `--copy_labels` (optional): Set to `1` if copying associated label files. Necessary for segmentation datasets.

#### Output:
- Cleaned dataset in output directory
- `duplicate_report.txt` containing list of removed duplicates and their matches

#### Example Output:
```
Scanning '/datasets/dataset1' for visually identical images...
Unique images will be safely copied to '/datasets/dataset1_cleaned'

Skipped duplicate: /datasets/dataset1/class_1/img_256.jpg (Duplicate of /datasets/dataset1/class_1/img_001.jpg)
...
Total: 5000 images processed, 4823 unique images copied, 177 duplicates removed
```

---

#### (Optional) Command 2 For Wound Classifcation Dataset:
The following commands are optional. We find that resizing before encryption helps improve model utility as resizing will be dependent on the image matrix which will be affeted during encryption. Only valuable for wound classification dataset which has images of varying sizes.

```bash
cd encryption

python img-size.py \
  --input ../Data/dataset1_cleaned \
  --output ../Data/dataset1_cleaned_resized
```

#### Parameters:
- `--input`: Source directory containing cleaned images
- `--output`: Destination for resized images

#### Output:
- Images of uniform size all in the output dataset


### Step 2: Image Encryption

Encrypt your dataset using one of three available encryption methods.

#### Basic Command Structure:
```bash
cd encryption

python main.py \
  --input /path/to/Dataset \
  --output /path/to/encrypted_output \
  --method [RMT|AES|NeuraCrypt] \
  --block_size 4 \
  --noise_level 0 \
  --shuffle False
```

#### Sample Calls:

**RMT Encryption (Block Size 4, No Noise)**:
```bash
python main.py \
  --input ../Data/dataset1_cleaned \
  --output ../ModelData/Dataset1/RMT-B4N0 \
  --method RMT \
  --block_size 4 \
  --noise_level 0 \
  --shuffle False
```

**RMT Encryption (Block Size 8, With Noise)**:
```bash
python main.py \
  --input ../Data/dataset1_cleaned \
  --output ../ModelData/Dataset1/RMT-B8N4 \
  --method RMT \
  --block_size 8 \
  --noise_level 4 \
  --shuffle True
```

**AES Encryption**:
```bash
python main.py \
  --input ../Data/dataset1_cleaned \
  --output ../ModelData/Dataset1/AES \
  --method AES
```

**NeuraCrypt Encryption**:
```bash
python main.py \
  --input ../Data/dataset1_cleaned \
  --output ../ModelData/Dataset1/NeuraCrypt \
  --method NeuraCrypt
```

#### Parameters:
- `--input` (required): Path to clean dataset directory
- `--output` (required): Output directory for encrypted images
- `--method` (required): Encryption method: `RMT`, `AES`, or `NeuraCrypt`
- `--block_size` (default: 4): Block size for RMT (e.g., 4, 8, 16)
- `--noise_level` (default: 0): Noise level (0 = no noise, higher = more noise)
- `--shuffle` (default: False): Whether to shuffle pixel blocks

#### Output:
- Encrypted image dataset in specified output directory
- Directory structure mirrors input dataset

---

### Step 3: Classification

Train and evaluate image classifiers on encrypted datasets.

#### Step 3.1: Prepare Dataset CSV

Create a stratified cross-validation split configuration:

```bash
cd classification

python prepare-cv.py \
  ../ModelData/Dataset1/RMT-B4N0 \
  Dataset1_RMT_B4N0_cvidx.csv
```

Parameters:
- Argument 1: Path to encrypted dataset
- Argument 2: Output CSV filename

**Output**: CSV file with columns:
- `path`: Image file path
- `class_name`: Class label
- `label`: Numeric label encoding
- `fold`: Fold assignment (0-4 for 5-fold CV)

#### Step 3.2: Train Classification Model

```bash
python classify.py \
  --model_name vgg16 \
  --cvidx Dataset1_RMT_B4N0_cvidx.csv \
  --nfolds 5 \
  --n_classes 2 \
  --num_epochs 20 \
  --batch_size 32 \
  --learning_rate 0.001
```

#### Available Models:
- `vgg16`: VGG-16 architecture
- `resnet50`: ResNet-50
- `unet`: Unet
- `unet++`: Unet++

#### Example Calls with Different Settings:

**Lightweight Model (Fast Training)**:
```bash
python classify.py \
  --model_name vgg16 \
  --cvidx Dataset1_RMT_B4N0_cvidx.csv \
  --nfolds 5 \
  --n_classes 2 \
  --num_epochs 30 \
  --batch_size 64 \
  --learning_rate 0.001 \
  >> results_vgg16_rmt_b4n0.log
```

**Robust Model (Longer Training)**:
```bash
python classify.py \
  --model_name resnet50 \
  --cvidx Dataset1_RMT_B4N0_cvidx.csv \
  --nfolds 5 \
  --n_classes 2 \
  --num_epochs 100 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  >> results_resnet50_rmt_b4n0.log
```

#### Output:
- Trained model saved to `./models_temp/`
- Log file with training metrics:
  - Per-epoch accuracy, loss
  - Precision, recall, F1-score
  - Fold-wise performance summary

---

### Step 4: Segmentation

Train segmentation models for pixel-level predictions on encrypted data.

#### Step 4.1: Encrypt Segmentation Dataset

**Expected Directory Structure**:
```
segmentation_dataset/
├── images/
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── ...
└── labels/
    ├── img_001.png
    ├── img_002.png
    └── ...
```


```bash
cd ../encryption

python main.py \
  --input ../datasets/segmentation_dataset/images \
  --output ../ModelData/Segmentation/RMT-B4N0 \
  --method RMT \
  --block_size 4 \
  --noise_level 0
```

#### Step 4.2: Prepare Segmentation Dataset CSV

Prepare dataset with associated label masks:

```bash
cd segmentation

python prepare-cv-seg.py \
  ../datasets/segmentation_dataset \
  Dataset_seg_cvidx.csv \
  5
```

Parameters:
- Argument 1: Root directory (must contain `images/` and `labels/` subdirectories)
- Argument 2: Output CSV filename
- Argument 3 (optional): Number of folds (default: 5)

#### Step 4.3: Train Segmentation Model

```bash
cd ../segmentation

python segmentation.py \
  --cvidx Dataset_seg_cvidx.csv \
  --nfolds 5 \
  --num_classes 2 \
  --num_epochs 50 \
  --batch_size 8 \
  --learning_rate 0.001
```

#### Output:
- Segmentation model weights saved
- Per-epoch metrics (Dice coefficient, IoU, pixel accuracy)
- Validation visualizations

---

### Step 5: NeuraCrypt Architecture

NeuraCrypt combines neural networks with encryption for joint privacy and utility preservation.

#### Step 5.1: NeuraCrypt Classification

```bash
cd classification

python neura.py \
  --cvidx Dataset1_NeuraCrypt_cvidx.csv \
  --nfolds 5 \
  --n_classes 2 \
  --num_epochs 50 \
  --batch_size 32
```

#### Step 5.2: NeuraCrypt Segmentation

```bash
cd ../segmentation

python neura-seg.py \
  --cvidx Dataset_seg_NeuraCrypt_cvidx.csv \
  --nfolds 5 \
  --num_classes 2 \
  --num_epochs 50 \
  --batch_size 8
```

#### NeuraCrypt Parameters:
- `--cvidx`: Path to cross-validation index CSV
- `--nfolds`: Number of cross-validation folds
- `--n_classes` (classification) or `--num_classes` (segmentation): Number of classes
- `--num_epochs`: Training epochs
- `--batch_size`: Batch size for training

#### Sample Output:
```
Epoch 1/50: Train Loss: 2.1523, Val Accuracy: 45.2%
Epoch 2/50: Train Loss: 1.8932, Val Accuracy: 62.1%
...
Epoch 50/50: Train Loss: 0.3421, Val Accuracy: 92.3%
```

---

### Step 6: Cryptographic Attacks

Test the vulnerability of RMT and AES methods using appropriate attacks.

#### Basic Attack Command:

```bash
cd encryption

python main.py \
  --input ../datasets/dataset1_cleaned \
  --output ../recovered/dataset1_recovered_rmt_b4n0 \
  --method RMT \
  --block_size 4 \
  --attack True \
  --encrypted_dir ../ModelData/Dataset1/RMT-B4N0 \
  --known_pairs 10
```

#### Attack Types:

**RMT Attack (Random Matrix Theory Recovery)**:
```bash
python main.py \
  --input ../datasets/dataset1_cleaned \
  --output ../recovered/RMT_B4N0_recovered \
  --method RMT \
  --block_size 4 \
  --noise_level 0 \
  --attack True \
  --encrypted_dir ../ModelData/Dataset1/RMT-B4N0 \
  --known_pairs 10 \
  --shuffle False
```

**AES Codebook Attack**:
```bash
python main.py \
  --input ../datasets/dataset1_cleaned \
  --output ../recovered/AES_recovered \
  --method AES \
  --attack True \
  --encrypted_dir ../ModelData/Dataset1/AES \
  --known_pairs 20
```


#### Attack Parameters:
- `--attack True`: Enable attack mode
- `--encrypted_dir`: Directory of encrypted images
- `--known_pairs`: Number of known plaintext-ciphertext pairs (RMT/AES)
- `--output`: Directory to save recovered images

#### Output:
- Recovered image dataset in output directory
- Attack success metrics logged to console
- Recovery quality assessment

---

### Step 7: Recovery and Evaluation

Evaluate model performance on recovered images to assess encryption robustness.

#### Step 7.1: Prepare Recovered Dataset CSV

Create evaluation CSV for recovered images:

```bash
cd classification

python prepare_recovered_csv.py \
  --base_csv Dataset1_RMT_B4N0_cvidx.csv \
  --encrypted_dir ../ModelData/Dataset1/RMT-B4N0 \
  --recovered_dir ../recovered/RMT_B4N0_recovered \
  --output_csv Dataset1_RMT_B4N0_recovered_cvidx.csv
```

#### Parameters:
- `--base_csv`: Original training CSV
- `--encrypted_dir`: Original encrypted directory path
- `--recovered_dir`: Path to recovered dataset
- `--output_csv`: Output CSV for recovered images

#### Output:
- `Dataset1_RMT_B4N0_recovered_cvidx.csv` with updated paths pointing to recovered images
- Verification of path existence for all samples

#### Step 7.2: Evaluate Model on Recovered Images

```bash
python evaluate_on_recovered.py \
  --model_path ./models_temp/fold_0_best_model.pt \
  --recovered_csv Dataset1_RMT_B4N0_recovered_cvidx.csv \
  --n_classes 2 \
  --batch_size 32
```

#### Parameters:
- `--model_path`: Path to trained model checkpoint
- `--recovered_csv`: CSV file for recovered images
- `--n_classes`: Number of classes
- `--batch_size`: Batch size for evaluation

#### Output:
- Accuracy on recovered images
- Per-class precision, recall, F1-score
- Comparison with performance on encrypted images
- Recovery success rate

#### Complete Evaluation Workflow Example:

```bash
# 1. Train model on encrypted images
python classify.py \
  --model_name vgg16 \
  --cvidx Dataset1_RMT_B4N0_cvidx.csv \
  --nfolds 5 \
  --n_classes 2 >> results_encrypted.log

# 2. Perform attack
cd ../encryption
python main.py \
  --input ../datasets/dataset1_cleaned \
  --output ../recovered/RMT_B4N0_recovered \
  --method RMT \
  --block_size 4 \
  --attack True \
  --encrypted_dir ../ModelData/Dataset1/RMT-B4N0 \
  --known_pairs 10

# 3. Prepare evaluation CSV
cd ../classification
python prepare_recovered_csv.py \
  --base_csv Dataset1_RMT_B4N0_cvidx.csv \
  --encrypted_dir ../ModelData/Dataset1/RMT-B4N0 \
  --recovered_dir ../recovered/RMT_B4N0_recovered \
  --output_csv Dataset1_RMT_B4N0_recovered_cvidx.csv

# 4. Evaluate on recovered images
python evaluate_on_recovered.py \
  --model_path ./models_temp/fold_0_best_model.pt \
  --recovered_csv Dataset1_RMT_B4N0_recovered_cvidx.csv \
  --n_classes 2
```

---

## Detailed Command Reference

### Encryption Module Commands

| Command | Purpose |
|---------|---------|
| `python find-dupes.py` | Detect and remove duplicate images |
| `python main.py --method RMT` | Encrypt images with Random Matrix Theory |
| `python main.py --method AES` | Encrypt images with AES |
| `python main.py --method NeuraCrypt` | Encrypt images with NeuraCrypt |
| `python main.py --attack True` | Perform cryptographic attack |
| `python cryp.py` | Core encryption algorithms |
| `python img-size.py` | Analyze/adjust image dimensions |

### Classification Module Commands

| Command | Purpose |
|---------|---------|
| `python prepare-cv.py` | Create stratified cross-validation splits |
| `python classify.py` | Train classification model |
| `python neura.py` | Train NeuraCrypt classifier |
| `python prepare_recovered_csv.py` | Prepare CSV for recovered images |
| `python evaluate_on_recovered.py` | Evaluate model on recovered dataset |

### Segmentation Module Commands

| Command | Purpose |
|---------|---------|
| `python prepare-cv-seg.py` | Create segmentation dataset splits |
| `python segmentation.py` | Train segmentation model |
| `python neura-seg.py` | Train NeuraCrypt segmentation model |

---

## Project Structure

```
Secure-By-Disguise/
├── README.md                      # This file
│
├── encryption/                    # Encryption methods and attacks
│   ├── main.py                    # Main encryption/attack pipeline
│   ├── cryp.py                    # RMT and AES implementations
│   ├── find-dupes.py              # Duplicate detection utility
│   ├── img-size.py                # Image dimension utility
│   ├── requirements.txt           # Python dependencies
│   └── Neuracrypt.py              # NeuraCrypt implementation
│
├── classification/                # Classification models and evaluation
│   ├── classify.py                # Classification training script
│   ├── neura.py                   # NeuraCrypt classifier
│   ├── prepare-cv.py              # Dataset preparation
│   ├── prepare_recovered_csv.py    # Recovered dataset preparation
│   ├── evaluate_on_recovered.py    # Recovery robustness evaluation
│   ├── exp.sh                     # Example experiments script
│   └── exp-rmt-attacks.sh         # RMT attack experiments
│
├── segmentation/                  # Segmentation models
│   ├── segmentation.py            # Segmentation training script
│   ├── neura-seg.py               # NeuraCrypt segmentation
│   ├── prepare-cv-seg.py          # Segmentation dataset preparation
│   └── exp-seg.sh                 # Segmentation experiments
│
└── Data/                          # (Create this) Data directory
    ├── dataset1/
    ├── dataset2/
    ├── dataset3/
    └── dataset4/
```

---

## Typical Research Workflow

### Full Pipeline Example: Evaluating RMT Encryption on Classification

```bash
# 1. Setup
cd Secure-By-Disguise
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r encryption/requirements.txt

# 2. Data Preparation
cd encryption
python find-dupes.py \
  --target_dir ../datasets/dataset1 \
  --output_dir ../datasets/dataset1_clean

# 3. Encryption
python main.py \
  --input ../datasets/dataset1_clean \
  --output ../ModelData/Dataset1/RMT-B4N0 \
  --method RMT \
  --block_size 4

# 4. Classification Training
cd ../classification
python prepare-cv.py ../ModelData/Dataset1/RMT-B4N0 dataset1_rmt_cv.csv
python classify.py \
  --model_name vgg16 \
  --cvidx dataset1_rmt_cv.csv \
  --nfolds 5 \
  --n_classes 2 >> results_vgg16.log

# 5. Attack and Recovery
cd ../encryption
python main.py \
  --input ../datasets/dataset1_clean \
  --output ../recovered/RMT_B4N0_recovered \
  --method RMT \
  --block_size 4 \
  --attack True \
  --encrypted_dir ../ModelData/Dataset1/RMT-B4N0 \
  --known_pairs 10

# 6. Evaluation on Recovered Images
cd ../classification
python prepare_recovered_csv.py \
  --base_csv dataset1_rmt_cv.csv \
  --encrypted_dir ../ModelData/Dataset1/RMT-B4N0 \
  --recovered_dir ../recovered/RMT_B4N0_recovered \
  --output_csv dataset1_rmt_recovered_cv.csv

python evaluate_on_recovered.py \
  --model_path ./models_temp/fold_0_best_model.pt \
  --recovered_csv dataset1_rmt_recovered_cv.csv \
  --n_classes 2
```

---

## References

For more information on the encryption methods and deep learning-based attacks, please refer to:

- DisguisedNets: Secure Image Outsourcing for Confidential Model Training in Clouds
- NeuraCrypt: Hiding Private Health Data via Random Neural Networks for Public Training

---

## License and Citation

If you use this framework in your research, please cite:

```bibtex
@software{securebydisguise2024,
  title={Secure-By-Disguise: Clinical Validation of Image Disguising Methods for Confidential Medical Image Modeling},
  author={Jason Rojas, Jiajie He, Yash Patel Yuechun Gu, Zeyun Yu, Keke Chen},
  year={2026},
  url={https://github.com/Jasonmix84/Secure-By-Disguise}
}
```

---

## Support and Questions

For issues, questions, or contributions, please open an issue on the GitHub repository or contact the authors.

**Last Updated**: April 2026