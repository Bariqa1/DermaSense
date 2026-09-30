import os
import sys
import copy
import json
import time
import random
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

import kagglehub

from core.models import build_efficientnet
from core.transforms import get_train_transforms_s1, get_train_transforms_s2, get_val_transforms
from core.pipeline import STAGE1_CLASSES
from training.dataset import SkinDataset
from training.loss_functions import MedicalFocalLoss
from training.evaluate import (
    generate_confusion_matrix,
    generate_comparative_bar_chart,
    SEVERITY_CLASSES
)

SEED = 42
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

def get_compute_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def acquire_datasets():
    print("=" * 70)
    print("📥 STEP 1: Verifying & Acquiring Clinical Datasets (Kagglehub)...")
    print("=" * 70)
    print("Fetching Acne dataset (jincyjis/acne04)...")
    path_acne = kagglehub.dataset_download("jincyjis/acne04")
    print(f"[✓] Acne dataset located at: {path_acne}")

    print("\nFetching Multi-disease dataset (ismailpromus/skin-diseases-image-dataset)...")
    path_skin = kagglehub.dataset_download("ismailpromus/skin-diseases-image-dataset")
    print(f"[✓] Skin disease dataset located at: {path_skin}")

    return path_skin, path_acne

def prepare_dataframes(path_skin, path_acne):
    print("\n" + "=" * 70)
    print("📊 STEP 2: Indexing & Building Clinical Cohorts...")
    print("=" * 70)

    SKIN_BASE = os.path.join(path_skin, "IMG_CLASSES")
    ACNE_BASE = path_acne
    for f in ["JPEGImages", "acne04", ".", "Images"]:
        t = os.path.join(path_acne, f)
        if os.path.exists(t) and len([x for x in os.listdir(t) if x.lower().endswith(".jpg")]) > 10:
            ACNE_BASE = t
            break

    # Stage 1 Dataset
    rows_s1 = []
    TARGETS = {
        "1. Eczema": "Eczema", "2. Melanoma": "Melanoma", "3. Atopic": "Eczema",
        "4. Basal": "Basal Cell Carcinoma", "5. Melanocytic": "Melanocytic Nevi",
        "6. Benign": "Benign Keratosis", "7. Psoriasis": "Psoriasis",
        "8. Seborrheic": "Seborrheic Keratoses"
    }
    MAX_IMAGES = 1600

    if os.path.exists(SKIN_BASE):
        for d in os.listdir(SKIN_BASE):
            label = next((v for k, v in TARGETS.items() if d.startswith(k)), None)
            if label:
                folder = os.path.join(SKIN_BASE, d)
                imgs = [x for x in os.listdir(folder) if x.lower().endswith((".jpg", ".png"))][:MAX_IMAGES]
                for img in imgs:
                    rows_s1.append({"path": os.path.join(folder, img), "label": label})

    # Add Acne & Clear cohorts
    if os.path.exists(ACNE_BASE):
        for img in [x for x in os.listdir(ACNE_BASE) if x.lower().endswith(".jpg")]:
            try:
                sev = int(img.split("_")[0].replace("levle", ""))
                label = "Clear_AlmostClear" if sev == 0 else "Acne"
                rows_s1.append({"path": os.path.join(ACNE_BASE, img), "label": label})
            except Exception:
                continue

    df_s1 = shuffle(pd.DataFrame(rows_s1), random_state=SEED).reset_index(drop=True)
    print(f"Stage 1 Cohort Total: {len(df_s1)} images across {df_s1['label'].nunique()} classes.")

    # Stage 2 Severity Dataset
    rows_s2 = []
    if os.path.exists(ACNE_BASE):
        for img in [x for x in os.listdir(ACNE_BASE) if x.lower().endswith(".jpg")]:
            try:
                sev = int(img.split("_")[0].replace("levle", ""))
                rows_s2.append({"path": os.path.join(ACNE_BASE, img), "label": sev})
            except Exception:
                continue

    df_s2 = shuffle(pd.DataFrame(rows_s2), random_state=SEED).reset_index(drop=True)
    print(f"Stage 2 Acne Severity Cohort Total: {len(df_s2)} images across 4 levels.")

    return df_s1, df_s2

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data).item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data).item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    loss = running_loss / total
    acc = correct / total
    return loss, acc, np.array(all_labels), np.array(all_preds)

def train_stage(stage_num, df, num_classes, class_names, device, epochs=25, patience=5):
    print("\n" + "=" * 70)
    print(f"🚀 STEP 3.{stage_num}: Training Stage {stage_num} ({num_classes} Classes) on {device} (Epochs={epochs})...")
    print("=" * 70)

    # Data split
    train_df, test_df = train_test_split(df, test_size=0.15, stratify=df["label"], random_state=SEED)
    train_df, val_df = train_test_split(train_df, test_size=0.1765, stratify=train_df["label"], random_state=SEED)

    train_trans = get_train_transforms_s1() if stage_num == 1 else get_train_transforms_s2()
    val_trans = get_val_transforms()

    train_ds = SkinDataset(train_df, transform=train_trans, stage=stage_num)
    val_ds = SkinDataset(val_df, transform=val_trans, stage=stage_num)
    test_ds = SkinDataset(test_df, transform=val_trans, stage=stage_num)

    sampler = None
    if stage_num == 2:
        class_counts = train_df["label"].value_counts().sort_index().values
        weights = 1.0 / class_counts
        sample_weights = [weights[int(l)] for l in train_df["label"].values]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    batch_size = 32
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=(sampler is None), sampler=sampler, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = build_efficientnet(num_classes=num_classes, pretrained=True).to(device)

    # Medical loss: Focal Loss for Stage 1 to guard Melanoma, label-smoothing CrossEntropy for Stage 2
    if stage_num == 1:
        criterion = MedicalFocalLoss(gamma=2.0)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = optim.AdamW([
        {'params': model.features.parameters(), 'lr': 1e-4},
        {'params': model.classifier.parameters(), 'lr': 1e-3}
    ], weight_decay=1e-4)

    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    best_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())
    no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    start_time = time.time()
    for ep in range(1, epochs + 1):
        ep_start = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_acc, _, _ = evaluate_model(model, val_loader, criterion, device)
        scheduler.step(v_acc)

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(v_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(v_acc)

        ep_duration = time.time() - ep_start
        print(f"Epoch [{ep:02d}/{epochs}] ({ep_duration:.1f}s) | Train Loss: {tr_loss:.4f} Acc: {tr_acc*100:.2f}% | Val Loss: {v_loss:.4f} Acc: {v_acc*100:.2f}%")

        if v_acc > best_acc:
            best_acc = v_acc
            best_weights = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping triggered at epoch {ep}. Best Val Acc: {best_acc*100:.2f}%")
                break

    total_time = (time.time() - start_time) / 60
    print(f"Training completed in {total_time:.1f} minutes. Best Validation Accuracy: {best_acc*100:.2f}%")

    model.load_state_dict(best_weights)
    save_path = f"weights/efficientnet_s{stage_num}_trained.pth"
    torch.save(best_weights, save_path)
    print(f"[✓] Checkpoint saved: {save_path}")

    # Test evaluation
    test_loss, test_acc, y_true, y_pred = evaluate_model(model, test_loader, criterion, device)
    print(f"Test Accuracy (Stage {stage_num}): {test_acc*100:.2f}%")

    return model, history, (y_true, y_pred), test_acc

def run():
    device = get_compute_device()
    print("=" * 70)
    print(f"🌟 DERMASENSE CLINICAL TRAINING PIPELINE (PyTorch MPS / Metal Acceleration)")
    print(f"🖥️  Host Hardware: Apple Silicon Mac Pro (Active Device: {device})")
    print("=" * 70)

    path_skin, path_acne = acquire_datasets()
    df_s1, df_s2 = prepare_dataframes(path_skin, path_acne)

    # Train Stage 1
    model_s1, hist_s1, (y_true_s1, y_pred_s1), acc_s1 = train_stage(
        1, df_s1, len(STAGE1_CLASSES), STAGE1_CLASSES, device, epochs=25, patience=5
    )

    # Train Stage 2
    model_s2, hist_s2, (y_true_s2, y_pred_s2), acc_s2 = train_stage(
        2, df_s2, len(SEVERITY_CLASSES), SEVERITY_CLASSES, device, epochs=25, patience=5
    )

    print("\n" + "=" * 70)
    print("🎉 Full Training Completed Successfully on Apple Silicon (MPS)!")
    print("=" * 70)

if __name__ == "__main__":
    run()
