import os
import sys

# Ensure root directory is always on Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import copy
import time
import random
from collections import Counter
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

from core.models import build_efficientnet
from training.loss_functions import MedicalFocalLoss

ACNE_DIR = "/Users/bariqa/.cache/kagglehub/datasets/jincyjis/acne04/versions/1/JPEGImages"
SEVERITY_CLASSES = ["Level 0 (Clear)", "Level 1 (Mild)", "Level 2 (Moderate)", "Level 3 (Severe)"]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_enhanced_train_transforms():
    """Enhanced data augmentation focusing on lesion morphology & erythema (redness)"""
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.15, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

def get_eval_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

class PureAcneDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                if self.transform:
                    img = self.transform(img)
        except Exception:
            img = Image.new("RGB", (224, 224), color=(128, 128, 128))
            if self.transform:
                img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)

def save_confusion_matrix(cm, classes, title, output_paths):
    plt.figure(figsize=(7, 6), dpi=300)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title, fontsize=12, fontweight='bold', pad=12)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=25, ha='right', fontsize=9)
    plt.yticks(tick_marks, classes, fontsize=9)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            plt.text(j, i, f"{int(val)}",
                     horizontalalignment="center",
                     verticalalignment="center",
                     color="white" if val > thresh else "black",
                     fontweight='bold', fontsize=10)

    plt.ylabel('Ground Truth Severity', fontsize=10, fontweight='600')
    plt.xlabel('Predicted Severity', fontsize=10, fontweight='600')
    plt.tight_layout()
    for p in output_paths:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        plt.savefig(p, bbox_inches='tight')
    plt.close()
    print(f"[✓] Saved updated confusion matrix plot to {output_paths[0]}")

def main():
    print("=" * 70)
    print("STAGE 2 (ACNE SEVERITY) ADVANCED CLINICAL RETRAINING")
    print("Focus: MedicalFocalLoss + Lesion Augmentations + TTA Evaluation")
    print("=" * 70)

    # 1. Device Selection (Metal Performance Shaders on Apple Silicon)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Compute Device: Apple Silicon GPU via Metal Performance Shaders (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Compute Device: NVIDIA CUDA GPU")
    else:
        device = torch.device("cpu")
        print("Compute Device: CPU")

    # 2. Index Dataset
    print(f"\nScanning images in: {ACNE_DIR}")
    samples = []
    for img_name in sorted(os.listdir(ACNE_DIR)):
        if img_name.lower().endswith(".jpg"):
            try:
                sev = int(img_name.split("_")[0].replace("levle", ""))
                samples.append((os.path.join(ACNE_DIR, img_name), sev))
            except Exception:
                continue

    print(f"[✓] Indexed {len(samples)} valid clinical images.")
    counts = Counter([s[1] for s in samples])
    for lvl in range(4):
        cnt = counts[lvl]
        print(f"  - {SEVERITY_CLASSES[lvl]}: {cnt} images ({cnt/len(samples)*100:.1f}%)")

    # 3. Stratified Split (70% Train, 15% Val, 15% Test)
    random.seed(42)
    per_class = {i: [] for i in range(4)}
    for s in samples:
        per_class[s[1]].append(s)

    train_samples, val_samples, test_samples = [], [], []
    for i in range(4):
        c_list = per_class[i]
        random.shuffle(c_list)
        n = len(c_list)
        n_train = int(0.70 * n)
        n_val = int(0.15 * n)
        train_samples.extend(c_list[:n_train])
        val_samples.extend(c_list[n_train:n_train + n_val])
        test_samples.extend(c_list[n_train + n_val:])

    print(f"\nPartitioning: Train={len(train_samples)} | Val={len(val_samples)} | Test={len(test_samples)}")

    # 4. Balanced Sampler
    train_labels = [s[1] for s in train_samples]
    tr_counts = Counter(train_labels)
    class_weights = {cls: 1.0 / tr_counts[cls] for cls in range(4)}
    sample_weights = [class_weights[l] for l in train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(
        PureAcneDataset(train_samples, transform=get_enhanced_train_transforms()),
        batch_size=32, sampler=sampler, num_workers=0
    )
    val_loader = DataLoader(
        PureAcneDataset(val_samples, transform=get_eval_transforms()),
        batch_size=32, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        PureAcneDataset(test_samples, transform=get_eval_transforms()),
        batch_size=32, shuffle=False, num_workers=0
    )

    # 5. Model Architecture & Loss
    print("\nInitializing EfficientNet-B0 backbone with MedicalFocalLoss...")
    model = build_efficientnet(num_classes=4, pretrained=True).to(device)

    # Class-weighted Medical Focal Loss to penalize under-represented and hard borderline cases
    weights_tensor = torch.tensor([1.4, 1.0, 1.0, 1.1], device=device)
    criterion = MedicalFocalLoss(alpha=weights_tensor, gamma=2.0)

    optimizer = optim.AdamW([
        {'params': model.features.parameters(), 'lr': 8e-5},
        {'params': model.classifier.parameters(), 'lr': 8e-4}
    ], weight_decay=1e-4)

    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, min_lr=1e-6)

    # 6. Training Loop (25 Epochs with Early Stopping)
    epochs = 25
    patience = 7
    best_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())
    no_improve = 0

    print(f"\nStarting Advanced Training Loop on {device}...")
    start_time = time.time()

    for ep in range(1, epochs + 1):
        ep_start = time.time()
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            tr_loss += loss.item() * x.size(0)
            tr_correct += (out.argmax(1) == y).sum().item()
            tr_total += x.size(0)

        tr_acc = tr_correct / tr_total
        tr_loss = tr_loss / tr_total

        # Validation
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                v_loss += loss.item() * x.size(0)
                v_correct += (out.argmax(1) == y).sum().item()
                v_total += x.size(0)

        v_acc = v_correct / v_total
        v_loss = v_loss / v_total
        scheduler.step(v_acc)

        ep_duration = time.time() - ep_start
        print(f"Epoch [{ep:02d}/{epochs}] ({ep_duration:4.1f}s) | Train Loss: {tr_loss:.4f} Acc: {tr_acc*100:.2f}% | Val Loss: {v_loss:.4f} Acc: {v_acc*100:.2f}%", end="")

        if v_acc > best_acc:
            best_acc = v_acc
            best_weights = copy.deepcopy(model.state_dict())
            no_improve = 0
            print("  --> Best Model Updated!")
        else:
            no_improve += 1
            print()
            if no_improve >= patience:
                print(f"\n[!] Early stopping criteria reached at epoch {ep}. Best Val Acc: {best_acc*100:.2f}%")
                break

    total_duration = (time.time() - start_time) / 60
    print(f"\n[✓] Training Completed in {total_duration:.2f} minutes.")
    print(f"[✓] Peak Validation Accuracy: {best_acc*100:.2f}%")

    # 7. Final Test Evaluation with TTA (Test-Time Augmentation)
    model.load_state_dict(best_weights)
    model.eval()
    
    test_correct_std = 0
    test_correct_tta = 0
    test_adjacent_correct = 0
    test_binary_triage_correct = 0
    test_total = 0
    cm_tta = np.zeros((4, 4), dtype=int)

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            # Standard single pass
            out_std = model(x)
            preds_std = out_std.argmax(1)
            test_correct_std += (preds_std == y).sum().item()

            # TTA: original, horizontal flip, vertical flip
            x_h = torch.flip(x, dims=[3])
            x_v = torch.flip(x, dims=[2])
            out_tta = (out_std + model(x_h) + model(x_v)) / 3.0
            preds_tta = out_tta.argmax(1)

            test_correct_tta += (preds_tta == y).sum().item()
            test_adjacent_correct += ((preds_tta - y).abs() <= 1).sum().item()
            test_binary_triage_correct += ((preds_tta >= 2) == (y >= 2)).sum().item()
            test_total += x.size(0)

            for t, p in zip(y.view(-1).cpu().numpy(), preds_tta.view(-1).cpu().numpy()):
                cm_tta[t, p] += 1

    acc_std = test_correct_std / test_total
    acc_tta = test_correct_tta / test_total
    acc_adjacent = test_adjacent_correct / test_total
    acc_triage = test_binary_triage_correct / test_total

    print("\n" + "=" * 70)
    print("FINAL HELD-OUT TEST EVALUATION (ACNE04)")
    print("=" * 70)
    print(f"  • Standard Exact Test Accuracy:  {acc_std*100:.2f}%")
    print(f"  • TTA-Enhanced Test Accuracy:    {acc_tta*100:.2f}%")
    print(f"  • Adjacent Accuracy (±1 Grade):  {acc_adjacent*100:.2f}% (Clinical Ordinal Reliability)")
    print(f"  • Binary Triage (Mild vs Severe):{acc_triage*100:.2f}% (Actionable Care Triage)")
    print("=" * 70)

    print("\nConfusion Matrix (TTA Enhanced):")
    for i in range(4):
        print(f"  {SEVERITY_CLASSES[i]:25s}: {cm_tta[i].tolist()}")

    # Class-wise metrics
    print("\nClass-wise Clinical Metrics:")
    for i in range(4):
        tp = cm_tta[i, i]
        fp = cm_tta[:, i].sum() - tp
        fn = cm_tta[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        print(f"  {SEVERITY_CLASSES[i]:25s} | Precision: {prec*100:.1f}% | Recall: {rec*100:.1f}% | F1: {f1:.3f}")

    # Save weights locally
    os.makedirs("weights", exist_ok=True)
    save_path = "weights/efficientnet_s2_weights.pth"
    torch.save(best_weights, save_path)
    print(f"\n[✓] Trained model weights saved to: {save_path}")

    # Save confusion matrix plot
    save_confusion_matrix(
        cm_tta,
        SEVERITY_CLASSES,
        f"Stage 2: Acne Severity Confusion Matrix (TTA Acc: {acc_tta*100:.1f}%)",
        ['results/confusion_matrix_stage2.png', 'app/static/results/confusion_matrix_stage2.png']
    )

if __name__ == "__main__":
    main()
