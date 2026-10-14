import os
import sys
import copy
import time
import random
from collections import Counter
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

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
SEVERITY_CLASSES = [
    "Level 0 (Mild)",
    "Level 1 (Moderate)",
    "Level 2 (Severe)",
    "Level 3 (Very Severe)"
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_enhanced_train_transforms():
    """Anatomically valid facial data augmentation (No non-physiological vertical flips)"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
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
    plt.colorbar(fraction=0.046, pad=0.04)
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
    print(f"[✓] Saved Stage 2 confusion matrix to: {output_paths[0]}")

def evaluate_loader(model, loader, device, use_hflip=False):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            if use_hflip:
                x_h = torch.flip(x, dims=[3])
                out = (out + model(x_h)) / 2.0
            preds = out.argmax(1)
            correct += (preds == y).sum().item()
            total += x.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    acc = correct / total if total > 0 else 0
    return acc, np.array(all_targets), np.array(all_preds)

def main():
    print("=" * 75)
    print("🌟 DERMASENSE STAGE 2: ACNE SEVERITY RETRAINING & RIGOROUS EVALUATION")
    print("Nomenclature: Level 0 (Mild), Level 1 (Moderate), Level 2 (Severe), Level 3 (Very Severe)")
    print("=" * 75)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🖥️  Compute Device: Apple Silicon GPU via Metal Performance Shaders (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("🖥️  Compute Device: NVIDIA CUDA GPU")
    else:
        device = torch.device("cpu")
        print("🖥️  Compute Device: CPU")

    # Index ACNE04 Dataset
    print(f"\n📂 Indexing ACNE04 clinical images from: {ACNE_DIR}")
    samples = []
    for img_name in sorted(os.listdir(ACNE_DIR)):
        if img_name.lower().endswith(".jpg"):
            try:
                sev = int(img_name.split("_")[0].replace("levle", ""))
                samples.append((os.path.join(ACNE_DIR, img_name), sev))
            except Exception:
                continue

    print(f"[✓] Indexed {len(samples)} valid clinical cases.")
    counts = Counter([s[1] for s in samples])
    for lvl in range(4):
        cnt = counts[lvl]
        print(f"  - {SEVERITY_CLASSES[lvl]}: {cnt} images ({cnt/len(samples)*100:.1f}%)")

    # Stratified Split (70% Train, 15% Val, 15% Test) with fixed seed 42
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

    # Balanced Sampler for Imbalance Handling
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

    # Model & Loss
    print("\n🏗️  Initializing EfficientNet-B0 with MedicalFocalLoss...")
    model = build_efficientnet(num_classes=4, pretrained=True).to(device)

    weights_tensor = torch.tensor([1.4, 1.0, 1.0, 1.1], device=device)
    criterion = MedicalFocalLoss(alpha=weights_tensor, gamma=2.0)

    optimizer = optim.AdamW([
        {'params': model.features.parameters(), 'lr': 8e-5},
        {'params': model.classifier.parameters(), 'lr': 8e-4}
    ], weight_decay=1e-4)

    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, min_lr=1e-6)

    epochs = 25
    patience = 6
    best_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())
    no_improve = 0

    print(f"\n🚀 Starting Training Loop on {device} (Max Epochs={epochs}, Patience={patience})...\n")
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

        # Validation (Standard forward pass during training checkpoints)
        v_acc, _, _ = evaluate_loader(model, val_loader, device, use_hflip=False)
        scheduler.step(v_acc)

        ep_duration = time.time() - ep_start
        print(f"Epoch [{ep:02d}/{epochs}] ({ep_duration:4.1f}s) | Train Loss: {tr_loss:.4f} Acc: {tr_acc*100:.2f}% | Val Acc: {v_acc*100:.2f}%", end="")

        if v_acc > best_acc:
            best_acc = v_acc
            best_weights = copy.deepcopy(model.state_dict())
            no_improve = 0
            print("  --> Best Model Updated! 🌟")
        else:
            no_improve += 1
            print()
            if no_improve >= patience:
                print(f"\n[!] Early stopping criteria reached at epoch {ep}. Best Val Acc: {best_acc*100:.2f}%")
                break

    train_duration = (time.time() - start_time) / 60
    print(f"\n[✓] Stage 2 Training Completed in {train_duration:.2f} minutes.")
    print(f"[✓] Peak Validation Accuracy: {best_acc*100:.2f}%")

    model.load_state_dict(best_weights)

    # 1. Validation Set Decision: Select Inference Strategy ONLY on Validation Set!
    print("\n" + "=" * 75)
    print("🔬 VALIDATION-DRIVEN INFERENCE STRATEGY SELECTION")
    print("=" * 75)
    val_acc_std, _, _ = evaluate_loader(model, val_loader, device, use_hflip=False)
    val_acc_hflip, _, _ = evaluate_loader(model, val_loader, device, use_hflip=True)

    print(f"  • Validation Standard Single-Pass Accuracy:        {val_acc_std*100:.2f}%")
    print(f"  • Validation Horizontal-Flip Ensemble Accuracy:   {val_acc_hflip*100:.2f}%")

    use_hflip_selected = (val_acc_hflip > val_acc_std)
    selected_name = "Horizontal-Flip TTA" if use_hflip_selected else "Standard Single-Pass"
    print(f"[✓] Selected Inference Strategy for Final Test: {selected_name}")

    # 2. Final Evaluation on Held-Out Test Set (224 cases) using ONLY selected strategy
    print("\n" + "=" * 75)
    print("📊 FINAL INDEPENDENT TEST EVALUATION (HELD-OUT N=224)")
    print(f"Inference Mode: {selected_name}")
    print("=" * 75)

    test_acc, y_true, y_pred = evaluate_loader(model, test_loader, device, use_hflip=use_hflip_selected)

    cm = np.zeros((4, 4), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    # Adjacent Accuracy (Within +-1 Severity Grade)
    adjacent_correct = np.sum(np.abs(y_true - y_pred) <= 1)
    acc_adjacent = adjacent_correct / len(y_true)

    # Binary Acne Severity Grouping (Mild/Moderate [0,1] vs Severe/Very Severe [2,3])
    # Mild = 0 + 1 (173), Severe = 2 + 3 (51)
    true_severe = (y_true >= 2)
    pred_severe = (y_pred >= 2)
    binary_correct = np.sum(true_severe == pred_severe)
    acc_binary_grouping = binary_correct / len(y_true)

    tp_sev = np.sum((true_severe == 1) & (pred_severe == 1))
    fn_sev = np.sum((true_severe == 1) & (pred_severe == 0))
    fp_sev = np.sum((true_severe == 0) & (pred_severe == 1))
    tn_sev = np.sum((true_severe == 0) & (pred_severe == 0))

    sens_sev = tp_sev / (tp_sev + fn_sev) if (tp_sev + fn_sev) > 0 else 0
    spec_sev = tn_sev / (tn_sev + fp_sev) if (tn_sev + fp_sev) > 0 else 0
    prec_sev = tp_sev / (tp_sev + fp_sev) if (tp_sev + fp_sev) > 0 else 0

    print(f"  • Exact 4-Class Test Accuracy:                            {test_acc*100:.2f}% ({np.sum(y_true == y_pred)} / {len(y_true)})")
    print(f"  • Ordinal Adjacent Accuracy (±1 Grade):                   {acc_adjacent*100:.2f}% ({adjacent_correct} / {len(y_true)})")
    print(f"  • Binary Severity Grouping (Mild/Mod vs Severe/Very Sev): {acc_binary_grouping*100:.2f}% ({binary_correct} / {len(y_true)})")
    print(f"    ├─ Severe Case Detection Sensitivity:                   {sens_sev*100:.2f}% ({tp_sev} / {tp_sev+fn_sev})")
    print(f"    ├─ Mild/Mod Case Specificity:                           {spec_sev*100:.2f}% ({tn_sev} / {tn_sev+fp_sev})")
    print(f"    ├─ Severe Positive Predictive Value (Precision):        {prec_sev*100:.2f}%")
    print(f"    └─ False Negatives (Severe cases missed):               {fn_sev} cases")

    print("\nClass-wise Clinical Metrics:")
    f1_list = []
    report = {}
    for i in range(4):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = len(y_true) - tp - fn - fp
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        f1_list.append(f1)
        print(f"  {SEVERITY_CLASSES[i]:25s} | Support: {tp+fn:2d} | Sens/Recall: {rec*100:5.1f}% | Precision: {prec*100:5.1f}% | Specificity: {spec*100:5.1f}% | F1: {f1:.3f}")
        report[SEVERITY_CLASSES[i]] = {
            "support": int(tp + fn),
            "precision": round(float(prec), 3),
            "recall": round(float(rec), 3),
            "specificity": round(float(spec), 3),
            "f1_score": round(float(f1), 3)
        }

    macro_f1 = float(np.mean(f1_list))
    print(f"\nMacro Average F1-Score: {macro_f1:.3f}")

    # Save weights
    os.makedirs("weights", exist_ok=True)
    save_path = "weights/efficientnet_s2_weights.pth"
    torch.save(best_weights, save_path)
    print(f"\n💾 Saved Stage 2 retrained weights to: {save_path}")

    # Save Confusion Matrix Plot
    save_confusion_matrix(
        cm,
        SEVERITY_CLASSES,
        f"Stage 2: Acne Severity Confusion Matrix ({selected_name} Acc: {test_acc*100:.1f}%)",
        ["results/confusion_matrix_stage2.png", "app/static/results/confusion_matrix_stage2.png"]
    )

    return test_acc, macro_f1, acc_binary_grouping, report, cm

if __name__ == "__main__":
    main()
