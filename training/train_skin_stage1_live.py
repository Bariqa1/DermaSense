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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader

from core.models import build_efficientnet
from core.transforms import get_train_transforms_s1, get_val_transforms
from core.pipeline import STAGE1_CLASSES
from training.loss_functions import MedicalFocalLoss

SKIN_BASE = "/Users/bariqa/.cache/kagglehub/datasets/ismailpromus/skin-diseases-image-dataset/versions/1/IMG_CLASSES"
ACNE_BASE = "/Users/bariqa/.cache/kagglehub/datasets/jincyjis/acne04/versions/1/JPEGImages"

TARGETS = {
    "1. Eczema": "Eczema",
    "2. Melanoma": "Melanoma",
    "3. Atopic": "Eczema",
    "4. Basal": "Basal Cell Carcinoma",
    "5. Melanocytic": "Melanocytic Nevi",
    "6. Benign": "Benign Keratosis",
    "7. Psoriasis": "Psoriasis",
    "8. Seborrheic": "Seborrheic Keratoses"
}

class PureSkinDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label_idx = self.samples[idx]
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                if self.transform:
                    img = self.transform(img)
        except Exception:
            img = Image.new("RGB", (224, 224), color=(128, 128, 128))
            if self.transform:
                img = self.transform(img)
        return img, torch.tensor(label_idx, dtype=torch.long)

def main():
    print("=" * 70)
    print("🌟 STAGE 1 (MULTI-DISEASE & MELANOMA) CLINICAL RETRAINING (25 EPOCHS)")
    print("=" * 70)

    # 1. Device Selection (Metal Performance Shaders on Apple Silicon)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🖥️  Compute Device: Apple Silicon GPU via Metal Performance Shaders (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("🖥️  Compute Device: NVIDIA CUDA GPU")
    else:
        device = torch.device("cpu")
        print("🖥️  Compute Device: CPU")

    # 2. Index Dataset
    print(f"\n📂 Scanning images in: {SKIN_BASE} and {ACNE_BASE}")
    class_to_idx = {cls: i for i, cls in enumerate(STAGE1_CLASSES)}
    MAX_PER_CLASS = 600  # Balanced sampling for high clinical stability
    
    per_class_samples = {cls: [] for cls in STAGE1_CLASSES}

    if os.path.exists(SKIN_BASE):
        for d in sorted(os.listdir(SKIN_BASE)):
            label = next((v for k, v in TARGETS.items() if d.startswith(k)), None)
            if label and label in per_class_samples:
                folder = os.path.join(SKIN_BASE, d)
                imgs = [x for x in sorted(os.listdir(folder)) if x.lower().endswith((".jpg", ".png"))]
                for img in imgs:
                    if len(per_class_samples[label]) < MAX_PER_CLASS:
                        per_class_samples[label].append(os.path.join(folder, img))

    if os.path.exists(ACNE_BASE):
        for img in sorted(os.listdir(ACNE_BASE)):
            if img.lower().endswith(".jpg"):
                try:
                    sev = int(img.split("_")[0].replace("levle", ""))
                    label = "Clear_AlmostClear" if sev == 0 else "Acne"
                    if len(per_class_samples[label]) < MAX_PER_CLASS:
                        per_class_samples[label].append(os.path.join(ACNE_BASE, img))
                except Exception:
                    continue

    all_samples = []
    print("\nBalanced Class Distribution:")
    for cls in STAGE1_CLASSES:
        c_imgs = per_class_samples[cls]
        print(f"  - {cls:25s}: {len(c_imgs)} clinical images")
        for p in c_imgs:
            all_samples.append((p, class_to_idx[cls]))

    print(f"\nTotal Dataset: {len(all_samples)} clinical images across 9 classes.")

    # 3. Stratified Split (70% Train, 15% Val, 15% Test) in pure Python
    random.seed(42)
    per_class_grouped = {i: [] for i in range(len(STAGE1_CLASSES))}
    for s in all_samples:
        per_class_grouped[s[1]].append(s)

    train_samples, val_samples, test_samples = [], [], []
    for i in range(len(STAGE1_CLASSES)):
        c_list = per_class_grouped[i]
        random.shuffle(c_list)
        n = len(c_list)
        n_train = int(0.70 * n)
        n_val = int(0.15 * n)
        train_samples.extend(c_list[:n_train])
        val_samples.extend(c_list[n_train:n_train + n_val])
        test_samples.extend(c_list[n_train + n_val:])

    print(f"Partitioning: Train={len(train_samples)} | Val={len(val_samples)} | Test={len(test_samples)}")

    train_loader = DataLoader(
        PureSkinDataset(train_samples, transform=get_train_transforms_s1()),
        batch_size=32, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        PureSkinDataset(val_samples, transform=get_val_transforms()),
        batch_size=32, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        PureSkinDataset(test_samples, transform=get_val_transforms()),
        batch_size=32, shuffle=False, num_workers=0
    )

    # 4. Model Architecture & Medical Loss
    print("\n🏗️  Initializing EfficientNet-B0 with ImageNet pretrained backbone...")
    model = build_efficientnet(num_classes=len(STAGE1_CLASSES), pretrained=True).to(device)

    # Focal Loss: heavily penalizes errors on difficult classes and protects Melanoma
    criterion = MedicalFocalLoss(gamma=2.0)

    optimizer = optim.AdamW([
        {'params': model.features.parameters(), 'lr': 1e-4},
        {'params': model.classifier.parameters(), 'lr': 1e-3}
    ], weight_decay=1e-4)

    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    # 5. Training Loop (25 Epochs)
    epochs = 25
    patience = 6
    best_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())
    no_improve = 0

    print(f"\n🚀 Starting Stage 1 Training Loop on {device} (Epochs={epochs})...")
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
            print("  --> Best Model Updated! 🌟")
        else:
            no_improve += 1
            print()
            if no_improve >= patience:
                print(f"\n[!] Early stopping criteria reached at epoch {ep}. Best Val Acc: {best_acc*100:.2f}%")
                break

    total_duration = (time.time() - start_time) / 60
    print(f"\n[✓] Stage 1 Training Completed in {total_duration:.2f} minutes.")
    print(f"[✓] Peak Validation Accuracy: {best_acc*100:.2f}%")

    # 6. Final Test Evaluation
    model.load_state_dict(best_weights)
    model.eval()
    test_correct, test_total = 0, 0
    cm = torch.zeros(len(STAGE1_CLASSES), len(STAGE1_CLASSES), dtype=torch.int64)

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            preds = out.argmax(1)
            test_correct += (preds == y).sum().item()
            test_total += x.size(0)
            for t, p in zip(y.view(-1), preds.view(-1)):
                cm[t.long(), p.long()] += 1

    final_test_acc = test_correct / test_total
    print("\n" + "=" * 70)
    print(f"📊 FINAL HELD-OUT TEST EVALUATION (Accuracy: {final_test_acc*100:.2f}%)")
    print("=" * 70)

    # Class-wise metrics
    print("\nClass-wise Clinical Metrics:")
    for i in range(len(STAGE1_CLASSES)):
        tp = cm[i, i].item()
        fp = (cm[:, i].sum() - tp).item()
        fn = (cm[i, :].sum() - tp).item()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        star = "⭐" if STAGE1_CLASSES[i] == "Melanoma" else ""
        print(f"  {STAGE1_CLASSES[i]:25s} | Precision: {prec*100:5.1f}% | Recall: {rec*100:5.1f}% | F1: {f1:.3f} {star}")

    # Save weights locally
    os.makedirs("weights", exist_ok=True)
    save_path = "weights/efficientnet_s1_weights.pth"
    torch.save(best_weights, save_path)
    print(f"\n💾 Stage 1 trained weights successfully saved to: {save_path}")

if __name__ == "__main__":
    main()
