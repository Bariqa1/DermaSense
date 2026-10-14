import os
import sys
import copy
import time
import random
import json
import re
from collections import Counter
from PIL import Image

# Ensure root directory is on Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import matplotlib.pyplot as plt
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
CLEAR_BASE = "/Users/bariqa/.cache/kagglehub/datasets/shakyadissanayake/oily-dry-and-normal-skin-types-dataset/versions/1/Oily-Dry-Skin-Types"

TARGET_DISEASES = {
    "Melanoma": "2. Melanoma 15.75k",
    "Basal Cell Carcinoma": "4. Basal Cell Carcinoma (BCC) 3323",
    "Melanocytic Nevi": "5. Melanocytic Nevi (NV) - 7970",
    "Benign Keratosis": "6. Benign Keratosis-like Lesions (BKL) 2624",
    "Seborrheic Keratoses": "8. Seborrheic Keratoses and other Benign Tumors - 1.8k"
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

def save_confusion_matrix_plot(cm, classes, title, output_paths):
    plt.figure(figsize=(9, 8), dpi=300)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title, fontsize=12, fontweight='bold', pad=14)
    plt.colorbar(fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=35, ha='right', fontsize=9)
    plt.yticks(tick_marks, classes, fontsize=9)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            plt.text(j, i, f"{int(val)}",
                     horizontalalignment="center",
                     verticalalignment="center",
                     color="white" if val > thresh else "black",
                     fontweight='bold', fontsize=9)

    plt.ylabel('Ground Truth Label', fontsize=11, fontweight='600')
    plt.xlabel('Predicted Label', fontsize=11, fontweight='600')
    plt.tight_layout()
    for p in output_paths:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        plt.savefig(p, bbox_inches='tight')
    plt.close()
    print(f"[✓] Saved Stage 1 confusion matrix to: {output_paths[0]}")

def build_balanced_cohort(per_class=500, seed=42):
    random.seed(seed)
    cohort = {cls: [] for cls in STAGE1_CLASSES}

    # 1. Clear Skin (Healthy facial normal skin from Oily-Dry-and-Normal-Skin-Types-Dataset)
    clear_imgs = []
    for split in ["train", "valid", "test"]:
        s_dir = os.path.join(CLEAR_BASE, split, "normal")
        if os.path.exists(s_dir):
            for x in sorted(os.listdir(s_dir)):
                if x.lower().endswith(('.jpg', '.png', '.jpeg')):
                    clear_imgs.append(os.path.join(s_dir, x))
    random.shuffle(clear_imgs)
    cohort["Clear Skin"] = clear_imgs[:per_class]

    # 2. Acne Vulgaris (Stratified across all 4 severity levels from acne04)
    acne_by_level = {0: [], 1: [], 2: [], 3: []}
    for x in sorted(os.listdir(ACNE_BASE)):
        if x.lower().endswith(('.jpg', '.png', '.jpeg')):
            try:
                sev = int(x.split('_')[0].replace('levle', ''))
                acne_by_level[sev].append(os.path.join(ACNE_BASE, x))
            except Exception:
                pass

    acne_selected = []
    per_lvl = per_class // 4
    for lvl in range(4):
        l_imgs = acne_by_level[lvl]
        random.shuffle(l_imgs)
        acne_selected.extend(l_imgs[:per_lvl])

    if len(acne_selected) < per_class:
        sel_set = set(acne_selected)
        remaining = [p for lvl in range(4) for p in acne_by_level[lvl] if p not in sel_set]
        random.shuffle(remaining)
        acne_selected.extend(remaining[:per_class - len(acne_selected)])

    cohort["Acne"] = acne_selected[:per_class]

    # 3. Pure Mirror-Compatible Psoriasis (Option A: Filter out Lichen Planus, Lichen Sclerosus, Pityriasis Rosea, Genitals, Nails, Palms/Soles)
    p7_dir = os.path.join(SKIN_BASE, "7. Psoriasis pictures Lichen Planus and related diseases - 2k")
    pso_incompatible_sites = [
        'palms', 'soles', 'palm', 'sole', 'foot', 'feet', 'hand', 'hands',
        'nail', 'nails', 'onycho', 'oilspot',
        'penis', 'anus', 'vulva', 'groin', 'perianal', 'gluteal'
    ]
    face_keywords = ['face', 'facial', 'cheek', 'forehead', 'eyelid', 'ear', 'lip', 'nose', 'scalp', 'neck']

    pso_face = []
    pso_macro = []
    for x in sorted(os.listdir(p7_dir)):
        if not x.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue
        lower = x.lower()
        # Verify diagnosis is true Psoriasis (exclude Lichen, Pityriasis, Seborrheic, Reiter)
        if 'psoriasis' in lower and not any(k in lower for k in ['lichen', 'pityriasis', 'sebderm', 'seborrheic', 'reiter', 'biteline']):
            # Strictly exclude anatomical sites incompatible with facial smart mirror
            if any(k in lower for k in pso_incompatible_sites):
                continue
            p = os.path.join(p7_dir, x)
            if any(k in lower for k in face_keywords):
                pso_face.append(p)
            else:
                pso_macro.append(p)

    random.shuffle(pso_face)
    random.shuffle(pso_macro)
    # Face prioritized first, then high-fidelity macro plaque close-ups
    combined_pso = pso_face + pso_macro
    cohort["Psoriasis"] = combined_pso[:per_class]

    # 4. Pure Mirror-Compatible Eczema & Atopic Dermatitis (Option A: Filter out Ichthyosis, Keratosis Pilaris, Excoriations, Stasis Ulcers, Palms/Soles)
    ecz_dir = os.path.join(SKIN_BASE, "1. Eczema 1677")
    atopic_dir = os.path.join(SKIN_BASE, "3. Atopic Dermatitis - 1.25k")
    ecz_excluded_diagnoses = [
        'ichthyosis', 'icthyosis', 'keratosis', 'excoriation', 'scratch', 'prurigo',
        'keratolysis', 'heels', 'fissure', 'milroy', 'chapped', 'lichen'
    ]
    ecz_incompatible_sites = [
        'stasis', 'leg', 'lower-leg', 'ankle', 'palms', 'soles', 'palm', 'sole',
        'foot', 'feet', 'hand', 'hands', 'finger', 'fingers', 'toe', 'toes'
    ]

    ecz_face = []
    ecz_macro = []
    for fld in [ecz_dir, atopic_dir]:
        for x in sorted(os.listdir(fld)):
            if not x.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue
            if re.match(r'^\d+_\d+\.jpg$', x):
                continue
            lower = x.lower()
            if any(k in lower for k in ecz_excluded_diagnoses):
                continue
            if any(k in lower for k in ['eczema', 'atopic', 'dermatitis']):
                if any(k in lower for k in ecz_incompatible_sites):
                    continue
                p = os.path.join(fld, x)
                if any(k in lower for k in face_keywords):
                    ecz_face.append(p)
                else:
                    ecz_macro.append(p)

    random.shuffle(ecz_face)
    random.shuffle(ecz_macro)
    combined_ecz = ecz_face + ecz_macro
    cohort["Eczema"] = combined_ecz[:per_class]

    # 5. Remaining Dermatological Conditions
    for cls_name, fld in TARGET_DISEASES.items():
        f_path = os.path.join(SKIN_BASE, fld)
        imgs = [os.path.join(f_path, x) for x in sorted(os.listdir(f_path)) if x.lower().endswith(('.jpg', '.png', '.jpeg'))]
        random.shuffle(imgs)
        cohort[cls_name] = imgs[:per_class]

    return cohort

def main():
    print("=" * 75)
    print("🌟 DERMASENSE STAGE 1: MULTI-DISEASE CLINICAL RETRAINING (CORRECTED COHORT)")
    print("Features: Real Clear Skin + Stratified 4-Level Acne + Balanced Eczema/Atopic")
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

    # Build Cohort
    print("\n📂 Building balanced clinical cohorts (500 per class across 9 classes, Total=4,500)...")
    cohort = build_balanced_cohort(per_class=500, seed=42)

    class_to_idx = {cls: i for i, cls in enumerate(STAGE1_CLASSES)}
    all_samples = []
    for cls in STAGE1_CLASSES:
        imgs = cohort[cls]
        print(f"  - {cls:25s}: {len(imgs)} clinical images")
        for p in imgs:
            all_samples.append((p, class_to_idx[cls]))

    print(f"\nTotal Dataset: {len(all_samples)} clinical images.")

    # 70% Train, 15% Val, 15% Test Stratified Split (350 Train / 75 Val / 75 Test per class)
    random.seed(42)
    per_class_grouped = {i: [] for i in range(len(STAGE1_CLASSES))}
    for s in all_samples:
        per_class_grouped[s[1]].append(s)

    train_samples, val_samples, test_samples = [], [], []
    for i in range(len(STAGE1_CLASSES)):
        c_list = per_class_grouped[i]
        random.shuffle(c_list)
        n = len(c_list)
        n_train = int(0.70 * n)  # 350
        n_val = int(0.15 * n)    # 75
        train_samples.extend(c_list[:n_train])
        val_samples.extend(c_list[n_train:n_train + n_val])
        test_samples.extend(c_list[n_train + n_val:])  # 75

    print(f"\nPartitioning: Train={len(train_samples)} (350/cls) | Val={len(val_samples)} (75/cls) | Test={len(test_samples)} (75/cls)")

    # DataLoaders
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

    # Model & Loss
    print("\n🏗️  Initializing EfficientNet-B0 with ImageNet pretrained backbone...")
    model = build_efficientnet(num_classes=len(STAGE1_CLASSES), pretrained=True).to(device)
    criterion = MedicalFocalLoss(gamma=2.0)

    optimizer = optim.AdamW([
        {'params': model.features.parameters(), 'lr': 1e-4},
        {'params': model.classifier.parameters(), 'lr': 1e-3}
    ], weight_decay=1e-4)

    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    epochs = 20
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

    train_duration = (time.time() - start_time) / 60
    print(f"\n[✓] Stage 1 Training Completed in {train_duration:.2f} minutes.")
    print(f"[✓] Peak Validation Accuracy: {best_acc*100:.2f}%")

    # Final Evaluation on Held-Out Test Set (675 unseen images: 75 per class)
    print("\n" + "=" * 75)
    print("📊 FINAL INDEPENDENT TEST EVALUATION (HELD-OUT N=675)")
    print("=" * 75)
    model.load_state_dict(best_weights)
    model.eval()

    cm = np.zeros((len(STAGE1_CLASSES), len(STAGE1_CLASSES)), dtype=int)
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            preds = out.argmax(1)
            test_correct += (preds == y).sum().item()
            test_total += x.size(0)
            for t_val, p_val in zip(y.cpu().numpy(), preds.cpu().numpy()):
                cm[t_val, p_val] += 1

    final_test_acc = test_correct / test_total
    print(f"\nOverall Test Accuracy: {test_correct}/{test_total} = {final_test_acc*100:.2f}%\n")

    report = {}
    f1_list = []
    print("Class-wise Performance Metrics:")
    for i, cls in enumerate(STAGE1_CLASSES):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = test_total - tp - fn - fp
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        f1_list.append(f1)
        star = " ⭐ [MALIGNANT]" if cls in ["Melanoma", "Basal Cell Carcinoma"] else ""
        print(f"  {cls:25s}: Support={tp+fn:2d} | Sens/Recall={rec*100:5.1f}% | Precision={prec*100:5.1f}% | Specificity={spec*100:5.1f}% | F1={f1:.3f}{star}")
        report[cls] = {
            "support": int(tp + fn),
            "precision": round(float(prec), 3),
            "recall": round(float(rec), 3),
            "specificity": round(float(spec), 3),
            "f1_score": round(float(f1), 3)
        }

    macro_f1 = float(np.mean(f1_list))
    print(f"\nMacro Average F1-Score: {macro_f1:.3f}")

    # Specific Eczema vs Psoriasis Confusion Check
    ecz_idx = class_to_idx["Eczema"]
    pso_idx = class_to_idx["Psoriasis"]
    ecz_as_pso = cm[ecz_idx, pso_idx]
    pso_as_ecz = cm[pso_idx, ecz_idx]
    print(f"\n🔍 Clinical Cross-Talk Check:")
    print(f"   - Eczema misclassified as Psoriasis : {ecz_as_pso}/{cm[ecz_idx, :].sum()} cases")
    print(f"   - Psoriasis misclassified as Eczema : {pso_as_ecz}/{cm[pso_idx, :].sum()} cases")

    # Save weights
    os.makedirs("weights", exist_ok=True)
    save_path = "weights/efficientnet_s1_weights.pth"
    torch.save(best_weights, save_path)
    print(f"\n💾 Saved retrained weights to: {save_path}")

    # Save Confusion Matrix Plot
    save_confusion_matrix_plot(
        cm,
        STAGE1_CLASSES,
        f"Stage 1: Multi-Disease Confusion Matrix (Test Acc: {final_test_acc*100:.1f}%)",
        ["results/confusion_matrix_stage1.png", "app/static/results/confusion_matrix_stage1.png"]
    )

    # Update results/metrics_summary.json
    metrics_path = "results/metrics_summary.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics_summary = json.load(f)
    else:
        metrics_summary = {"models": {}}

    metrics_summary["models"]["Stage 1 (MedicalFocalLoss)"] = {
        "test_accuracy": round(float(final_test_acc), 4),
        "f1_macro": round(float(macro_f1), 3),
        "test_samples": int(test_total),
        "status": "Balanced 9-Class Cohort with Option A Clinical Curation (Pure Mirror-Compatible)"
    }
    report_dict = {}
    for cls in STAGE1_CLASSES:
        report_dict[cls] = report[cls]
    report_dict["accuracy"] = round(float(final_test_acc), 4)
    report_dict["macro_avg"] = {
        "precision": round(float(np.mean([report[c]["precision"] for c in STAGE1_CLASSES])), 3),
        "recall": round(float(np.mean([report[c]["recall"] for c in STAGE1_CLASSES])), 3),
        "f1_score": round(float(macro_f1), 3)
    }
    metrics_summary["stage1_classification_report"] = report_dict

    with open(metrics_path, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"[✓] Updated {metrics_path} successfully!")

    return final_test_acc, macro_f1, report, cm

if __name__ == "__main__":
    main()
