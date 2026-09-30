import os
import copy
import time
import random
from collections import Counter
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from core.models import build_efficientnet
from core.transforms import get_train_transforms_s2, get_val_transforms

ACNE_DIR = "/Users/bariqa/.cache/kagglehub/datasets/jincyjis/acne04/versions/1/JPEGImages"
SEVERITY_CLASSES = ["Level 0 (Clear)", "Level 1 (Mild)", "Level 2 (Moderate)", "Level 3 (Severe)"]

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

def main():
    print("=" * 70)
    print("🔥 STAGE 2 (ACNE SEVERITY) CLINICAL RETRAINING (25 EPOCHS)")
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
    print(f"\n📂 Scanning images in: {ACNE_DIR}")
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

    # 3. Stratified Split (70% Train, 15% Val, 15% Test) in pure Python
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

    # 4. Weighted Sampler to counter class imbalance
    train_labels = [s[1] for s in train_samples]
    tr_counts = Counter(train_labels)
    class_weights = {cls: 1.0 / tr_counts[cls] for cls in range(4)}
    sample_weights = [class_weights[l] for l in train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(
        PureAcneDataset(train_samples, transform=get_train_transforms_s2()),
        batch_size=32, sampler=sampler, num_workers=0
    )
    val_loader = DataLoader(
        PureAcneDataset(val_samples, transform=get_val_transforms()),
        batch_size=32, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        PureAcneDataset(test_samples, transform=get_val_transforms()),
        batch_size=32, shuffle=False, num_workers=0
    )

    # 5. Model Architecture & Optimizer
    print("\n🏗️  Initializing EfficientNet-B0 with ImageNet pretrained backbone...")
    model = build_efficientnet(num_classes=4, pretrained=True).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW([
        {'params': model.features.parameters(), 'lr': 1e-4},
        {'params': model.classifier.parameters(), 'lr': 1e-3}
    ], weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    # 6. Training Loop (25 Epochs)
    epochs = 25
    patience = 6
    best_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())
    no_improve = 0

    print(f"\n🚀 Starting 25-Epoch Training Loop on {device}...")
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
    print(f"\n[✓] Training Completed in {total_duration:.2f} minutes.")
    print(f"[✓] Peak Validation Accuracy: {best_acc*100:.2f}%")

    # 7. Final Test Evaluation
    model.load_state_dict(best_weights)
    model.eval()
    test_correct, test_total = 0, 0
    cm = torch.zeros(4, 4, dtype=torch.int64)

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
    print("\nConfusion Matrix (Rows=True, Columns=Predicted):")
    for i in range(4):
        print(f"  {SEVERITY_CLASSES[i]:25s}: {cm[i].tolist()}")

    # Class-wise metrics
    print("\nClass-wise Metrics:")
    for i in range(4):
        tp = cm[i, i].item()
        fp = (cm[:, i].sum() - tp).item()
        fn = (cm[i, :].sum() - tp).item()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        print(f"  {SEVERITY_CLASSES[i]:25s} | Precision: {prec*100:.1f}% | Recall: {rec*100:.1f}% | F1: {f1:.3f}")

    # Save weights locally
    os.makedirs("weights", exist_ok=True)
    save_path = "weights/efficientnet_s2_weights.pth"
    torch.save(best_weights, save_path)
    print(f"\n💾 Trained weights saved to: {save_path}")

if __name__ == "__main__":
    main()
