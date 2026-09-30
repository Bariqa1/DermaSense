import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from core.models import build_efficientnet
from core.pipeline import SEVERITY_CLASSES

def train_stage2(dataloaders, device, epochs=25, patience=5, save_path="weights/efficientnet_s2_weights.pth"):
    """
    Trains Stage 2 Acne Severity Estimation (0-3) using EfficientNet-B0
    with balanced sampling and learning rate scheduling.
    """
    model = build_efficientnet(num_classes=len(SEVERITY_CLASSES), pretrained=True).to(device)

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

    print(f"[*] Commencing Stage 2 Training (Epochs={epochs}, Patience={patience})...")

    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for x, y in dataloaders['train']:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)
            train_correct += (out.argmax(1) == y).sum().item()
            train_total += x.size(0)

        epoch_train_acc = train_correct / train_total
        epoch_train_loss = train_loss / train_total

        # Validation phase
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in dataloaders['val']:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                val_loss += loss.item() * x.size(0)
                val_correct += (out.argmax(1) == y).sum().item()
                val_total += x.size(0)

        epoch_val_acc = val_correct / val_total
        epoch_val_loss = val_loss / val_total

        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_acc'].append(epoch_val_acc)

        print(f"Epoch [{epoch+1:02d}/{epochs:02d}] "
              f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc*100:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc*100:.2f}%")

        scheduler.step(epoch_val_acc)

        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            best_weights = copy.deepcopy(model.state_dict())
            no_improve = 0
            torch.save(best_weights, save_path)
            print(f"  --> Checkpoint saved with validation accuracy: {best_acc*100:.2f}%")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[!] Early stopping triggered after {patience} epochs without improvement.")
                break

    model.load_state_dict(best_weights)
    return model, best_acc, history
