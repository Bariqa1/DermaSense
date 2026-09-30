import torch
import torch.nn as nn
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

def extract_features(backbone, dataloader, device):
    """
    Extracts deep embeddings (1280-dim) from the penultimiate layer of EfficientNet.
    """
    backbone.eval()
    features_list, labels_list = [], []

    # Temporarily substitute classifier head with Identity
    orig_head = backbone.classifier[1]
    backbone.classifier[1] = nn.Identity()

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            feats = backbone(x).cpu().numpy()
            features_list.append(feats)
            labels_list.append(y.numpy())

    # Restore head
    backbone.classifier[1] = orig_head
    return np.vstack(features_list), np.hstack(labels_list)


def run_hybrid_experiments(backbone, loaders, device, seed=42):
    """
    Trains and evaluates Classical ML classifiers (SVM, Logistic Regression, MLP)
    on top of PCA-reduced deep features with StandardScaler normalization.
    """
    print("  [+] Extracting deep features from backbone...")
    X_train, y_train = extract_features(backbone, loaders['train'], device)
    X_val, y_val = extract_features(backbone, loaders['val'], device)
    X_test, y_test = extract_features(backbone, loaders['test'], device)

    # 1. Standard Scaling (Crucial for SVM & PCA)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 2. PCA Dimensionality Reduction
    pca = PCA(n_components=0.95, random_state=seed)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    print(f"  [+] PCA reduced feature dimensions from {X_train.shape[1]} to {X_train_pca.shape[1]}")

    classifiers = {
        "Logistic Regression": LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000, class_weight='balanced'),
        "SVM (RBF Kernel)": SVC(C=1.0, kernel='rbf', gamma='scale', class_weight='balanced'),
        "MLP Classifier": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1000, early_stopping=True, alpha=0.0001, random_state=seed)
    }

    results = {}
    best_acc = 0.0
    best_model_name = ""
    best_preds = None

    for name, clf in classifiers.items():
        clf.fit(X_train_pca, y_train)
        preds = clf.predict(X_test_pca)
        acc = accuracy_score(y_test, preds)
        results[name] = acc
        print(f"    --> {name}: Test Accuracy = {acc*100:.2f}%")

        if acc > best_acc:
            best_acc = acc
            best_model_name = name
            best_preds = preds

    return results, y_test, best_preds, best_model_name
