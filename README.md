# DermaSense 🩺🔬
### Hierarchical Deep Learning for Facial Skin Disease Diagnosis & Acne Severity Grading

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Backbone](https://img.shields.io/badge/Backbone-EfficientNet--B0-green.svg)](https://arxiv.org/abs/1905.11946)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**DermaSense** is a two-stage clinical decision support system designed to assist healthcare professionals and dermatological researchers in early detection of facial skin conditions and granular severity quantification of acne vulgaris.

---

## 🌟 Key Highlights

1. **Two-Stage Hierarchical Classification**:
   - **Stage 1 (Disease Diagnosis):** 9-class differential diagnosis covering inflammatory dermatoses, benign growths, and life-threatening malignancies (*Eczema*, *Melanoma*, *Basal Cell Carcinoma*, *Psoriasis*, *Seborrheic Keratoses*, *Benign Keratosis*, *Melanocytic Nevi*, *Acne*, *Clear Skin*).
   - **Stage 2 (Condition-Specific Grading):** Triggered conditionally when acne lesions are identified, estimating Global Acne Grading severity (*Level 0: Clear* to *Level 3: Severe*).
2. **Transfer Learning Backbone**:
   - Utilizes **EfficientNet-B0** pretrained on ImageNet with differential learning rates and label smoothing.
3. **Rigorous Experimental Benchmarking**:
   - Benchmarked against a scratch-trained **Custom CNN Baseline** and **Hybrid Embeddings Classifiers** (EfficientNet Features + PCA + SVM/MLP/Logistic Regression).
4. **Production-Ready Web Application**:
   - Interactive medical dashboard built with **FastAPI**, featuring real-time image analysis, confidence gauges, differential diagnoses, and clinical care recommendations.

---

## 🧠 System Architecture

```text
               +----------------------------------+
               |   Facial Image / Dermoscopy      |
               +-----------------+----------------+
                                 |
                                 v
                     [ 224x224 Preprocessing ]
                                 |
                                 v
               +----------------------------------+
               |   Stage 1: EfficientNet-B0       |
               |   Multi-Class Disease Diagnosis  |
               +-----------------+----------------+
                                 |
        +------------------------+------------------------+
        |                                                 |
        v                                                 v
[ Malignant / Other Conditions ]                  [ Acne / Clear Skin ]
- Melanoma                                                |
- Basal Cell Carcinoma                                    v
- Eczema / Psoriasis                    +----------------------------------+
- Keratoses / Nevi                      |   Stage 2: EfficientNet-B0       |
                                        |   Acne Severity Grading (0 - 3)  |
                                        +-----------------+----------------+
                                                          |
                                                          v
                                        Level 0: Clear / Almost Clear
                                        Level 1: Mild
                                        Level 2: Moderate
                                        Level 3: Severe
```

---

## 📁 Repository Structure

```text
DermaSense/
├── app/                          # Production Web Application
│   ├── app.py                    # FastAPI server & inference endpoints
│   ├── templates/
│   │   └── index.html            # Medical dashboard interface
│   └── static/
│       ├── css/style.css         # Glassmorphism aesthetic styling
│       └── js/app.js             # Asynchronous image processing & UI logic
├── core/                         # Core Machine Learning Framework
│   ├── models.py                 # CustomCNN & EfficientNet-B0 architectures
│   ├── pipeline.py               # Two-stage clinical diagnostic pipeline
│   └── transforms.py             # Preprocessing & data augmentation
├── training/                     # Research & Training Pipeline
│   ├── dataset.py                # Dataset loaders with stratified sampling
│   ├── train_stage1.py           # Stage 1 fine-tuning script
│   ├── train_stage2.py           # Stage 2 severity training with balanced sampler
│   └── hybrid_models.py          # PCA feature extraction & classical ML models
├── weights/                      # Model Checkpoints
│   ├── download_weights.py       # Automated weight downloader
│   └── .gitkeep
├── notebooks/                    # Interactive Colab / Jupyter Research Notebooks
│   └── Skin_Disease_Project.ipynb
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🚀 Quickstart & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Bariqa1/DermaSense.git
cd DermaSense
```

### 2. Create and Activate Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Model Weights
If you do not have the trained weights locally in `weights/`, run the automated downloader:
```bash
python weights/download_weights.py
```

### 5. Launch the Web Application
```bash
uvicorn app.app:app --host 0.0.0.0 --port 8000 --reload
```
Open your browser and navigate to: **`http://localhost:8000`**

---

## 📡 REST API Reference

### Health Check
- **Endpoint**: `GET /api/health`
- **Response**:
```json
{
  "status": "online",
  "models_loaded": true,
  "device": "cuda"
}
```

### Predict Skin Lesion
- **Endpoint**: `POST /api/predict`
- **Body**: `multipart/form-data` with key `file` (Image)
- **Sample Response**:
```json
{
  "success": true,
  "data": {
    "primary": {
      "disease": "Acne",
      "confidence": 92.45,
      "metadata": {
        "risk": "Low / Benign",
        "category": "Sebaceous Gland Disorder",
        "description": "Common dermatological condition caused by clogged hair follicles."
      }
    },
    "differential_diagnosis": [
      {"rank": 1, "disease": "Acne", "confidence": 92.45},
      {"rank": 2, "disease": "Eczema", "confidence": 4.12},
      {"rank": 3, "disease": "Seborrheic Keratoses", "confidence": 1.28}
    ],
    "is_acne": true,
    "acne_severity": {
      "level": 2,
      "label": "Level 2 (Moderate Acne)",
      "confidence": 88.30,
      "clinical_advice": "Topical retinoid + antimicrobial treatment under dermatological guidance."
    }
  }
}
```

---

## 🔬 Datasets Utilized

- **Skin Diseases Multi-Class Dataset**: [Kaggle skin-diseases-image-dataset](https://www.kaggle.com/datasets/ismailpromus/skin-diseases-image-dataset)
- **Acne04 Facial Dataset**: [Kaggle acne04](https://www.kaggle.com/datasets/jincyjis/acne04) containing 4 standardized severity grading levels.

---

## ⚖️ Clinical Disclaimer

*DermaSense is intended strictly as a research investigation tool and decision-support prototype. It is not approved as an independent diagnostic medical device. Final clinical management must always be determined by licensed dermatologists and confirmed via histopathology when indicated.*

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
