import os
import torch
import torch.nn.functional as F
from PIL import Image
from typing import Dict, Any, List

from core.models import build_efficientnet
from core.transforms import get_val_transforms

STAGE1_CLASSES: List[str] = [
    "Acne",
    "Basal Cell Carcinoma",
    "Benign Keratosis",
    "Clear_AlmostClear",
    "Eczema",
    "Melanocytic Nevi",
    "Melanoma",
    "Psoriasis",
    "Seborrheic Keratoses"
]

SEVERITY_CLASSES: List[str] = [
    "Level 0 (Clear / Almost Clear)",
    "Level 1 (Mild Acne)",
    "Level 2 (Moderate Acne)",
    "Level 3 (Severe Acne)"
]

DISEASE_METADATA: Dict[str, Dict[str, str]] = {
    "Acne": {
        "risk": "Low / Benign",
        "category": "Sebaceous Gland Disorder",
        "description": "Common dermatological condition caused by clogged hair follicles with oil and dead skin cells.",
        "badge_color": "warning"
    },
    "Basal Cell Carcinoma": {
        "risk": "High / Malignant",
        "category": "Non-Melanoma Skin Cancer",
        "description": "The most common form of skin cancer. Locally invasive, requires professional dermatological biopsy and excision.",
        "badge_color": "danger"
    },
    "Benign Keratosis": {
        "risk": "Low / Benign",
        "category": "Epidermal Growth",
        "description": "Non-cancerous skin lesion commonly appearing with aging. Usually harmless unless irritated.",
        "badge_color": "success"
    },
    "Clear_AlmostClear": {
        "risk": "Healthy",
        "category": "Normal Skin",
        "description": "No significant dermatological lesions or active inflammation detected.",
        "badge_color": "success"
    },
    "Eczema": {
        "risk": "Moderate / Chronic",
        "category": "Inflammatory Dermatosis",
        "description": "Atopic dermatitis causing itchy, dry, reddened patches of skin with periodic flare-ups.",
        "badge_color": "info"
    },
    "Melanocytic Nevi": {
        "risk": "Low / Monitored",
        "category": "Melanocytic Lesion",
        "description": "Common mole consisting of melanocyte clusters. Routine self-inspection (ABCDE rule) recommended.",
        "badge_color": "info"
    },
    "Melanoma": {
        "risk": "Critical / Malignant",
        "category": "Melanoma Skin Cancer",
        "description": "Aggressive malignant tumor originating from melanocytes. Urgent specialist consultation and staging required.",
        "badge_color": "danger"
    },
    "Psoriasis": {
        "risk": "Moderate / Autoimmune",
        "category": "Papulosquamous Disorder",
        "description": "Chronic autoimmune condition leading to rapid skin cell proliferation, forming silvery, scaly plaques.",
        "badge_color": "warning"
    },
    "Seborrheic Keratoses": {
        "risk": "Low / Benign",
        "category": "Benign Neoplasm",
        "description": "Common benign epidermal tumor with a waxy, 'pasted-on' appearance.",
        "badge_color": "success"
    }
}

SEVERITY_GUIDELINES: Dict[int, Dict[str, str]] = {
    0: {
        "title": "Clear / Almost Clear",
        "action": "Maintain daily hygiene with a non-comedogenic gentle cleanser and SPF 30+ sunscreen.",
        "badge": "success"
    },
    1: {
        "title": "Mild Acne",
        "action": "Topical OTC agents (Salicylic Acid 2% or Benzoyl Peroxide 2.5-5%). Avoid picking or physical scrubs.",
        "badge": "info"
    },
    2: {
        "title": "Moderate Acne",
        "action": "Topical retinoid (Adapalene/Tretinoin) + topical antimicrobial. Consult a dermatologist if persistent.",
        "badge": "warning"
    },
    3: {
        "title": "Severe Acne",
        "action": "High risk of scarring. Immediate medical dermatology consultation recommended for systemic therapies.",
        "badge": "danger"
    }
}

class DermaSensePipeline:
    """
    Two-Stage Clinical Diagnostic Pipeline:
      Stage 1: Multi-disease classification (EfficientNet-B0)
      Stage 2: Conditional Acne Severity Estimation (Levels 0-3)
    """
    def __init__(self, s1_weights_path: str, s2_weights_path: str, device: str = None):
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.transforms = get_val_transforms()

        # Build & Load Stage 1
        self.model_s1 = build_efficientnet(num_classes=len(STAGE1_CLASSES), pretrained=False)
        if os.path.exists(s1_weights_path):
            state_dict_s1 = torch.load(s1_weights_path, map_location=self.device)
            self.model_s1.load_state_dict(state_dict_s1)
        else:
            raise FileNotFoundError(f"Stage 1 weights not found at: {s1_weights_path}")
        self.model_s1.to(self.device)
        self.model_s1.eval()

        # Build & Load Stage 2
        self.model_s2 = build_efficientnet(num_classes=len(SEVERITY_CLASSES), pretrained=False)
        if os.path.exists(s2_weights_path):
            state_dict_s2 = torch.load(s2_weights_path, map_location=self.device)
            self.model_s2.load_state_dict(state_dict_s2)
        else:
            raise FileNotFoundError(f"Stage 2 weights not found at: {s2_weights_path}")
        self.model_s2.to(self.device)
        self.model_s2.eval()

    def predict(self, image: Image.Image, use_tta: bool = True) -> Dict[str, Any]:
        """
        Runs complete inference pipeline on a PIL Image with optional Test-Time Augmentation (TTA).
        Returns diagnosis, top-3 differential list, probabilities, critical alerts, and severity (if applicable).
        """
        if image.mode != "RGB":
            image = image.convert("RGB")

        base_tensor = self.transforms(image)

        if use_tta:
            # Test-Time Augmentation: 5 distinct geometric orientations for clinical robustness
            t_orig = base_tensor
            t_hflip = torch.flip(base_tensor, dims=[2])
            t_vflip = torch.flip(base_tensor, dims=[1])
            t_hvflip = torch.flip(base_tensor, dims=[1, 2])
            t_rot = torch.rot90(base_tensor, k=1, dims=[1, 2])
            batch_tensor = torch.stack([t_orig, t_hflip, t_vflip, t_hvflip, t_rot]).to(self.device)

            # Stage 1: Diagnosis with TTA Ensemble Averaging
            with torch.no_grad():
                logits_s1 = self.model_s1(batch_tensor)
                probs_s1 = F.softmax(logits_s1, dim=1).mean(dim=0).cpu().numpy()
        else:
            tensor = base_tensor.unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits_s1 = self.model_s1(tensor)
                probs_s1 = F.softmax(logits_s1, dim=1)[0].cpu().numpy()

        sorted_indices = probs_s1.argsort()[::-1]
        top_idx = int(sorted_indices[0])
        primary_disease = STAGE1_CLASSES[top_idx]
        primary_prob = float(probs_s1[top_idx])

        # Clinical Safeguard: Critical Malignancy Early Alert
        # If Melanoma probability is elevated (>20%) even if not the top ranked label
        melanoma_idx = STAGE1_CLASSES.index("Melanoma")
        melanoma_prob = float(probs_s1[melanoma_idx])
        critical_alert = None
        if melanoma_prob >= 0.20:
            critical_alert = {
                "flagged": True,
                "condition": "Melanoma (Malignant)",
                "confidence": round(melanoma_prob * 100, 2),
                "is_primary": (primary_disease == "Melanoma"),
                "advisory": "Clinical safety protocol triggered: elevated melanoma probability detected. Immediate dermoscopy evaluation strongly indicated."
            }

        # Differential Diagnosis (Top 3)
        top_predictions = []
        for rank, idx in enumerate(sorted_indices[:3], start=1):
            d_name = STAGE1_CLASSES[idx]
            top_predictions.append({
                "rank": rank,
                "disease": d_name,
                "confidence": round(float(probs_s1[idx]) * 100, 2),
                "metadata": DISEASE_METADATA.get(d_name, {})
            })

        # Stage 2: Acne Severity (Triggered if primary or high secondary is Acne)
        severity_info = None
        is_acne_case = primary_disease in ["Acne", "Clear_AlmostClear"]

        if is_acne_case:
            with torch.no_grad():
                if use_tta:
                    logits_s2 = self.model_s2(batch_tensor)
                    probs_s2 = F.softmax(logits_s2, dim=1).mean(dim=0).cpu().numpy()
                else:
                    logits_s2 = self.model_s2(tensor)
                    probs_s2 = F.softmax(logits_s2, dim=1)[0].cpu().numpy()

            s2_idx = int(probs_s2.argmax())
            severity_prob = float(probs_s2[s2_idx])
            guideline = SEVERITY_GUIDELINES.get(s2_idx, {})

            # Distribution across all 4 levels
            severity_breakdown = [
                {
                    "level": i,
                    "label": SEVERITY_CLASSES[i],
                    "confidence": round(float(probs_s2[i]) * 100, 2)
                }
                for i in range(len(SEVERITY_CLASSES))
            ]

            severity_info = {
                "level": s2_idx,
                "label": SEVERITY_CLASSES[s2_idx],
                "confidence": round(severity_prob * 100, 2),
                "clinical_advice": guideline.get("action", ""),
                "badge": guideline.get("badge", "info"),
                "breakdown": severity_breakdown
            }

        return {
            "primary": {
                "disease": primary_disease,
                "confidence": round(primary_prob * 100, 2),
                "metadata": DISEASE_METADATA.get(primary_disease, {})
            },
            "differential_diagnosis": top_predictions,
            "critical_alert": critical_alert,
            "tta_enabled": use_tta,
            "is_acne": is_acne_case,
            "acne_severity": severity_info
        }
