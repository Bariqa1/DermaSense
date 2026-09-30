import os
import json
from PIL import Image
from core.pipeline import DermaSensePipeline

def evaluate_samples():
    pipeline = DermaSensePipeline(
        "weights/efficientnet_s1_weights.pth",
        "weights/efficientnet_s2_weights.pth"
    )

    samples = [
        ("Acne Sample", "app/static/samples/acne.jpg"),
        ("Eczema Sample", "app/static/samples/eczema.jpg"),
        ("Melanoma Sample", "app/static/samples/melanoma.jpg"),
        ("Clear Skin Sample", "app/static/samples/clear.jpg")
    ]

    print("\n" + "="*80)
    print("           DERMASENSE AI - CLINICAL INFERENCE EVALUATION REPORT")
    print("="*80 + "\n")

    for title, path in samples:
        if not os.path.exists(path):
            print(f"[-] Sample file not found: {path}")
            continue

        img = Image.open(path)
        res = pipeline.predict(img)

        print(f"📸 Case: {title} ({path})")
        print(f"   ├─ Primary Diagnosis : {res['primary']['disease']} (Confidence: {res['primary']['confidence']}%)")
        print(f"   ├─ Category          : {res['primary']['metadata'].get('category', 'N/A')}")
        print(f"   ├─ Risk Level        : {res['primary']['metadata'].get('risk', 'N/A')}")
        print(f"   ├─ Description       : {res['primary']['metadata'].get('description', '')}")
        
        diff_str = " | ".join([f"{d['rank']}. {d['disease']} ({d['confidence']}%)" for d in res['differential_diagnosis']])
        print(f"   ├─ Differential (Top-3): {diff_str}")

        if res['is_acne'] and res['acne_severity']:
            sev = res['acne_severity']
            print(f"   ├─ Stage 2 (Acne)    : {sev['label']} (Confidence: {sev['confidence']}%)")
            print(f"   └─ Clinical Advice   : {sev['clinical_advice']}")
        else:
            print(f"   └─ Stage 2 (Acne)    : Not Applicable (Non-acne presentation)")
        print("-" * 80)

if __name__ == "__main__":
    evaluate_samples()
