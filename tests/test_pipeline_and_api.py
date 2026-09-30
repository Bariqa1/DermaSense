import os
import io
import time
import unittest
import numpy as np
import torch
from PIL import Image
from fastapi.testclient import TestClient

from core.models import CustomCNN, build_efficientnet
from core.pipeline import DermaSensePipeline, STAGE1_CLASSES, SEVERITY_CLASSES
from app.app import app, startup_event

class TestDermaSense(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.weights_s1 = "weights/efficientnet_s1_weights.pth"
        cls.weights_s2 = "weights/efficientnet_s2_weights.pth"
        
        # Verify weight files exist
        assert os.path.exists(cls.weights_s1), f"Missing {cls.weights_s1}"
        assert os.path.exists(cls.weights_s2), f"Missing {cls.weights_s2}"
        
        # Initialize pipeline
        cls.pipeline = DermaSensePipeline(cls.weights_s1, cls.weights_s2)
        
        # Initialize FastAPI TestClient
        startup_event()
        cls.client = TestClient(app)

    # 1. Model Architecture & Tensor Tests
    def test_01_model_shapes(self):
        """Verifies that model architectures produce expected tensor output shapes."""
        dummy_tensor = torch.randn(2, 3, 224, 224).to(self.pipeline.device)
        
        # Stage 1
        out_s1 = self.pipeline.model_s1(dummy_tensor)
        self.assertEqual(out_s1.shape, (2, 9), "Stage 1 output must be (batch, 9)")
        
        # Stage 2
        out_s2 = self.pipeline.model_s2(dummy_tensor)
        self.assertEqual(out_s2.shape, (2, 4), "Stage 2 output must be (batch, 4)")

        # CustomCNN Baseline
        base_cnn = CustomCNN(num_classes=4).to(self.pipeline.device)
        out_base = base_cnn(dummy_tensor)
        self.assertEqual(out_base.shape, (2, 4), "Custom CNN output must be (batch, 4)")

    # 2. Clinical Pipeline Logic Tests
    def test_02_pipeline_acne_inference(self):
        """Tests that Acne sample correctly executes Stage 1 and Stage 2 conditional severity."""
        acne_path = "app/static/samples/acne.jpg"
        self.assertTrue(os.path.exists(acne_path), "Sample acne.jpg must exist")
        
        img = Image.open(acne_path)
        result = self.pipeline.predict(img)
        
        self.assertIn("primary", result)
        self.assertIn("differential_diagnosis", result)
        self.assertEqual(len(result["differential_diagnosis"]), 3, "Must return Top-3 differentials")
        
        # Check confidence ranges
        primary_conf = result["primary"]["confidence"]
        self.assertGreaterEqual(primary_conf, 0.0)
        self.assertLessEqual(primary_conf, 100.0)
        
        # Verify metadata presence
        meta = result["primary"]["metadata"]
        self.assertIn("risk", meta)
        self.assertIn("category", meta)
        self.assertIn("description", meta)

    def test_03_pipeline_non_acne_behavior(self):
        """Verifies that non-acne diseases (e.g., Eczema) do not trigger Stage 2 severity."""
        eczema_path = "app/static/samples/eczema.jpg"
        img = Image.open(eczema_path)
        result = self.pipeline.predict(img)
        
        # If predicted disease is not Acne or Clear_AlmostClear, acne_severity should be None
        if not result["is_acne"]:
            self.assertIsNone(result["acne_severity"], "Stage 2 must not be triggered for non-acne cases")
        else:
            self.assertIsNotNone(result["acne_severity"])

    # 3. Latency & Performance Benchmark
    def test_04_latency_benchmark(self):
        """Benchmarks inference latency across 10 sequential predictions."""
        img = Image.fromarray((np.random.rand(224, 224, 3) * 255).astype('uint8'))
        
        # Warmup
        _ = self.pipeline.predict(img)
        
        latencies = []
        for _ in range(10):
            t0 = time.perf_counter()
            _ = self.pipeline.predict(img)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)
            
        avg_latency = np.mean(latencies)
        min_latency = np.min(latencies)
        max_latency = np.max(latencies)
        
        print(f"\n[Benchmark] Latency over 10 iterations: Avg={avg_latency:.2f}ms | Min={min_latency:.2f}ms | Max={max_latency:.2f}ms")
        self.assertLess(avg_latency, 500, "Average inference latency should be under 500ms on CPU")

    # 4. REST API Endpoint Tests
    def test_05_api_health(self):
        """Tests the GET /api/health endpoint."""
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "online")
        self.assertTrue(data["models_loaded"])
        self.assertIn("device", data)

    def test_06_api_root(self):
        """Tests the GET / root template rendering."""
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn("DermaSense", res.text)
        self.assertIn("engineStatusBadge", res.text)
        # Ensure removed static elements are not present
        self.assertNotIn("Clinical AI v1.0", res.text)
        self.assertNotIn("Neural Engine Online", res.text)

    def test_07_api_predict_valid(self):
        """Tests POST /api/predict with a valid JPEG image."""
        img = Image.new("RGB", (224, 224), color=(200, 150, 130))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)
        
        res = self.client.post("/api/predict", files={"file": ("test.jpg", buf, "image/jpeg")})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["success"])
        self.assertIn("primary", data["data"])

    def test_08_api_predict_invalid_file(self):
        """Tests POST /api/predict rejection when uploading non-image data."""
        text_data = io.BytesIO(b"Hello world, this is a plain text file, not a dermatological image.")
        res = self.client.post("/api/predict", files={"file": ("document.txt", text_data, "text/plain")})
        self.assertEqual(res.status_code, 400)
        self.assertIn("Uploaded file must be a valid image", res.json().get("detail", ""))

if __name__ == "__main__":
    unittest.main()
