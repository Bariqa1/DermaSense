import os
import io
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from core.pipeline import DermaSensePipeline

app = FastAPI(
    title="DermaSense AI",
    description="Two-Stage Facial Skin Disease Diagnosis & Acne Severity Estimation API",
    version="1.0.0"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "weights")

# Mount Static and Templates
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# Initialize Pipeline
S1_WEIGHTS = os.path.join(WEIGHTS_DIR, "efficientnet_s1_weights.pth")
S2_WEIGHTS = os.path.join(WEIGHTS_DIR, "efficientnet_s2_weights.pth")

pipeline = None

@app.on_event("startup")
def startup_event():
    global pipeline
    if os.path.exists(S1_WEIGHTS) and os.path.exists(S2_WEIGHTS):
        try:
            pipeline = DermaSensePipeline(S1_WEIGHTS, S2_WEIGHTS)
            print("[✓] DermaSense Pipeline successfully loaded onto device.")
        except Exception as e:
            print(f"[!] Warning: Failed to initialize pipeline: {e}")
    else:
        print("[!] Warning: Model weights not found in weights/ directory.")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/health")
async def health_check():
    return {
        "status": "online" if pipeline is not None else "degraded (weights missing)",
        "models_loaded": pipeline is not None,
        "device": str(pipeline.device) if pipeline else "none"
    }

@app.post("/api/predict")
async def predict_image(file: UploadFile = File(...)):
    global pipeline
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Model weights are not loaded. Please verify weights in the weights/ folder."
        )

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a valid image format.")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        results = pipeline.predict(image)
        return JSONResponse(content={"success": True, "data": results})
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.app:app", host="0.0.0.0", port=8000, reload=True)
