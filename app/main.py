import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Path to weights directory
WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"

# Global predictor instance — loaded once on startup
predictor = None


# ── Lifespan: load model before serving any requests ─────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    print(f"⏳ Loading model weights from: {WEIGHTS_DIR}")
    from src.inference import CaptionPredictor
    predictor = CaptionPredictor(weights_dir=str(WEIGHTS_DIR))
    yield
    print("🛑 Shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Image Captioning API",
    description="Transformer-based image captioning using InceptionV3 + custom Transformer.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────
class PredictResponse(BaseModel):
    caption:      str
    inference_ms: float


class HealthResponse(BaseModel):
    status:      str
    model_ready: bool
    vocab_size:  Optional[int]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["General"])
def health():
    return HealthResponse(
        status="ok",
        model_ready=predictor is not None,
        vocab_size=predictor.vocab_size if predictor else None,
    )


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
async def predict(file: UploadFile = File(...)):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{file.content_type}'. Please upload an image."
        )

    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        t0      = time.perf_counter()
        caption = predictor.predict(image_bytes)
        elapsed = round((time.perf_counter() - t0) * 1000, 2)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    return PredictResponse(caption=caption, inference_ms=elapsed)


# ── Serve frontend — must be LAST ─────────────────────────────────────────────
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")