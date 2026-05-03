"""
Forest Cover Monitoring — FastAPI Application Entry Point
=========================================================
Registers all routers, configures CORS for the Next.js dev server,
and mounts the static directory so the frontend can fetch generated
change map images via /static/change_maps/<uuid>.png
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import numpy as np

from app.routes.detection import router as detection_router
from app.routes.export import router as export_router
from app.services.inference_service import InferenceService
from app.services.model_loader import ModelLoader

# ── Startup / Shutdown lifecycle ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the segmentation model once at startup (singleton)."""
    print("Forest Monitor -- loading AI model...")

	# 1. Define your model's expected input shape
    # (Batch Size, Height, Width, Channels). 
    # Change 256, 256 to whatever your Attention U-Net expects!
    dummy_input = np.zeros((512, 512, 4), dtype=np.float32)
    
    try:
        # 2. Force the default model to compile its graph and allocate memory
        print(f"Passing dummy tensor of shape {dummy_input.shape} into default model...")
        inference = InferenceService(model_name="attention_unet")
        inference.predict(dummy_input)
        print("Warm-up successful! The model is optimized and ready.")
    except Exception as e:
        print(f"CRITICAL: Model warm-up failed! Check your input shapes: {e}")
        # Raising the error here prevents the server from starting in a broken state
        raise e

    # ModelLoader.get_model()          # warms the singleton

    print("Model ready -- accepting requests.")
    yield
    print("Shutting down Forest Monitor.")


# ── Application factory ───────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="Forest Cover Monitoring API",
        description=(
            "Research-grade Earth Observation API for semantic segmentation "
            "and pixel-wise forest change detection. Designed for ISRO faculty demo."
        ),
        version="1.0.0",
        contact={
            "name": "EO Research Team",
            "email": "research@example.org",
        },
        lifespan=lifespan,
    )

    # ── CORS (allow Next.js dev server + any future deployment origin) ─────────
    origins = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Static files (change maps) ─────────────────────────────────────────────
    static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
    os.makedirs(static_dir, exist_ok=True)
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    # ── Routers ────────────────────────────────────────────────────────────────
    app.include_router(detection_router, prefix="/api/v1", tags=["Change Detection"])
    app.include_router(export_router, prefix="/api/v1", tags=["Export"])

    @app.get("/health", tags=["Health"])
    async def health_check():
        return {"status": "ok", "service": "forest-monitor"}

    return app


app = create_app()
