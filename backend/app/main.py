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

from app.routes.detection import router as detection_router
from app.services.model_loader import ModelLoader

# ── Startup / Shutdown lifecycle ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the segmentation model once at startup (singleton)."""
    print("🌲 Forest Monitor — loading AI model...")
    ModelLoader.get_instance()          # warms the singleton
    print("✅ Model ready — accepting requests.")
    yield
    print("🛑 Shutting down Forest Monitor.")


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

    @app.get("/health", tags=["Health"])
    async def health_check():
        return {"status": "ok", "service": "forest-monitor"}

    return app


app = create_app()
