import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import numpy as np
import time

from app.routes.detection import router as detection_router
from app.routes.export import router as export_router
from app.routes.news import router as news_router
from app.services.inference_service import InferenceService
from app.services.news_pipeline import generate_real_time_incidents
from app.routes.historic import router as historic_router

@asynccontextmanager
async def lifespan(app: FastAPI):

    print("Forest Monitor -- loading AI model...")

    dummy_input = np.zeros((512, 512, 4), dtype=np.float32)

    try:
        print(f"Passing dummy tensor of shape {dummy_input.shape} into default model...")

        inference = InferenceService(model_name="attention_unet")
        inference.predict(dummy_input)

        print("Warm-up successful!")

    except Exception as e:
        print(f"CRITICAL: Model warm-up failed! {e}")
        raise e

    # =========================================================
    # PRE-COMPUTE NEWS ON STARTUP
    # =========================================================
    print("Forest Monitor -- pre-computing live news...")
    try:
        start_time = time.time()
        generate_real_time_incidents()
        elapsed = time.time() - start_time
        print(f"News pre-computation completed in {elapsed:.2f}s")
    except Exception as e:
        print(f"Warning: News pre-computation failed: {e}")

    yield

    print("Shutting down Forest Monitor.")


def create_app() -> FastAPI:

    app = FastAPI(
        title="Forest Cover Monitoring API",
        version="1.0.0",
        lifespan=lifespan,
    )

    origins = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
		max_age=86400,
    )

    static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
    os.makedirs(static_dir, exist_ok=True)
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    # ── Routers ────────────────────────────────────────────────────────────────
    app.include_router(detection_router, prefix="/api/v1", tags=["Change Detection"])
    app.include_router(export_router, prefix="/api/v1", tags=["Export"])
    app.include_router(news_router)
    app.include_router(historic_router)
    
    @app.get("/health")
    async def health_check():
        return {"status": "ok"}

    return app


app = create_app()