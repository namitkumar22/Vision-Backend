"""
Vision Backend — FastAPI
Diabetic Retinopathy Detection System
"""

import sys
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.core.config import settings
from app.core.model import load_model
from app.api.routes import router
from app.core.worker import poll_jobs
import asyncio


@asynccontextmanager
async def lifespan(app: FastAPI):
    import logging
    log = logging.getLogger("vision.startup")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")

    log.info("=" * 60)
    log.info("🚀  Vision Backend starting up")
    log.info(f"    Supabase URL : {settings.SUPABASE_URL or '⚠️  NOT SET'}")
    log.info(f"    Supabase Key : {(settings.SUPABASE_KEY[:25] + '...') if settings.SUPABASE_KEY else '⚠️  NOT SET'}")
    log.info("=" * 60)

    if not settings.SUPABASE_URL or not settings.SUPABASE_KEY:
        log.error("❌  SUPABASE credentials missing!")
        log.error("    Add NEXT_PUBLIC_SUPABASE_URL and NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY to Vision/.env")

    # Load ML model on startup
    load_model()
    # Start the background job worker
    task = asyncio.create_task(poll_jobs())
    log.info("✅  Vision Backend ready — worker running")
    yield
    task.cancel()
    log.info("🛑  Vision Backend shutting down")



app = FastAPI(
    title="Vision API",
    description="Diabetic Retinopathy Detection Backend",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.get("/")
def root():
    return {"status": "ok", "message": "Vision API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}
