"""
Prediction endpoint.

Flow:
  1. Receive scan UUID from frontend
  2. Fetch all 8 images from Supabase Storage for that UUID
  3. Download images to temp dir
  4. Pick the sharpest image using pick_best_image pipeline (unchanged)
  5. Run image_pipeline preprocessing (unchanged)
  6. Run model inference
  7. Save result to Supabase `scans` table
  8. Return prediction result
"""

import sys
import os
import uuid
import tempfile
import shutil
import urllib.request

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.config import settings
from app.core.model import predict as run_predict
from app.core.supabase import get_supabase

# Import pipelines from Image_Processing (unmodified)
sys.path.insert(0, settings.IMAGE_PROCESSING_PATH)
from pick_best_image import score_sharpness
from image_pipeline import preprocess_retinal_image

router = APIRouter()


class PredictRequest(BaseModel):
    scan_uuid: str  # UUID shared by all 8 images


@router.post("/")
async def predict_scan(req: PredictRequest):
    """
    Main prediction endpoint.
    Fetches images from Supabase, picks best, runs pipeline + model.
    """
    supabase = get_supabase()
    scan_uuid = req.scan_uuid

    # ── 1. Fetch image URLs from DB ──────────────────────────────
    result = (
        supabase.table("scan_images")
        .select("*")
        .eq("scan_uuid", scan_uuid)
        .execute()
    )

    if not result.data:
        raise HTTPException(status_code=404, detail="No images found for this UUID")

    image_records = result.data  # list of {id, scan_uuid, image_url, created_at}

    # ── 2. Download images to temp folder ────────────────────────
    tmp_dir = tempfile.mkdtemp(prefix="vision_")
    downloaded_paths = []

    try:
        for idx, record in enumerate(image_records):
            img_url = record["image_url"]
            ext = os.path.splitext(img_url)[-1] or ".jpg"
            local_path = os.path.join(tmp_dir, f"image_{idx}{ext}")

            try:
                urllib.request.urlretrieve(img_url, local_path)
                downloaded_paths.append(local_path)
            except Exception as e:
                print(f"⚠️  Could not download {img_url}: {e}")

        if not downloaded_paths:
            raise HTTPException(status_code=500, detail="Failed to download any images")

        # ── 3. Pick best (sharpest) image ─────────────────────────
        scored = [(p, score_sharpness(p)) for p in downloaded_paths]
        scored.sort(key=lambda x: x[1], reverse=True)
        best_path = scored[0][0]
        print(f"Best image: {os.path.basename(best_path)}  score={scored[0][1]:.2f}")

        # ── 4. Run image pipeline (unmodified) ────────────────────
        processed = preprocess_retinal_image(best_path, target_size=380)

        # ── 5. Model inference ────────────────────────────────────
        prediction = run_predict(processed)

        # ── 6. Store result in Supabase ───────────────────────────
        supabase.table("scans").upsert({
            "scan_uuid": scan_uuid,
            "grade": prediction["grade"],
            "label": prediction["label"],
            "confidence": prediction["confidence"],
            "description": prediction["description"],
            "all_probabilities": prediction["all_probabilities"],
            "status": "completed",
        }).execute()

        return {
            "scan_uuid": scan_uuid,
            "prediction": prediction,
            "images_analyzed": len(downloaded_paths),
            "best_image_score": round(scored[0][1], 2),
        }

    finally:
        # Always clean up temp files
        shutil.rmtree(tmp_dir, ignore_errors=True)


@router.get("/status/{scan_uuid}")
async def get_scan_status(scan_uuid: str):
    """Check if a scan UUID has been processed."""
    supabase = get_supabase()

    result = (
        supabase.table("scans")
        .select("*")
        .eq("scan_uuid", scan_uuid)
        .execute()
    )

    if not result.data:
        return {"status": "pending", "scan_uuid": scan_uuid}

    return {"status": "completed", "scan_uuid": scan_uuid, "result": result.data[0]}
