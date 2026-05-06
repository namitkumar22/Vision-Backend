"""
Vision Background Worker
Polls Supabase for jobs with status='processing', runs the full
image pipeline + model inference, and writes results back to DB.
"""

import asyncio
import sys
import os
import tempfile
import urllib.request
import urllib.error
import shutil
import traceback
import time
import logging

from app.core.supabase import get_supabase
from app.core.model import predict as run_predict
from app.core.config import settings

# ── Configure a clean, structured logger ──────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("vision.worker")

# Import pipelines
sys.path.insert(0, settings.IMAGE_PROCESSING_PATH)
try:
    from pick_best_image import score_sharpness
    from image_pipeline import preprocess_retinal_image
    log.info("✅  Image processing pipeline imported successfully")
except ImportError as e:
    log.error(f"❌  Failed to import image pipeline: {e}")
    raise

POLL_INTERVAL_IDLE   = 3    # seconds between polls when no jobs
POLL_INTERVAL_ERROR  = 10   # seconds to wait after a Supabase error
MAX_DOWNLOAD_TIMEOUT = 30   # seconds per image download


# ── Supabase connectivity test ────────────────────────────────
def test_supabase_connection() -> bool:
    """Verify Supabase is reachable on startup."""
    log.info(f"🔗  Testing Supabase connection to: {settings.SUPABASE_URL}")
    try:
        supabase = get_supabase()
        # Lightweight ping — select 1 row limit
        result = supabase.table("scans").select("scan_uuid").limit(1).execute()
        log.info("✅  Supabase connection OK")
        return True
    except Exception as e:
        log.error(f"❌  Supabase connection FAILED: {e}")
        log.error("    → Check NEXT_PUBLIC_SUPABASE_URL and NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY in .env")
        return False


# ── Single job processor ──────────────────────────────────────
async def process_job(scan_uuid: str):
    """Download images, pick best, run pipeline + model, save result."""
    supabase = get_supabase()
    t_start = time.perf_counter()
    log.info(f"🚀  [{scan_uuid[:8]}] Job started")

    try:
        # ── 1. Fetch image records ────────────────────────────
        # NOTE: We intentionally do NOT update status to "running" here because
        # the scans_status_check constraint may not include that value.
        # Flow: processing → completed | failed
        log.info(f"   [{scan_uuid[:8]}] Fetching image records...")
        result = supabase.table("scan_images") \
            .select("*") \
            .eq("scan_uuid", scan_uuid) \
            .execute()

        if not result.data:
            raise ValueError(f"No image records found for UUID {scan_uuid}")

        log.info(f"   [{scan_uuid[:8]}] Found {len(result.data)} image record(s) in DB")

        # ── 3. Download images to temp dir ────────────────────
        tmp_dir = tempfile.mkdtemp(prefix="vision_")
        downloaded_paths: list[str] = []

        try:
            for idx, record in enumerate(result.data):
                img_url = record.get("image_url", "")
                if not img_url:
                    log.warning(f"   [{scan_uuid[:8]}] Record {idx} has no image_url — skipping")
                    continue

                ext = os.path.splitext(img_url.split("?")[0])[-1] or ".jpg"
                local_path = os.path.join(tmp_dir, f"image_{idx:02d}{ext}")

                log.info(f"   [{scan_uuid[:8]}] Downloading image {idx+1}/{len(result.data)}...")
                try:
                    urllib.request.urlretrieve(img_url, local_path)
                    size_kb = os.path.getsize(local_path) / 1024
                    downloaded_paths.append(local_path)
                    log.info(f"   [{scan_uuid[:8]}] ✓ image_{idx:02d} saved ({size_kb:.1f} KB)")
                except urllib.error.URLError as e:
                    log.warning(f"   [{scan_uuid[:8]}] ✗ Download failed for image {idx}: {e}")
                except Exception as e:
                    log.warning(f"   [{scan_uuid[:8]}] ✗ Unexpected download error for image {idx}: {e}")

            if not downloaded_paths:
                raise RuntimeError("All image downloads failed — cannot proceed")

            log.info(f"   [{scan_uuid[:8]}] Downloaded {len(downloaded_paths)}/{len(result.data)} images")

            # ── 4. Pick sharpest image ────────────────────────
            log.info(f"   [{scan_uuid[:8]}] Scoring image sharpness...")
            scored = []
            for p in downloaded_paths:
                try:
                    score = score_sharpness(p)
                    scored.append((p, score))
                    log.info(f"   [{scan_uuid[:8]}]   {os.path.basename(p)}: score={score:.2f}")
                except Exception as e:
                    log.warning(f"   [{scan_uuid[:8]}]   Could not score {os.path.basename(p)}: {e}")

            if not scored:
                raise RuntimeError("Could not score any images")

            scored.sort(key=lambda x: x[1], reverse=True)
            best_path, best_score = scored[0]
            log.info(f"   [{scan_uuid[:8]}] Best image → {os.path.basename(best_path)} (score={best_score:.2f})")

            # ── 5. Run preprocessing pipeline ────────────────
            log.info(f"   [{scan_uuid[:8]}] Running retinal preprocessing pipeline...")
            t_pipe = time.perf_counter()
            processed = preprocess_retinal_image(best_path, target_size=380)
            log.info(f"   [{scan_uuid[:8]}] Pipeline done in {time.perf_counter()-t_pipe:.2f}s → array shape {processed.shape}")

            # ── 6. Model inference ────────────────────────────
            log.info(f"   [{scan_uuid[:8]}] Running model inference...")
            t_infer = time.perf_counter()
            prediction = run_predict(processed)
            log.info(
                f"   [{scan_uuid[:8]}] Inference done in {time.perf_counter()-t_infer:.2f}s → "
                f"Grade {prediction['grade']} ({prediction['label']}) | "
                f"Confidence: {prediction['confidence']}%"
            )
            log.info(f"   [{scan_uuid[:8]}] All probabilities: {prediction['all_probabilities']}")

            # ── 7. Write result to DB ─────────────────────────
            log.info(f"   [{scan_uuid[:8]}] Saving result to Supabase...")
            supabase.table("scans").update({
                "grade":             prediction["grade"],
                "label":             prediction["label"],
                "confidence":        prediction["confidence"],
                "description":       prediction["description"],
                "all_probabilities": prediction["all_probabilities"],
                "status":            "completed",
            }).eq("scan_uuid", scan_uuid).execute()

            elapsed = time.perf_counter() - t_start
            log.info(f"✅  [{scan_uuid[:8]}] Job COMPLETED in {elapsed:.2f}s")

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            log.info(f"   [{scan_uuid[:8]}] Temp directory cleaned up")

    except Exception as e:
        elapsed = time.perf_counter() - t_start
        log.error(f"❌  [{scan_uuid[:8]}] Job FAILED after {elapsed:.2f}s: {e}")
        traceback.print_exc()
        # Try to mark as failed — if the constraint rejects it, log which values ARE allowed
        # and add "failed" to the scans_status_check constraint in Supabase dashboard.
        for fail_status in ("failed", "error"):
            try:
                supabase.table("scans") \
                    .update({"status": fail_status}) \
                    .eq("scan_uuid", scan_uuid) \
                    .execute()
                log.info(f"   [{scan_uuid[:8]}] Status → {fail_status} (written to DB)")
                break
            except Exception as db_err:
                log.warning(f"   [{scan_uuid[:8]}] Status '{fail_status}' rejected by constraint: {db_err}")
        else:
            log.error(
                f"   [{scan_uuid[:8]}] Could not mark job as failed. "
                f"Add 'failed' or 'error' to the scans_status_check constraint in Supabase."
            )


# ── Main polling loop ─────────────────────────────────────────
async def poll_jobs():
    """Continuously polls Supabase for jobs with status='processing'."""
    log.info("=" * 60)
    log.info("🔄  Vision Background Worker starting")
    log.info(f"    Supabase URL : {settings.SUPABASE_URL}")
    log.info(f"    Poll interval: {POLL_INTERVAL_IDLE}s (idle) / {POLL_INTERVAL_ERROR}s (on error)")
    log.info("=" * 60)

    # Test connectivity before entering the loop
    connected = test_supabase_connection()
    if not connected:
        log.error("⛔  Worker cannot start — Supabase unreachable. Will retry every 30s.")

    consecutive_errors = 0

    while True:
        try:
            supabase = get_supabase()
            result = supabase.table("scans") \
                .select("scan_uuid") \
                .eq("status", "processing") \
                .limit(1) \
                .execute()

            consecutive_errors = 0  # reset on success

            if result.data and len(result.data) > 0:
                scan_uuid = result.data[0]["scan_uuid"]
                log.info(f"📥  New job found: {scan_uuid}")
                await process_job(scan_uuid)
            else:
                # No jobs — quiet wait (don't spam logs)
                await asyncio.sleep(POLL_INTERVAL_IDLE)

        except asyncio.CancelledError:
            log.info("🛑  Worker task cancelled — shutting down gracefully")
            raise

        except Exception as e:
            consecutive_errors += 1
            wait = min(POLL_INTERVAL_ERROR * consecutive_errors, 60)
            log.error(f"⚠️   Poll error (attempt #{consecutive_errors}): {e}")
            if consecutive_errors == 1:
                # First error — print full details
                log.error(f"    URL being contacted: {settings.SUPABASE_URL}")
                log.error(f"    Key prefix: {settings.SUPABASE_KEY[:30]}...")
                traceback.print_exc()
            log.info(f"    Retrying in {wait}s...")
            await asyncio.sleep(wait)
