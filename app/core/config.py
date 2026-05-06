"""App configuration — reads from environment variables."""

import os
from dotenv import load_dotenv

# Try loading .env from the Backend folder first (deployment),
# then fall back to the project root .env (local development).
_backend_dir = os.path.join(os.path.dirname(__file__), "..", "..")
_root_dir    = os.path.join(_backend_dir, "..")
load_dotenv(dotenv_path=os.path.join(_backend_dir, ".env"), override=False)
load_dotenv(dotenv_path=os.path.join(_root_dir,    ".env"), override=False)

class Settings:
    # Supabase — must match Frontend/.env.local
    SUPABASE_URL: str = os.getenv("NEXT_PUBLIC_SUPABASE_URL", "")
    SUPABASE_KEY: str = os.getenv("NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY", "")
    SUPABASE_DB_PASSWORD: str = os.getenv("VISION_DATABASE_PASSWORD", "")

    # CORS — allow frontend origins
    ALLOWED_ORIGINS: list = [
        "http://localhost:3000",
        "http://localhost:3001",
        "https://*.vercel.app",
        "*",
    ]

    # Model path — Model/ is now inside Backend/ (sibling of app/)
    MODEL_PATH: str = os.path.join(
        os.path.dirname(__file__), "..", "..", "Model", "retinascan_best.pth"
    )

    # Image Processing path — Image_Processing/ is now inside Backend/ (sibling of app/)
    IMAGE_PROCESSING_PATH: str = os.path.join(
        os.path.dirname(__file__), "..", "..", "Image_Processing"
    )

    # Diabetic Retinopathy grades
    DR_GRADES: dict = {
        0: "No DR",
        1: "Mild DR",
        2: "Moderate DR",
        3: "Severe DR",
        4: "Proliferative DR",
    }

    DR_DESCRIPTIONS: dict = {
        0: "No signs of diabetic retinopathy detected.",
        1: "Mild non-proliferative diabetic retinopathy. Microaneurysms present.",
        2: "Moderate non-proliferative diabetic retinopathy. More vessels blocked.",
        3: "Severe non-proliferative diabetic retinopathy. Many blocked vessels.",
        4: "Proliferative diabetic retinopathy. Advanced stage, risk of blindness.",
    }


settings = Settings()
