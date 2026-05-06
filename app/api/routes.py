"""All API routes combined."""

from fastapi import APIRouter
from app.api import predict, scans

router = APIRouter()
router.include_router(predict.router, prefix="/predict", tags=["Prediction"])
router.include_router(scans.router, prefix="/scans", tags=["Scans"])
