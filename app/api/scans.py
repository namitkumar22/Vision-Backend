"""Scans CRUD — fetch history for a user."""

from fastapi import APIRouter, HTTPException
from app.core.supabase import get_supabase

router = APIRouter()


@router.get("/user/{user_id}")
async def get_user_scans(user_id: str):
    """Get all scan results for a user."""
    supabase = get_supabase()

    result = (
        supabase.table("scans")
        .select("*")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .execute()
    )

    return {"scans": result.data or [], "count": len(result.data or [])}


@router.get("/{scan_uuid}")
async def get_scan(scan_uuid: str):
    """Get a single scan result by UUID."""
    supabase = get_supabase()

    result = (
        supabase.table("scans")
        .select("*")
        .eq("scan_uuid", scan_uuid)
        .execute()
    )

    if not result.data:
        raise HTTPException(status_code=404, detail="Scan not found")

    return result.data[0]
