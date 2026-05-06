"""Supabase client singleton."""

from supabase import create_client, Client
from app.core.config import settings

_supabase: Client = None


def get_supabase() -> Client:
    """Return cached Supabase client."""
    global _supabase
    if _supabase is None:
        _supabase = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
    return _supabase
