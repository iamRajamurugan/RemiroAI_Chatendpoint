import os
from typing import Optional, Dict, Any

from supabase import Client, create_client

_supabase_client: Optional[Client] = None


def get_supabase() -> Client:
    """Return a singleton Supabase client configured from environment variables.

    Requires SUPABASE_URL and SUPABASE_ANON_KEY to be set in the environment
    (for example via a .env file loaded by dotenv in graph.py).
    """

    global _supabase_client

    if _supabase_client is not None:
        return _supabase_client

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_ANON_KEY")

    if not url or not key:
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_ANON_KEY must be set in the environment "
            "to use Supabase-backed persistence."
        )

    _supabase_client = create_client(url, key)
    return _supabase_client


def _extract_user_id_from_auth_response(resp: Any) -> str:
    """Best-effort helper to extract a user ID from a Supabase auth response."""

    # supabase-py 2.x typically returns an object with `.user` attribute
    user = getattr(resp, "user", None)
    if not user and isinstance(resp, dict):
        user = resp.get("user")

    if not user:
        raise RuntimeError("Supabase auth response did not contain a user object.")

    # Try attribute first, then dict access
    user_id = getattr(user, "id", None)
    if not user_id and isinstance(user, dict):
        user_id = user.get("id")

    if not user_id:
        raise RuntimeError("Supabase user object did not contain an 'id' field.")

    return str(user_id)


def sign_up_user(email: str, password: str) -> Dict[str, Any]:
    """Create a new Supabase auth user and return its ID.

    This uses Supabase Auth (GoTrue) via supabase-py. The returned `user_id`
    can be used as the primary identifier in your own tables (profiles,
    chat_sessions, messages).
    """

    sb = get_supabase()
    resp = sb.auth.sign_up({"email": email, "password": password})
    user_id = _extract_user_id_from_auth_response(resp)
    return {"user_id": user_id, "raw": resp}


def sign_in_user(email: str, password: str) -> Dict[str, Any]:
    """Sign in an existing Supabase auth user and return its ID.

    On success, returns a dict with `user_id` and the raw auth response.
    Frontends can store any access/refresh tokens contained in the raw
    response if they need to call Supabase directly.
    """

    sb = get_supabase()
    resp = sb.auth.sign_in_with_password({"email": email, "password": password})
    user_id = _extract_user_id_from_auth_response(resp)
    return {"user_id": user_id, "raw": resp}
