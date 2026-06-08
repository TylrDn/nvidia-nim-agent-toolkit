"""NIM service readiness probe."""
from __future__ import annotations

import os

import httpx
from dotenv import load_dotenv

load_dotenv()


def check_nim_health(base_url: str | None = None, api_key: str | None = None) -> bool:
    """Return True if the NIM endpoint is reachable and returns a valid model list."""
    url = base_url or os.environ.get("NIM_BASE_URL", "")
    key = api_key or os.environ.get("NIM_API_KEY", "")
    if not (url and key):
        print("[health_check] NIM_BASE_URL or NIM_API_KEY not set.")
        return False
    try:
        resp = httpx.get(
            f"{url}/models",
            headers={"Authorization": f"Bearer {key}"},
            timeout=10,
        )
        resp.raise_for_status()
        models = resp.json().get("data", [])
        print(f"[health_check] NIM healthy — {len(models)} model(s) available.")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[health_check] NIM unreachable: {exc}")
        return False


if __name__ == "__main__":
    import sys
    sys.exit(0 if check_nim_health() else 1)
