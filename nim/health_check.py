"""NIM service readiness probe."""
from __future__ import annotations

import os
import time

import httpx


NIM_BASE_URL = os.getenv("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")


def is_nim_ready(base_url: str = NIM_BASE_URL, timeout: int = 5) -> bool:
    """Return True if NIM /models endpoint responds successfully."""
    try:
        r = httpx.get(f"{base_url}/models", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def wait_for_nim(
    base_url: str = NIM_BASE_URL,
    max_wait: int = 120,
    interval: int = 5,
) -> None:
    """Block until NIM is ready or raise TimeoutError."""
    elapsed = 0
    while elapsed < max_wait:
        if is_nim_ready(base_url):
            return
        print(f"[nim/health_check] NIM not ready — retrying in {interval}s...")
        time.sleep(interval)
        elapsed += interval
    raise TimeoutError(f"NIM at {base_url} did not become ready within {max_wait}s")


if __name__ == "__main__":
    wait_for_nim()
    print("[nim/health_check] NIM is ready ✓")
