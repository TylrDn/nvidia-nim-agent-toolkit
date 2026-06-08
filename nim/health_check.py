"""NIM service readiness probe.

Polls NIM endpoint until ready or timeout exceeded.
Used by docker-compose healthcheck and CI startup scripts.
"""
from __future__ import annotations

import os
import time
import logging

import httpx
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


def check_nim_health(
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: int = 30,
    interval: int = 2,
) -> bool:
    """Poll /v1/models until NIM responds 200 or timeout.

    Args:
        base_url: NIM base URL (falls back to NIM_BASE_URL env var).
        api_key: NIM API key (falls back to NIM_API_KEY env var).
        timeout: Max seconds to wait.
        interval: Seconds between retries.

    Returns:
        True if healthy, False if timed out.
    """
    url = (base_url or os.environ["NIM_BASE_URL"]).rstrip("/") + "/models"
    key = api_key or os.environ["NIM_API_KEY"]
    headers = {"Authorization": f"Bearer {key}"}

    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = httpx.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                logger.info("NIM healthy at %s", url)
                return True
        except httpx.RequestError as exc:
            logger.debug("NIM not ready yet: %s", exc)
        time.sleep(interval)

    logger.error("NIM health check timed out after %ds", timeout)
    return False


if __name__ == "__main__":
    import sys
    healthy = check_nim_health()
    sys.exit(0 if healthy else 1)
