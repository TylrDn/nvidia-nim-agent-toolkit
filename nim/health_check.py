"""NIM service readiness probe."""
from __future__ import annotations

import logging
import os
import time

import httpx

logger = logging.getLogger(__name__)

NIM_BASE_URL = os.getenv("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")


def is_nim_ready(base_url: str = NIM_BASE_URL, timeout: int = 5) -> bool:
    """Return True if the NIM ``/models`` endpoint responds successfully.

    Args:
        base_url: NIM endpoint base URL.
        timeout: Request timeout in seconds.

    Returns:
        bool: True when the endpoint returns HTTP 200.
    """
    try:
        response = httpx.get(f"{base_url}/models", timeout=timeout)
        return response.status_code == 200
    except httpx.HTTPError as exc:
        logger.debug("NIM readiness check failed: %s", exc)
        return False


async def check_nim_health(base_url: str = NIM_BASE_URL, timeout: int = 5) -> bool:
    """Async readiness probe against the NIM ``/models`` endpoint.

    Args:
        base_url: NIM endpoint base URL.
        timeout: Request timeout in seconds.

    Returns:
        bool: True when the endpoint returns HTTP 200.
    """
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(f"{base_url}/models")
            return response.status_code == 200
    except httpx.HTTPError as exc:
        logger.debug("NIM async readiness check failed: %s", exc)
        return False


def wait_for_nim(
    base_url: str = NIM_BASE_URL,
    max_wait: int = 120,
    interval: int = 5,
) -> None:
    """Block until NIM is ready or raise ``TimeoutError``.

    Args:
        base_url: NIM endpoint base URL.
        max_wait: Maximum seconds to wait.
        interval: Seconds between attempts.

    Raises:
        TimeoutError: If NIM does not become ready within ``max_wait``.
    """
    elapsed = 0
    while elapsed < max_wait:
        if is_nim_ready(base_url):
            return
        logger.info("NIM not ready — retrying in %ds...", interval)
        time.sleep(interval)
        elapsed += interval
    raise TimeoutError(f"NIM at {base_url} did not become ready within {max_wait}s")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    wait_for_nim()
    logger.info("NIM is ready.")
