"""NIM service readiness probe.

Can be run as a standalone script or imported as a module.
Exits with code 0 on success, 1 on failure — suitable for Docker
HEALTHCHECK and Kubernetes liveness probes.
"""
from __future__ import annotations

import sys
import logging

from nim.client import NIMClient

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def check(client: NIMClient | None = None) -> bool:
    """Return True if NIM is reachable and at least one model is listed."""
    c = client or NIMClient()
    result = c.health_check()
    if result["status"] == "ok":
        logger.info("NIM healthy — available models: %s", result.get("models"))
        return True
    logger.error("NIM unhealthy — %s", result.get("detail"))
    return False


if __name__ == "__main__":
    sys.exit(0 if check() else 1)
