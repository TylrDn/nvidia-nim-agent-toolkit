"""NIM service readiness probe — usable as a Docker HEALTHCHECK command."""
import sys

from nim.client import NIMClient


def main() -> None:
    result = NIMClient().health_check()
    if result["status"] == "ok":
        print("NIM healthy")
        sys.exit(0)
    else:
        print(f"NIM not healthy: {result}")
        sys.exit(1)


if __name__ == "__main__":
    main()
