---
name: fix-docker-compose
description: Invoke this agent when docker-compose.yml is missing from the repo root — specifically when `docker compose up` from the project root fails because the file only exists under deploy/.
model: inherit
readonly: false
---

# Fix Root docker-compose.yml — nvidia-nim-agent-toolkit

## Objective

Create a `docker-compose.yml` at the repository root that serves as a developer-friendly convenience entry point for `docker compose up`. The existing `deploy/docker-compose.yml` is the production-parameterized file and must not be modified. Both files must coexist with clearly documented purposes.

## Context

The project currently only has `deploy/docker-compose.yml`. Per cross-repo conventions, every repo in the NVIDIA SA portfolio must have a `docker-compose.yml` at the repo root so developers can run the full stack with a single `docker compose up` from the project root without navigating to subdirectories.

## Files to Create / Modify

1. `docker-compose.yml` — **Create** at repo root.
2. `README.md` — **Update** the "Getting Started" / "Running Locally" section to reference root compose.
3. `deploy/docker-compose.yml` — **Do not modify**, but verify it still works standalone.

## Step-by-Step Instructions

### Step 1 — Read `deploy/docker-compose.yml`

Before creating the root file, read `deploy/docker-compose.yml` in full to understand:
- Service names and their build contexts
- Port mappings
- Volume definitions
- Environment variable references
- Network definitions
- Any profiles defined

### Step 2 — Create `docker-compose.yml` at repo root

Create `docker-compose.yml` with the following structure (adapt service names and ports from the deploy version):

```yaml
# docker-compose.yml
# Root-level developer convenience compose file.
# For production deployment, see deploy/docker-compose.yml.
#
# Usage:
#   docker compose up              # start all services
#   docker compose up api          # start only the API service
#   docker compose down -v         # stop and remove volumes

version: "3.9"

services:
  api:
    build:
      context: .
      dockerfile: deploy/Dockerfile
    ports:
      - "8000:8000"
    env_file:
      - .env
    volumes:
      - .:/app
    depends_on:
      - nim-proxy
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 10s

  nim-proxy:
    image: nginx:alpine
    # Placeholder for NIM endpoint proxy in local dev.
    # In production, replace with actual NIM service or remove if using cloud NIM.
    ports:
      - "8080:80"
    volumes:
      - ./deploy/nginx.conf:/etc/nginx/nginx.conf:ro
    restart: unless-stopped

networks:
  default:
    name: nim-agent-toolkit-network
```

**Important rules for the root compose file:**
- Use `env_file: .env` (not inline environment variables with hardcoded values).
- Reference `deploy/Dockerfile` as the build dockerfile.
- Build context must be `.` (repo root), not `./deploy`.
- Include a `healthcheck` for the API service.
- Add a header comment block explaining this is the developer convenience file and pointing to `deploy/docker-compose.yml` for production.
- Do not duplicate volume definitions that are only needed in production (e.g., persistent database volumes). Keep the root file minimal and fast to start.

### Step 3 — Update README.md

Find the section in `README.md` that describes how to run the project locally. Update it to include:

```markdown
## Running Locally

### Quick Start (Docker Compose)

```bash
cp .env.template .env
# Edit .env with your NIM_API_KEY and other credentials
docker compose up
```

The API will be available at http://localhost:8000.

For production deployment configuration, see [`deploy/docker-compose.yml`](deploy/docker-compose.yml) and [`docs/architecture.md`](docs/architecture.md).
```

Remove any instructions that require `cd deploy && docker compose up`.

### Step 4 — Verify the deploy/ version still works

Read `deploy/docker-compose.yml` and confirm it has its own `context:` or `build:` path that remains valid when run from the `deploy/` directory. If it uses relative paths like `../Dockerfile`, those must still resolve correctly.

### Step 5 — Add a `.dockerignore` if missing

If `.dockerignore` doesn't exist at the repo root, create one:

```
.git
.env
*.pyc
__pycache__
.pytest_cache
.mypy_cache
.ruff_cache
notebooks/
docs/
*.md
tests/
evals/
```

## Acceptance Criteria

- [ ] `docker-compose.yml` exists at the repo root.
- [ ] Running `docker compose config` from the repo root exits with code 0 (valid YAML syntax).
- [ ] Root `docker-compose.yml` has a header comment explaining it is the developer convenience file.
- [ ] Root `docker-compose.yml` uses `env_file: .env` — no hardcoded secrets.
- [ ] Root `docker-compose.yml` references `deploy/Dockerfile` with build context `.`.
- [ ] `README.md` "Getting Started" section shows `docker compose up` from the root — not `cd deploy && docker compose up`.
- [ ] `deploy/docker-compose.yml` is unchanged.
- [ ] `git diff deploy/docker-compose.yml` shows no changes.
- [ ] The new file is committed with message: `chore: add root docker-compose.yml for developer convenience`.
