#!/usr/bin/env bash
# Serve the card generator on a rented GPU host, outside SLURM.
#
# The cluster job (cluster/jobs/07_serve_app.sh) assumes sbatch, /vol/cuda,
# /vol/bitbucket and an SSH tunnel for access. None of that exists on a rented
# box, so this is the same app with those assumptions removed and an access
# gate added, because the endpoint is reachable from the internet.
#
# Required:
#   GC_ACCESS_TOKEN     shared demo token; the app refuses to start without it
#   GC_ALLOWED_ORIGINS  origin the static frontend is served from
#   ANTHROPIC_API_KEY   brief generation
#
# Optional:
#   PORT                default 8000
#   POSTGRES_HOST/PORT  default localhost:5432
#   GC_BLOB_ROOT        where card images are written
#   GC_RATE_LIMIT       cards per token per hour, default 12
#
# Set secrets through the host's own secret store. Do not paste them into a
# shell here: the command lands in the shell history and, on most providers,
# in the machine's start-up log.

set -euo pipefail

PORT="${PORT:-8000}"

echo "=== Card generator, hosted ==="
echo "Start: $(date)"

# 1. Refuse to serve an ungated instance. This is the whole reason the file
#    exists rather than reusing the cluster job: a public endpoint spends API
#    credit and GPU time per request, and serves a LoRA trained on designs this
#    project holds no licence to publish (thesis Section 5.4).
python -c "from app.auth import require_gate_configured; require_gate_configured()"

if [ -z "${GC_ALLOWED_ORIGINS:-}" ]; then
    echo "WARNING: GC_ALLOWED_ORIGINS is unset, so no browser origin is allowed" >&2
    echo "         and the split frontend will be blocked by CORS." >&2
fi

# 2. The corpus has to be reachable: briefs are built from market signals read
#    out of Postgres, so an app with no database generates nothing.
python - <<'PY'
import sys
from common.db import engine
try:
    with engine().connect() as c:
        c.exec_driver_sql("SELECT 1")
    print("database reachable")
except Exception as exc:
    sys.exit(
        f"Cannot reach Postgres: {exc}\n"
        "The brief generator reads market signals from the corpus, so the "
        "database is not optional. Restore the dump onto the host's volume "
        "and point POSTGRES_HOST/POSTGRES_PORT at it."
    )
PY

# 3. Load Flux before opening the port. Roughly fifteen minutes; a visitor who
#    arrives during it would otherwise wait for all of it inside one request.
echo "Loading Flux (this takes several minutes)…"
python -u -c "
from generation.image.diffusion import get_runner
r = get_runner()
r._load_pipeline()
print('Flux loaded and resident', flush=True)
"

# 4. One worker. The job store is in-process, so a second worker would answer
#    status requests for jobs it has never heard of.
echo "Serving on 0.0.0.0:$PORT"
exec uvicorn app.api:app --host 0.0.0.0 --port "$PORT" --workers 1
