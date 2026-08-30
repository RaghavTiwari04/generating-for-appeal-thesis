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
#   HF_TOKEN            FLUX.1-dev is a gated repo. The token has to belong to
#                       an account that has accepted the licence; merely having
#                       a token is not enough.
#
# Optional:
#   PORT                default 8000
#   HF_HOME             where the 34 GB of weights are cached. Defaults to
#                       /workspace when that exists, since that is where
#                       persistent storage mounts on the usual GPU hosts.
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

# 2. Decide where the weights are cached before anything imports a Hugging
#    Face library, since they read HF_HOME once at import and never again.
#
#    This costs real money to get wrong. Flux is 34 GB, and on a rented box
#    the home directory is usually the ephemeral container disk, so an unset
#    HF_HOME means downloading it again on every single start. Persistent
#    storage mounts at /workspace on the common providers, so prefer that when
#    it is there rather than guessing.
if [ -z "${HF_HOME:-}" ]; then
    if [ -d /workspace ] && [ -w /workspace ]; then
        export HF_HOME=/workspace/.cache/huggingface
    else
        export HF_HOME="$PWD/.cache/huggingface"
    fi
    echo "HF_HOME was not set, using $HF_HOME"
fi
mkdir -p "$HF_HOME"
echo "Weights cache: $HF_HOME"

FLUX_CACHE="$HF_HOME/hub/models--black-forest-labs--FLUX.1-dev"
if [ -d "$FLUX_CACHE" ]; then
    echo "  Flux already cached here, nothing to download"
else
    # -Pk rather than --output=avail, which is GNU only and these images vary.
    FREE_GB="$(df -Pk "$HF_HOME" 2>/dev/null | awk 'NR==2 {print int($4/1048576)}')"
    echo "  Flux is not cached yet: about 34 GB to fetch. Free here: ${FREE_GB:-unknown} GB"

    if [ -n "${FREE_GB:-}" ] && [ "$FREE_GB" -lt 40 ]; then
        echo "FATAL: ${FREE_GB} GB free is not enough for Flux." >&2
        echo "Point HF_HOME at a larger disk, or grow this one." >&2
        exit 1
    fi

    case "$HF_HOME" in
        /root/*|/home/*|/tmp/*)
            echo "  WARNING: on a rented host this path is usually the disk that is" >&2
            echo "  wiped when the pod stops, so this download would repeat every" >&2
            echo "  session. Point HF_HOME at persistent storage instead." >&2
            ;;
    esac
fi

# 3. Confirm the weights can actually be fetched before anything slow starts.
#    FLUX.1-dev is gated, so this fails for two different reasons that look
#    identical from inside a fifteen minute model load: no token, or a token
#    whose account never accepted the licence. Asking the API costs one request
#    and turns both into an error in the first second.
python - <<'PY'
import sys

from common.config import settings

token = settings.hf_token
if not token:
    sys.exit(
        "HF_TOKEN is not set.\n"
        "black-forest-labs/FLUX.1-dev is a gated repository, so the weights "
        "cannot be downloaded anonymously. Create a token at "
        "https://huggingface.co/settings/tokens, accept the licence on the "
        "model page with that same account, and set HF_TOKEN."
    )

repo = settings.flux_model_id
try:
    from huggingface_hub import HfApi
    from huggingface_hub.utils import GatedRepoError, HfHubHTTPError

    api = HfApi()
    # auth_check, not model_info. A gated repo still serves its metadata to
    # anyone, so model_info returns happily for a token that is pure nonsense
    # and the check proves nothing. auth_check asks the question that actually
    # matters, which is whether this token may read the files.
    if hasattr(api, "auth_check"):
        api.auth_check(repo, token=token)
    else:
        # Older huggingface_hub. Downloading the smallest file in the repo
        # exercises the same permission.
        from huggingface_hub import hf_hub_download

        hf_hub_download(repo, "model_index.json", token=token)
    print(f"Hugging Face access to {repo} confirmed")
except GatedRepoError:
    sys.exit(
        f"This token cannot read {repo}.\n"
        "Either the licence has not been accepted by the account that owns "
        "the token, or the token itself is wrong: Hugging Face reports both "
        "the same way. Open the model page signed in as that account, check "
        "it says access granted, and check the token was copied whole."
    )
except HfHubHTTPError as exc:
    sys.exit(f"Hugging Face rejected the token: {exc}")
except Exception as exc:
    # A network wobble should not stop a demo from starting. The download
    # retries on its own, and the failures worth catching here are the two
    # above, which are the ones that waste fifteen minutes.
    print(f"WARNING: could not verify Hugging Face access ({exc}); continuing")
PY

# 4. The corpus has to be reachable: briefs are built from market signals read
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

# 5. Load Flux before opening the port. Roughly fifteen minutes; a visitor who
#    arrives during it would otherwise wait for all of it inside one request.
echo "Loading Flux (this takes several minutes)…"
python -u -c "
from generation.image.diffusion import get_runner
r = get_runner()
r._load_pipeline()
print('Flux loaded and resident', flush=True)
"

# 6. One worker. The job store is in-process, so a second worker would answer
#    status requests for jobs it has never heard of.
echo "Serving on 0.0.0.0:$PORT"
exec uvicorn app.api:app --host 0.0.0.0 --port "$PORT" --workers 1
