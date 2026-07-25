#!/usr/bin/env bash
# Sourced by SLURM job scripts to start Postgres + MinIO on the compute node.
# Usage (in job script): source cluster/jobs/_start_services.sh

WORK="/vol/bitbucket/$USER"
export HF_HOME="$WORK/.cache/huggingface"

# Load .env so API keys (HF_TOKEN, ANTHROPIC_API_KEY) are available to subprocesses
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi
PG_DIR="$WORK/pgdata"
MINIO_DIR="$WORK/minio-data"
MINIO_BIN="$WORK/bin/minio"

# Add Postgres binaries if installed via conda-forge prefix
PG_PREFIX="$WORK/pg_install"
if [ -d "$PG_PREFIX/bin" ]; then
    export PATH="$PG_PREFIX/bin:$PATH"
    export LD_LIBRARY_PATH="$PG_PREFIX/lib:${LD_LIBRARY_PATH:-}"
fi

# Start Postgres — pg_ctl handles stale PIDs natively.
# Do NOT manually rm postmaster.pid — causes NFS cache invalidation issues
# that make Postgres shut down immediately after start.
pg_ctl -D "$PG_DIR" -l "$PG_DIR/postgres.log" start 2>/dev/null || true

# Poll until Postgres actually answers. Crash recovery on NFS can fsync for
# several minutes, far longer than pg_ctl's own start timeout, so a fixed
# sleep here silently hands a not-yet-ready DB to the pipeline.
pg_wait() {
    local deadline=$((SECONDS + ${1:-600}))
    while [ $SECONDS -lt $deadline ]; do
        if pg_isready -h localhost -p 5433 -d greeting_cards -q 2>/dev/null; then
            return 0
        fi
        sleep 5
    done
    return 1
}

if ! pg_wait 600; then
    echo "Postgres not ready after 600s, restarting cleanly..."
    # -m fast is a CLEAN shutdown; -m immediate would force crash recovery
    # on the next start and make the problem compound run over run.
    pg_ctl -D "$PG_DIR" stop -m fast -w -t 300 2>/dev/null || true
    pg_ctl -D "$PG_DIR" -l "$PG_DIR/postgres.log" start 2>/dev/null || true
    if ! pg_wait 600; then
        echo "FATAL: Postgres would not come up; aborting before generation." >&2
        return 1 2>/dev/null || exit 1
    fi
fi
echo "Postgres ready after ${SECONDS}s"

# Start MinIO — use 9002 to avoid port conflicts with system services
MINIO_PORT="${MINIO_PORT:-9002}"
MINIO_CONSOLE_PORT="${MINIO_CONSOLE_PORT:-9003}"
export MINIO_ROOT_USER=minioadmin
export MINIO_ROOT_PASSWORD=minioadmin
# Kill stale MinIO on this port
fuser -k "$MINIO_PORT/tcp" 2>/dev/null || true
sleep 1
"$MINIO_BIN" server "$MINIO_DIR" --address ":$MINIO_PORT" --console-address ":$MINIO_CONSOLE_PORT" &>/dev/null &
sleep 2
export MINIO_ENDPOINT="http://localhost:$MINIO_PORT"

echo "Services: Postgres :5433, MinIO :$MINIO_PORT"
