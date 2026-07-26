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

# After an unclean shutdown Postgres fsyncs EVERY file in the data directory
# before recovery. Over NFS that is a per-file round trip and took 90s+ per
# attempt here. syncfs() does it in one call per filesystem instead (PG 14+).
PG_START_OPTS=""
PG_MAJOR="$(pg_ctl --version 2>/dev/null | grep -oE '[0-9]+' | head -1)"
if [ -n "$PG_MAJOR" ] && [ "$PG_MAJOR" -ge 14 ] 2>/dev/null; then
    PG_START_OPTS="-c recovery_init_sync_method=syncfs"
fi

# Only one Postgres can own $PG_DIR, so a second job on the same node must
# attach to the running server rather than start its own. Record which case we
# are in: a job that did not start the server must not stop it on exit, or it
# kills the database under whichever job did.
if pg_isready -h localhost -p 5433 -q 2>/dev/null; then
    PG_STARTED_HERE=0
    echo "Postgres already running on this node — attaching, will not stop it"
else
    PG_STARTED_HERE=1
fi

# Start Postgres — pg_ctl handles stale PIDs natively.
# Do NOT manually rm postmaster.pid — causes NFS cache invalidation issues
# that make Postgres shut down immediately after start.
pg_start() {
    if [ -n "$PG_START_OPTS" ]; then
        pg_ctl -D "$PG_DIR" -l "$PG_DIR/postgres.log" -o "$PG_START_OPTS" start 2>/dev/null || true
    else
        pg_ctl -D "$PG_DIR" -l "$PG_DIR/postgres.log" start 2>/dev/null || true
    fi
}
pg_start

# Poll until Postgres actually answers. Crash recovery on NFS can fsync for
# several minutes, far longer than pg_ctl's own start timeout, so a fixed
# sleep here silently hands a not-yet-ready DB to the pipeline.
PG_WAIT_START=$SECONDS
pg_wait() {
    local deadline=$((SECONDS + ${1:-600}))
    local last=0
    while [ $SECONDS -lt $deadline ]; do
        if pg_isready -h localhost -p 5433 -d greeting_cards -q 2>/dev/null; then
            return 0
        fi
        # Report progress — recovery is silent and multi-minute, and without
        # this it is indistinguishable from a hang.
        if [ $((SECONDS - last)) -ge 30 ]; then
            last=$SECONDS
            echo "  ... still recovering ($((SECONDS - PG_WAIT_START))s): $(tail -1 "$PG_DIR/postgres.log" 2>/dev/null | cut -c1-100)"
        fi
        sleep 5
    done
    return 1
}

if ! pg_wait 600; then
    echo "Postgres not ready after 600s, restarting cleanly..."
    # -m fast is a CLEAN shutdown; -m immediate would force crash recovery
    # on the next start and make the problem compound run over run.
    pg_ctl -D "$PG_DIR" stop -m fast -w -t 120 2>/dev/null || true
    pg_start
    if ! pg_wait 600; then
        echo "FATAL: Postgres would not come up; aborting before generation." >&2
        return 1 2>/dev/null || exit 1
    fi
fi
echo "Postgres ready after $((SECONDS - PG_WAIT_START))s"

# Sourced, so this trap installs in the job script's own shell.
if [ "$PG_STARTED_HERE" = "1" ]; then
    trap 'pg_ctl -D "$PG_DIR" stop -m fast -w -t 20 2>/dev/null || true' EXIT
fi

# Start MinIO — use 9002 to avoid port conflicts with system services
MINIO_PORT="${MINIO_PORT:-9002}"
MINIO_CONSOLE_PORT="${MINIO_CONSOLE_PORT:-9003}"
export MINIO_ROOT_USER=minioadmin
export MINIO_ROOT_PASSWORD=minioadmin
# Reuse a live MinIO rather than killing it — the kill below would otherwise
# take down a concurrent job's blob store, and two servers cannot share
# $MINIO_DIR anyway.
if curl -sf --max-time 2 "http://localhost:$MINIO_PORT/minio/health/live" >/dev/null 2>&1; then
    echo "MinIO already running on :$MINIO_PORT — reusing"
else
    fuser -k "$MINIO_PORT/tcp" 2>/dev/null || true
    sleep 1
    "$MINIO_BIN" server "$MINIO_DIR" --address ":$MINIO_PORT" --console-address ":$MINIO_CONSOLE_PORT" &>/dev/null &
    sleep 2
fi
export MINIO_ENDPOINT="http://localhost:$MINIO_PORT"

echo "Services: Postgres :5433, MinIO :$MINIO_PORT"
