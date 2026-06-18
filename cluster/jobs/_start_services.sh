#!/usr/bin/env bash
# Sourced by SLURM job scripts to start Postgres + MinIO on the compute node.
# Usage (in job script): source cluster/jobs/_start_services.sh

WORK="/vol/bitbucket/$USER"
PG_DIR="$WORK/pgdata"
MINIO_DIR="$WORK/minio-data"
MINIO_BIN="$WORK/bin/minio"

# Add Postgres binaries if installed via conda-forge prefix
PG_PREFIX="$WORK/pg_install"
if [ -d "$PG_PREFIX/bin" ]; then
    export PATH="$PG_PREFIX/bin:$PATH"
    export LD_LIBRARY_PATH="$PG_PREFIX/lib:${LD_LIBRARY_PATH:-}"
fi

# Start Postgres
pg_ctl -D "$PG_DIR" -l "$PG_DIR/postgres.log" start 2>/dev/null || true
sleep 2

# Start MinIO
export MINIO_ROOT_USER=minioadmin
export MINIO_ROOT_PASSWORD=minioadmin
"$MINIO_BIN" server "$MINIO_DIR" --address ":9000" --console-address ":9001" &>/dev/null &
sleep 2

echo "Services: Postgres :5433, MinIO :9000"
