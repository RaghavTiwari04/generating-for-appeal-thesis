#!/usr/bin/env bash
# Run on the CLUSTER to set up Postgres + MinIO and import data.
# Run from a shell server or interactive session — NOT gpucluster2/3 head nodes.
#
# Usage: bash cluster/import_cluster.sh
#
# Prerequisites:
#   - virtualenv activated: source /vol/bitbucket/$USER/venvs/gc/bin/activate
#   - cluster_data/ directory present (from export_local.sh)

set -euo pipefail

WORK="/vol/bitbucket/$USER"
DATA_DIR="cluster_data"
PG_DIR="$WORK/pgdata"
MINIO_DIR="$WORK/minio-data"

if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: $DATA_DIR not found. Run export_local.sh on laptop first, then scp here."
    exit 1
fi

# ── Install Postgres (userspace, via conda-forge into isolated prefix) ──────
echo "=== Setting up Postgres ==="
PG_PREFIX="$WORK/pg_install"
if ! command -v initdb &>/dev/null && [ ! -f "$PG_PREFIX/bin/initdb" ]; then
    echo "Postgres not found. Installing via conda-forge into $PG_PREFIX ..."
    if command -v conda &>/dev/null; then
        conda create -y -p "$PG_PREFIX" -c conda-forge postgresql pgvector
    else
        echo "conda not found. Bootstrapping miniconda..."
        MINICONDA="$WORK/miniconda.sh"
        curl -sSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o "$MINICONDA"
        bash "$MINICONDA" -b -p "$WORK/miniconda3"
        rm "$MINICONDA"
        export PATH="$WORK/miniconda3/bin:$PATH"
        conda create -y -p "$PG_PREFIX" -c conda-forge postgresql pgvector
    fi
fi
if [ -d "$PG_PREFIX/bin" ]; then
    export PATH="$PG_PREFIX/bin:$PATH"
    export LD_LIBRARY_PATH="$PG_PREFIX/lib:${LD_LIBRARY_PATH:-}"
fi

# ── Initialize and start Postgres ───────────────────────────────────────────
if [ ! -f "$PG_DIR/PG_VERSION" ]; then
    echo "Initializing Postgres data directory at $PG_DIR ..."
    initdb -D "$PG_DIR" --auth=trust
    # Use port 5433 to avoid conflicts with any system postgres
    sed -i "s/#port = 5432/port = 5433/" "$PG_DIR/postgresql.conf"
fi

echo "Starting Postgres..."
pg_ctl -D "$PG_DIR" -l "$PG_DIR/postgres.log" start 2>/dev/null || {
    echo "Postgres may already be running. Check: pg_ctl -D $PG_DIR status"
}
sleep 2

# Create user and database
createuser -p 5433 gc 2>/dev/null || true
createdb -p 5433 -O gc greeting_cards 2>/dev/null || true

# Enable pgvector extension
psql -p 5433 -U gc -d greeting_cards -c "CREATE EXTENSION IF NOT EXISTS vector;" 2>/dev/null || {
    echo "WARNING: pgvector extension not available. CLIP embedding queries may fail."
    echo "Install: pip install pgvector, or ask CSG for pgvector support."
}

# Import dump
echo "Importing DB dump..."
pg_restore -p 5433 -U gc -d greeting_cards --no-owner --clean --if-exists "$DATA_DIR/greeting_cards.dump" 2>/dev/null
echo "  Done."

# Verify
LISTING_COUNT=$(psql -p 5433 -U gc -d greeting_cards -t -c "SELECT COUNT(*) FROM listings;" 2>/dev/null || echo "?")
echo "  Listings in DB: $LISTING_COUNT"

# ── Set up MinIO ────────────────────────────────────────────────────────────
echo ""
echo "=== Setting up MinIO ==="
MINIO_BIN="$WORK/bin/minio"
if [ ! -f "$MINIO_BIN" ]; then
    mkdir -p "$WORK/bin"
    echo "Downloading MinIO server binary..."
    curl -sSL https://dl.min.io/server/minio/release/linux-amd64/minio -o "$MINIO_BIN"
    chmod +x "$MINIO_BIN"
fi

# Start MinIO
export MINIO_ROOT_USER=minioadmin
export MINIO_ROOT_PASSWORD=minioadmin
"$MINIO_BIN" server "$MINIO_DIR" --address ":9000" --console-address ":9001" &>/dev/null &
sleep 3

# Import images
echo "Importing images to MinIO..."
python3 -c "
from minio import Minio
import os, glob, sys

client = Minio('localhost:9000', access_key='minioadmin', secret_key='minioadmin', secure=False)

for bucket in ('greeting-cards', 'greeting-cards-raw'):
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)

files = sorted(glob.glob('$DATA_DIR/images/*'))
if not files:
    print('  No images found in $DATA_DIR/images/')
    sys.exit(0)

for i, fpath in enumerate(files):
    fname = os.path.basename(fpath)
    # Reconstruct S3 key: images_ab_cd_hash -> images/ab/cd/hash
    parts = fname.split('_', 3)
    if len(parts) == 4:
        key = '/'.join(parts)
    else:
        key = fname
    client.fput_object('greeting-cards', key, fpath)
    if (i + 1) % 200 == 0:
        print(f'  {i+1}/{len(files)} images imported...')

print(f'  Total: {len(files)} images imported')
"

echo ""
echo "=== Import complete ==="
echo ""
echo "Services running:"
echo "  Postgres: localhost:5433 (user=gc, db=greeting_cards)"
echo "  MinIO:    localhost:9000"
echo ""
echo "Make sure .env has:"
echo "  POSTGRES_PORT=5433"
echo "  MINIO_ENDPOINT=http://localhost:9000"
