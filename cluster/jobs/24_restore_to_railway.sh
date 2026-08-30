#!/bin/bash
#SBATCH --job-name=gc-restore-railway
#SBATCH --partition=a16
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:40:00
#SBATCH --output=logs/slurm-%j-restore-railway.out
#SBATCH --error=logs/slurm-%j-restore-railway.err

# Push a corpus dump into the hosted Postgres the demo reads from.
#
# Runs here rather than on a laptop for two reasons: the dump is already on
# this filesystem, and the client here is the same major version as the server
# it is going to, which is the thing that decides whether a custom format
# archive can be read at all.
#
# It does not need the cluster's own Postgres. pg_restore is a client and it
# connects outward, so _start_services.sh is deliberately not sourced.
#
#   sbatch cluster/jobs/24_restore_to_railway.sh /vol/bitbucket/$USER/dumps/corpus_YYYYMMDD_HHMM.dump
#
# The connection string is read from ~/.gc_railway_dsn, which should be mode
# 600 and is never printed. Passing it as an argument would put the password
# into the job's name, into squeue output for anyone on the cluster, and into
# this log.

set -euo pipefail
source /vol/bitbucket/$USER/venvs/gc/bin/activate
cd "$SLURM_SUBMIT_DIR"

# pg_restore lives with the rest of the Postgres install, not in the venv.
PG_PREFIX="/vol/bitbucket/$USER/pg_install"
if [ -d "$PG_PREFIX/bin" ]; then
    export PATH="$PG_PREFIX/bin:$PATH"
    export LD_LIBRARY_PATH="$PG_PREFIX/lib:${LD_LIBRARY_PATH:-}"
fi

DUMP="${1:-}"
if [ -z "$DUMP" ] || [ ! -f "$DUMP" ]; then
    echo "Usage: sbatch $0 /path/to/corpus_YYYYMMDD_HHMM.dump" >&2
    exit 1
fi

DSN_FILE="$HOME/.gc_railway_dsn"
if [ ! -f "$DSN_FILE" ]; then
    echo "No $DSN_FILE. Put the connection string in it and chmod 600." >&2
    exit 1
fi
DSN="$(tr -d '\r\n' < "$DSN_FILE")"

echo "=== Restore to hosted Postgres ==="
echo "Node: $(hostname)  Start: $(date)"
echo "Dump: $DUMP ($(stat -c %s "$DUMP") bytes)"
echo "Client: $(pg_restore --version)"

# Everything below prints the host but never the string holding the password.
echo "Target: $(printf '%s' "$DSN" | sed -E 's|^.*@|@|')"
echo "Server: $(psql "$DSN" -tAc 'SELECT version()' | cut -c1-60)"

# The schema's first migration creates this, but the extension has to exist
# before anything referencing the type is restored, and a target without it
# fails on an early statement rather than at the end.
echo
echo "--- extensions ---"
psql "$DSN" -c 'CREATE EXTENSION IF NOT EXISTS vector;'
psql "$DSN" -c 'CREATE EXTENSION IF NOT EXISTS pgcrypto;'
psql "$DSN" -tAc "SELECT extname || ' ' || extversion FROM pg_extension ORDER BY extname"

echo
echo "--- restoring ---"
# --no-owner and --no-privileges because rt325 does not exist over there.
# Not --clean: this is meant for an empty database, and dropping objects that
# a live demo might be reading is a worse failure than refusing to start.
# --exit-on-error so a broken restore stops here instead of leaving a half
# populated corpus that looks fine until a brief comes back wrong.
pg_restore --no-owner --no-privileges --exit-on-error \
    --dbname "$DSN" "$DUMP"

echo
echo "--- what landed ---"
psql "$DSN" -c "
SELECT 'listings'           AS table, count(*) FROM listings
UNION ALL SELECT 'listing_features',    count(*) FROM listing_features
UNION ALL SELECT 'listing_images',      count(*) FROM listing_images
UNION ALL SELECT 'saleability_labels',  count(*) FROM saleability_labels
UNION ALL SELECT 'generated_cards',     count(*) FROM generated_cards
ORDER BY 1;
"
psql "$DSN" -tAc "SELECT 'embeddings: ' || count(*) FROM listing_features WHERE clip_embedding IS NOT NULL"

# The demo's own table, which has never been applied on this cluster and so is
# not in any dump taken from it.
echo
echo "--- migration 0006 ---"
psql "$DSN" -f migrations/0006_demo_choice_events.sql
psql "$DSN" -tAc "SELECT 'demo_choice_events exists: ' || to_regclass('public.demo_choice_events')::text"

echo
echo "=== Done: $(date) ==="
