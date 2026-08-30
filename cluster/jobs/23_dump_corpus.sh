#!/bin/bash
#SBATCH --job-name=gc-dump-corpus
#SBATCH --partition=a16
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm-%j-dump-corpus.out
#SBATCH --error=logs/slurm-%j-dump-corpus.err

# Dump the corpus so it can be restored somewhere the demo can reach.
#
# The demo's brief generator reads market signals straight out of Postgres, so
# a hosted generator needs the corpus, not just the model weights. The cluster
# database only exists while a job is running: $WORK/pgdata is the data at
# rest and the server comes up here, so a dump cannot be taken from the login
# node.
#
# Read-only. It takes a copy and changes nothing, so it is safe to run while
# other work is going on, subject to the usual one-postmaster rule that
# _start_services.sh enforces.
#
#   sbatch cluster/jobs/23_dump_corpus.sh
#
# The scp line to fetch the result is printed at the end of the log.

set -euo pipefail
source /vol/bitbucket/$USER/venvs/gc/bin/activate
cd "$SLURM_SUBMIT_DIR"
source cluster/jobs/_start_services.sh

OUT_DIR="/vol/bitbucket/$USER/dumps"
STAMP="$(date +%Y%m%d_%H%M)"
DUMP="$OUT_DIR/corpus_${STAMP}.dump"
mkdir -p "$OUT_DIR"

echo "=== Corpus dump ==="
echo "Node: $(hostname)  Start: $(date)"

# What is going into the dump, so a truncated restore is obvious later.
echo
echo "--- row counts at dump time ---"
psql -h localhost -p 5433 -d greeting_cards -c "
SELECT relname AS table, n_live_tup AS approx_rows
FROM pg_stat_user_tables
ORDER BY n_live_tup DESC;
"

echo "--- extensions in use ---"
psql -h localhost -p 5433 -d greeting_cards -c "SELECT extname, extversion FROM pg_extension ORDER BY extname;"

# --no-owner and --no-privileges because the roles here do not exist on the
# target. Without them the restore stops on every GRANT and ALTER OWNER that
# names rt325, which no other database has ever heard of.
#
# Custom format rather than plain SQL: the CLIP embeddings are most of the
# bytes and this compresses them, and pg_restore can then be pointed at a
# single table if only part of it is wanted.
echo
echo "--- dumping ---"
pg_dump -h localhost -p 5433 -d greeting_cards \
    --format=custom --compress=9 \
    --no-owner --no-privileges \
    --file "$DUMP"

# A dump that fails halfway still leaves a file, so check it is readable as an
# archive rather than trusting the exit code alone.
echo "--- verifying the archive ---"
pg_restore --list "$DUMP" > "$DUMP.toc"
echo "table definitions in archive: $(grep -c 'TABLE DATA' "$DUMP.toc" || true)"
echo "size: $(du -h "$DUMP" | cut -f1)"

echo
echo "=== Done: $(date) ==="
cat <<INSTRUCTIONS

Fetch it:
  scp gpucluster:$DUMP ~/Desktop/

Restore into the Railway Postgres. Check the extension first, because the
schema needs pgvector and the restore stops at the first statement without it:

  psql "\$DATABASE_URL" -c 'CREATE EXTENSION IF NOT EXISTS vector;'
  psql "\$DATABASE_URL" -c 'CREATE EXTENSION IF NOT EXISTS pgcrypto;'

If the vector line fails, the image has no pgvector and the service needs
swapping for Railway's pgvector template before going further.

  pg_restore --no-owner --no-privileges --dbname "\$DATABASE_URL" corpus_${STAMP}.dump

Then add the demo choice table, which has never been applied here and so is
not in this dump:

  psql "\$DATABASE_URL" -f migrations/0006_demo_choice_events.sql

INSTRUCTIONS
