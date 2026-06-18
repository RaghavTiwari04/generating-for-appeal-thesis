#!/usr/bin/env bash
# Run on your LAPTOP to tunnel Postgres + MinIO to Imperial cluster.
# Usage: bash cluster/ssh_tunnel.sh USERNAME
#
# This forwards your local Postgres (5432) and MinIO (9000) so the
# cluster can reach them at localhost:5432 and localhost:9000.
#
# Keep this terminal open while working on the cluster.

set -euo pipefail

USERNAME="${1:?Usage: bash cluster/ssh_tunnel.sh YOUR_DOC_SHORTCODE}"
HOST="gpucluster2.doc.ic.ac.uk"

echo "Opening SSH tunnels to $HOST as $USERNAME..."
echo "  Remote :5432 → local Postgres"
echo "  Remote :9000 → local MinIO"
echo ""
echo "Keep this terminal open. Ctrl+C to close tunnels."
echo ""

ssh -N \
    -R 5432:localhost:5432 \
    -R 9000:localhost:9000 \
    "${USERNAME}@${HOST}"
