#!/usr/bin/env bash
# Run on your LAPTOP to export DB + images for cluster transfer.
# Usage: bash cluster/export_local.sh
#
# Produces: cluster_data/ directory ready to scp to cluster

set -euo pipefail

OUT="cluster_data"
mkdir -p "$OUT/images"

echo "=== Exporting Postgres dump ==="
docker exec gc_postgres pg_dump -U gc --format=custom greeting_cards > "$OUT/greeting_cards.dump"
echo "  DB dump: $(du -h "$OUT/greeting_cards.dump" | cut -f1)"

echo "=== Exporting images from MinIO ==="
# Use mc (MinIO client) or Python to pull images
python -c "
from common.storage import _client
from common.config import settings
import os

client = _client()
bucket = settings.minio_bucket
objects = client.list_objects(bucket, prefix='images/', recursive=True)
count = 0
for obj in objects:
    dest = os.path.join('$OUT/images', obj.object_name.replace('/', '_'))
    client.fget_object(bucket, obj.object_name, dest)
    count += 1
    if count % 100 == 0:
        print(f'  {count} images exported...')
print(f'  Total: {count} images')
"

echo ""
echo "=== Export complete ==="
echo "Transfer to cluster with:"
echo "  scp -r cluster_data/ SHORTCODE@gpucluster2.doc.ic.ac.uk:/vol/bitbucket/\$USER/masters_thesis/"
