#!/usr/bin/env bash
#
# Publish the static site to Railway.
#
# Run `railway link` once from site/ first, to pick the project and service.
#
# Two things this does that a bare `railway up` does not.
#
# It deploys site/ rather than the repo root. `railway up` uploads its working
# directory, and from the root Railway finds the backend Dockerfile and builds
# the generator image instead of the page.
#
# It stamps the commit into site/data/build.json so the footer says what is
# live. Without that the only record of which version is deployed is the
# Railway dashboard's clock, which tells you when but not what.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="$ROOT/site/data/build.json"

if ! command -v railway >/dev/null 2>&1; then
    echo "railway CLI not found. Install it, then run 'railway link' from site/." >&2
    exit 1
fi

commit="$(git -C "$ROOT" rev-parse --short HEAD)"
suffix=""

# A deploy from a dirty tree ships something no commit describes, which makes
# the stamp a lie rather than a record. Allowed, because a quick copy fix
# before a demo is a reasonable thing to want, but it is labelled.
if ! git -C "$ROOT" diff --quiet HEAD -- site; then
    suffix=" plus uncommitted changes"
    echo "WARNING: site/ has uncommitted changes, so what ships is not $commit." >&2
fi

restore() { git -C "$ROOT" checkout -- "$STAMP" 2>/dev/null || true; }
trap restore EXIT

cat > "$STAMP" <<JSON
{
  "commit": "${commit}${suffix}",
  "deployed_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
JSON

# Name the service explicitly. `railway up` otherwise deploys to whatever
# the directory is linked to, and that link moves without warning: running
# `railway add --database postgres` here repointed it at the new database, and
# the next deploy pushed this static site over the top of Postgres.
SERVICE="${RAILWAY_SERVICE:-generating-for-appeal}"

echo "Deploying site/ at ${commit}${suffix} to service $SERVICE"
cd "$ROOT/site"
railway up --service "$SERVICE" "$@"
