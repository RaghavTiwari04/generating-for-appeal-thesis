# The static site. Built and pushed with `bash scripts/deploy_site.sh`, which
# runs `railway up` from this directory rather than the repo root: `railway up`
# uploads its working directory, and from the root it would find the backend
# Dockerfile and build the generator instead of the page.
FROM caddy:2-alpine

COPY Caddyfile /etc/caddy/Caddyfile

# Copied file by file rather than `COPY . /srv`, for two reasons. A blind copy
# would serve this Dockerfile and the Caddyfile at fetchable URLs, and it would
# ship assets/cards/curation.json, which is the input to the image prep script
# and holds working notes rather than anything the page reads.
COPY index.html /srv/index.html
COPY css /srv/css
COPY js /srv/js
COPY data /srv/data

COPY assets/cards/hero /srv/assets/cards/hero
COPY assets/cards/examples /srv/assets/cards/examples
COPY assets/cards/quiz /srv/assets/cards/quiz
COPY assets/cards/failures /srv/assets/cards/failures

# Deliberately public: it records which card each image on the page is and what
# it scored, so anything shown here can be traced back to the evaluated set.
COPY assets/cards/manifest.json /srv/assets/cards/manifest.json

EXPOSE 8080
