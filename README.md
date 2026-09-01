# Generating for Appeal

An MSc thesis project. It reads a few thousand greeting cards that already
sell, learns something about what makes one worth choosing, then writes its own
briefs, draws the artwork with the greeting lettered into it, and tries to
guess which of its attempts a person would want.

The guessing is the part that could not be checked. There is no human
preference data anywhere in this project, so every label and every final score
comes from a model standing in for a shopper. That limit runs through
everything below.

## What it actually does

Six stages, in order.

**Scrape.** Birthday cards from two marketplaces. 3,906 listings, of which
3,491 carry a usable subtype.

**Deduplicate.** Print on demand catalogues list one artwork in many
colourways, and the copies score almost identically because they are the same
picture. Perceptual hashing plus embedding similarity collapses 3,906 listings
to 2,795 distinct designs.

**Label.** An LLM judge scores 2,468 designs on five dimensions. Purchase
intent uses Semantic Similarity Rating, where free text answers are mapped onto
a Likert scale by embedding similarity. The other four use a rubric judge.

**Train a predictor.** Ridge regression on frozen SigLIP image embeddings,
768 dimensions at 384 pixels. A larger MLP that also reads the headline text
and the occasion was built and measured, and the ridge on images alone beats it
on every head, which is a result rather than a shortcut.

**Generate.** A brief written from market signals read out of the corpus, then
FLUX.1-dev with a LoRA adapter trained on 240 of those designs, 60 per subtype.
The greeting is lettered into the artwork rather than typeset over it, because
that is how commercial cards do it.

**Rerank.** Generate several candidates, score them with the predictor, keep
the best.

## What the evaluation found

Four conditions, forty cards each, all judged the same way.

| Condition | Judged purchase intent |
|---|---|
| Naive prompt | 0.624 |
| Full pipeline | 0.685 |
| Pipeline, best of eight | 0.707 |
| Cards listed on marketplaces | 0.692 |

Three things follow, and two of them are negative.

The pipeline is worth building. The gap between prompting an image model
directly and running the whole thing is the largest effect in the study.

Reranking is not. Best of eight and a single candidate are statistically
equivalent, and that holds under every judge tested. Eight times the compute
buys nothing measurable.

Whether it matches the cards already on sale depends on who is asked. No judge
rated the pipeline below the marketplace cards, but whether the two count as
equivalent changes with the judge, so the honest claim is weaker than parity.

## Layout

```
app/          FastAPI service and the internal dev UI
site/         the public demo site, static, no build step
generation/   briefs, images, layout, inside messages
pipeline/     orchestration and reranking
models/       the appeal predictor, ridge and MLP
data/         scrapers, feature extraction, labelling
eval/         the four condition comparison and judge robustness
cluster/      SLURM jobs for everything that needs a GPU
deploy/       running the generator on a rented host
migrations/   database schema
tests/        300 of them
```

## Running it

Postgres with pgvector and an object store, then the package:

```bash
docker compose up -d
pip install -e .
pytest
```

Anything that touches a GPU runs as a SLURM job under `cluster/jobs/`,
numbered in the order they are meant to run. Each one carries a header
explaining what it does and why it is a batch job rather than something you
run interactively.

The demo site is plain HTML and CSS with a little Alpine, served by Caddy in a
container. `scripts/deploy_site.sh` publishes it.

## Things worth knowing before trusting any of this

The corpus carries no sales data. Every commercial field came back empty, so
"cards that sell" really means "cards a judge scored highly", and the baseline
is a sample of listings rather than proven bestsellers.

Nobody human ever scored a card. The predictor learns from judge labels and the
final comparison is judged by models too. Three of them, deliberately, because
one model judging its own training labels is circular. They agree about the
conditions and much less about individual cards.

The adapter is trained on designs belonging to the sellers who made them, and
nothing filters what comes out. The thesis treats that as a blocker to
deploying this as a product rather than a rough edge. The base image model is
licensed for non commercial use.

Lettering fails sometimes. The greeting is painted into the artwork by a
diffusion model, so it can misspell, and the only automatic checker available
cannot read brush script well enough to be trusted.
