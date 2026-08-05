# Local development targets. The cluster equivalents, which are what produced
# the reported results, live in cluster/jobs/ as SLURM scripts.

.PHONY: up down reset-db install discover-selectors fonts weights \
        test test-fast coverage lint fmt typecheck \
        scrape download-images embed-features occasions dedup vlm-labels \
        pipeline train-predictor train-ridge eval-predictor sweep \
        run-card run-card-llm train-loras system-eval figures serve serve-docker

# ── Infrastructure ────────────────────────────────────────────────────────────
up:
	docker compose up -d
	@echo "Waiting for Postgres..."
	@until docker exec gc_postgres pg_isready -U gc -d greeting_cards; do sleep 1; done
	@echo "Postgres ready."

down:
	docker compose down

reset-db:
	docker exec gc_postgres psql -U gc -d greeting_cards -c "DROP SCHEMA public CASCADE; CREATE SCHEMA public;"
	docker exec gc_postgres psql -U gc -d greeting_cards -f /docker-entrypoint-initdb.d/0001_init.sql

# ── Setup ─────────────────────────────────────────────────────────────────────
install:
	uv pip install -e ".[dev]"

discover-selectors:
	python scripts/discover_selectors.py

fonts:
	python -m generation.layout.download_fonts

weights:
	python -c "from generation.image.upscaler import download_realesrgan_weights; download_realesrgan_weights()"

# ── Tests and code quality ────────────────────────────────────────────────────
test:
	pytest tests/ -v --tb=short

test-fast:
	pytest tests/ -v --tb=short -x -q

coverage:
	pytest tests/ --cov=. --cov-report=html --cov-report=term-missing

lint:
	ruff check .

fmt:
	ruff format .

typecheck:
	mypy data/ models/ generation/ pipeline/ eval/ common/ --ignore-missing-imports

# ── Data pipeline (run in order, or use `make pipeline`) ──────────────────────
# Sources are redbubble and greetings_island; see data/scrapers/run_scraper.py.
scrape:
	python -m data.scrapers.run_scraper --source redbubble --limit 1000

download-images:
	python -m data.scrapers.image_downloader --limit 10000

embed-features:
	python -m data.features.clip_embed
	python -m data.features.ocr
	python -m data.features.palette
	python -m data.features.image_complexity

# Zero-shot NLI assigns the birthday subtype; there is no trained classifier.
occasions:
	python -m data.features.occasion_nli

dedup:
	python -m data.features.dedup

# The vision-language judge that produces every training label.
vlm-labels:
	python -m data.labels.vlm_labels

pipeline:
	python -m data.pipeline_runner

pipeline-from-%:
	python -m data.pipeline_runner --from $*

# ── Predictor ─────────────────────────────────────────────────────────────────
# Ridge is what the pipeline uses; the MLP is kept for the comparison in the
# results chapter.
train-ridge:
	python -m models.predictor.ridge

train-predictor:
	python -m models.predictor.train --epochs 30 --batch-size 64

eval-predictor:
	python -m eval.predictor_eval_standalone

sweep:
	wandb sweep models/predictor/sweep.yaml

# ── Generation ────────────────────────────────────────────────────────────────
run-card:
	python -m pipeline.orchestrator \
		--occasion birthday/general \
		--tone warm-humorous \
		--n 8 --top-k 3

run-card-llm:
	python -m pipeline.orchestrator \
		--occasion birthday/general \
		--tone warm-humorous \
		--n 4 --top-k 2 \
		--scorer llm

# ── LoRA training (GPU required, 24GB+ VRAM) ──────────────────────────────────
# One adapter covers all four birthday subtypes: they share a visual vocabulary
# and the subtype comes from the prompt. See cluster/jobs/04_train_lora.sh.
train-lora-%:
	python -m generation.image.loras.train_lora --occasion $* --rank 32 --steps 1000 --lr 1e-4

train-loras: train-lora-birthday

# ── Evaluation ────────────────────────────────────────────────────────────────
system-eval:
	python -m eval.llm_system_eval

figures:
	python -m eval.reports.thesis_figures

# ── Web app ───────────────────────────────────────────────────────────────────
serve:
	uvicorn app.api:app --reload --port 8000

serve-docker:
	docker compose --profile app up --build app
