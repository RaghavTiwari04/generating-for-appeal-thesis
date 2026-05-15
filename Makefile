.PHONY: up down reset-db test lint fmt typecheck fonts embed-features \
        proxy-labels occasion-clf dedup snapshot scrape-etsy \
        train-predictor eval-predictor run-card

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

# ── Setup ──────────────────────────────────────────────────────────────────────
install:
	uv pip install -e ".[dev]"

discover-selectors:
	python scripts/discover_selectors.py

fonts:
	python -m generation.layout.download_fonts

weights:
	python -c "from generation.image.upscaler import download_realesrgan_weights; download_realesrgan_weights()"

# ── Tests ──────────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v --tb=short

test-fast:
	pytest tests/ -v --tb=short -x -q

coverage:
	pytest tests/ --cov=. --cov-report=html --cov-report=term-missing

# ── Code quality ───────────────────────────────────────────────────────────────
lint:
	ruff check .

fmt:
	ruff format .

typecheck:
	mypy data/ models/ generation/ pipeline/ eval/ common/ --ignore-missing-imports

# ── Data pipeline (run in order) ──────────────────────────────────────────────
scrape-etsy:
	python -m data.scrapers.run_scraper --source etsy --limit 1000

snapshot:
	python -m data.scrapers.snapshot_job --limit 5000

embed-features:
	python -m data.features.clip_embed
	python -m data.features.ocr
	python -m data.features.palette
	python -m data.features.image_complexity

occasion-clf-train:
	python -m data.features.occasion_classifier train --epochs 5

occasion-clf:
	python -m data.features.occasion_classifier infer --limit 10000

dedup:
	python -m data.features.dedup

proxy-labels:
	python -m data.labels.proxy

survey-labels:
	python -m data.labels.survey_labels --study-id main_v1

pilot-analysis:
	python -m survey.analysis.pilot_analysis --study-id pilot_v1

snapshot-cron:
	python -m data.scrapers.scheduler install

snapshot-cron-windows:
	python -m data.scrapers.scheduler windows

# ── Model training ─────────────────────────────────────────────────────────────
train-predictor:
	python -m models.predictor.train --epochs 30 --batch-size 64

eval-predictor:
	python -m eval.predictor_eval_standalone

train-pricing:
	python -m models.pricing.train_pricing

survey-labels:
	python -m data.labels.survey_labels --study-id main_v1

pilot-analysis:
	python -m survey.analysis.pilot_analysis --study-id pilot_v1

ablation-no-lora:
	python -m eval.ablations.no_lora

ablation-no-layout:
	python -m eval.ablations.no_layout

ablation-no-distinctiveness:
	python -m eval.ablations.no_distinctiveness

ablations: ablation-no-lora ablation-no-layout ablation-no-distinctiveness best-of-n-curve

# ── Generation ────────────────────────────────────────────────────────────────
run-card:
	python -m pipeline.orchestrator \
		--occasion birthday/general \
		--tone warm-humorous \
		--n 8 --top-k 3

# ── LoRA training (GPU required — rent A100) ──────────────────────────────────
train-lora-%:
	python -m generation.image.loras.train_lora --occasion $* --rank 8 --steps 1000

# Shorthand for top-5 occasions
train-loras: \
	train-lora-birthday/general \
	train-lora-christmas/general \
	train-lora-mothers_day \
	train-lora-valentines_day \
	train-lora-sympathy/bereavement

# ── Web app ───────────────────────────────────────────────────────────────────
serve:
	uvicorn app.api:app --reload --port 8000

serve-docker:
	docker compose --profile app up --build app

# ── Data pipeline (single-command full run) ───────────────────────────────────
pipeline:
	python -m data.pipeline_runner

pipeline-from-%:
	python -m data.pipeline_runner --from $*

# ── Survey instrument ─────────────────────────────────────────────────────────
survey-static:
	python -m survey.instrument.create_static

survey-app: survey-static
	uvicorn survey.instrument.app:app --reload --port 8080

# ── Figures ───────────────────────────────────────────────────────────────────
figures:
	python -m eval.reports.figures

# ── Image download ────────────────────────────────────────────────────────────
download-images:
	python -m data.scrapers.image_downloader --limit 10000

# ── W&B sweep ─────────────────────────────────────────────────────────────────
sweep:
	wandb sweep models/predictor/sweep.yaml

# ── Evaluation ────────────────────────────────────────────────────────────────
system-eval:
	python -m eval.system_eval --study-id system_eval_v1

failure-analysis:
	python -m eval.failure_analysis --study-id system_eval_v1

best-of-n-curve:
	python -m eval.ablations.best_of_n_curve
