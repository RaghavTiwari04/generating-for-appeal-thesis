"""Ordered data-pipeline runner.

Runs all Phase 1 steps in dependency order with progress reporting.
Safe to re-run: every step is idempotent (ON CONFLICT DO NOTHING / DO UPDATE).

Steps:
  1. scrape      — discover + fetch listings from all sources
  2. download    — fetch images for new listings → MinIO
  3. embed       — CLIP embeddings
  4. ocr         — headline OCR
  5. palette     — LAB palette
  6. complexity  — image complexity score
  8. clf-infer   — run occasion classifier on unlabelled listings
  9. dedup       — cluster duplicates

Usage:
    python -m data.pipeline_runner              # all steps
    python -m data.pipeline_runner --from embed # skip scrape + download
    python -m data.pipeline_runner --only dedup # just one step
    python -m data.pipeline_runner --dry-run    # print what would run
"""

from __future__ import annotations

import time
from collections.abc import Callable

import typer
from rich.console import Console
from rich.table import Table

from common.logging import get_logger

log = get_logger(__name__)
console = Console()
app = typer.Typer()

STEPS: list[tuple[str, str, Callable]] = []  # populated below


def _step(name: str, description: str):
    def decorator(fn: Callable) -> Callable:
        STEPS.append((name, description, fn))
        return fn
    return decorator


# ---------------------------------------------------------------------------
# Step definitions
# ---------------------------------------------------------------------------

@_step("scrape", "Scrape Redbubble, Greetings Island")
def step_scrape(limit: int) -> int:
    import asyncio

    from common.occasions import ACTIVE_OCCASIONS
    from data.scrapers.run_scraper import _run

    def _occasion_query(o: str) -> str:
        return o.replace("/", " ").replace("_", " ") + " greeting card"

    queries = [_occasion_query(o) for o in ACTIVE_OCCASIONS]
    sources = ["redbubble", "greetings_island"]
    total = 0
    for src in sources:
        try:
            n = asyncio.run(_run(src, queries[:5], limit // len(sources) // 5, use_cache=True))
            total += n
        except Exception as e:
            log.warning(f"Scraper {src} failed: {e}")
    return total


@_step("download", "Download listing images → MinIO")
def step_download(limit: int) -> int:
    import asyncio

    from data.scrapers.image_downloader import download_batch
    return asyncio.run(download_batch(limit))


@_step("embed", "CLIP embeddings for images")
def step_embed(limit: int) -> int:
    from data.features.clip_embed import run_embed_missing
    return run_embed_missing(limit)


@_step("ocr", "OCR headline text extraction")
def step_ocr(limit: int) -> int:
    from data.features.ocr import run_ocr_missing
    return run_ocr_missing(limit)


@_step("palette", "LAB colour palette extraction")
def step_palette(limit: int) -> int:
    from data.features.palette import run_palette_missing
    return run_palette_missing(limit)


@_step("complexity", "Image complexity score")
def step_complexity(limit: int) -> int:
    from data.features.image_complexity import run_complexity_missing
    return run_complexity_missing(limit)


@_step("clf-infer", "Occasion classification (keyword rules)")
def step_clf_infer(limit: int) -> int:
    import subprocess
    import sys
    result = subprocess.run(
        [sys.executable, "-m", "data.features.occasion_classifier", "infer", f"--limit={limit}"],
        check=False,
    )
    return 0 if result.returncode == 0 else -1


@_step("dedup", "Deduplicate listings (pHash + CLIP + TF-IDF)")
def step_dedup(limit: int) -> int:
    from data.features.dedup import run_dedup
    stats = run_dedup(limit if limit else None)
    return stats.duplicates


@_step("score-listings", "Score listings with trained predictor (per-head scores)")
def step_score_listings(limit: int) -> int:
    from data.features.predictor_scores import run
    return run(limit=limit)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

STEP_NAMES = [s[0] for s in STEPS]


@app.command()
def run_pipeline(
    from_step: str = typer.Option("scrape", "--from", help="Start from this step"),
    only: str = typer.Option("", "--only", help="Run only this step"),
    limit: int = typer.Option(5000, "--limit", help="Max items per step"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    if only:
        steps_to_run = [(n, d, fn) for n, d, fn in STEPS if n == only]
        if not steps_to_run:
            console.print(f"[red]Unknown step: {only!r}. Choose from: {STEP_NAMES}")
            raise typer.Exit(1)
    else:
        start_idx = next((i for i, (n, _, __) in enumerate(STEPS) if n == from_step), 0)
        steps_to_run = STEPS[start_idx:]

    table = Table(title="Pipeline steps to run", show_lines=True)
    table.add_column("Step")
    table.add_column("Description")
    for n, d, _ in steps_to_run:
        table.add_row(n, d)
    console.print(table)

    if dry_run:
        console.print("[yellow]Dry run — no steps executed.")
        return

    results: list[tuple[str, float, int | str]] = []
    for name, desc, fn in steps_to_run:
        console.print(f"\n[bold cyan]▶ {name}[/] — {desc}")
        t0 = time.monotonic()
        try:
            count = fn(limit)
            elapsed = time.monotonic() - t0
            results.append((name, elapsed, count))
            console.print(f"  [green]✓[/] {count} items  ({elapsed:.1f}s)")
        except Exception as e:
            elapsed = time.monotonic() - t0
            results.append((name, elapsed, f"ERROR: {e}"))
            console.print(f"  [red]✗ {e}[/]")
            log.exception(f"Step {name} failed")

    summary = Table(title="Pipeline summary", show_lines=True)
    summary.add_column("Step")
    summary.add_column("Result")
    summary.add_column("Time")
    for name, t, r in results:
        summary.add_row(name, str(r), f"{t:.1f}s")
    console.print(summary)


if __name__ == "__main__":
    app()
