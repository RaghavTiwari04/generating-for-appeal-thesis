"""Snapshot job scheduler.

Sets up a weekly cron job for `data.scrapers.snapshot_job` using the
system crontab (Linux/Mac) or Windows Task Scheduler.

The snapshot job must run **every week for ≥ 12 consecutive weeks** starting
in month 1 to build enough history for meaningful velocity features.

Usage:
    # Add cron entry (Linux/Mac — requires crontab access)
    python -m data.scrapers.scheduler install

    # Print the cron line without installing
    python -m data.scrapers.scheduler print-cron

    # Windows: emit PowerShell command to create a scheduled task
    python -m data.scrapers.scheduler windows

    # Remove the cron entry
    python -m data.scrapers.scheduler remove
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import typer

app = typer.Typer()

PYTHON = sys.executable
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
LOG_FILE = REPO_ROOT / "logs" / "snapshot.log"
MARKER = "# greeting-cards snapshot-job"

# Run at 03:00 every Sunday
CRON_SCHEDULE = "0 3 * * 0"
CRON_CMD = (
    f"{PYTHON} -m data.scrapers.snapshot_job --limit 5000 "
    f">> {LOG_FILE} 2>&1"
)
CRON_LINE = f"{CRON_SCHEDULE} cd {REPO_ROOT} && {CRON_CMD}  {MARKER}"


@app.command()
def print_cron() -> None:
    """Print the cron line to add manually."""
    typer.echo(CRON_LINE)


@app.command()
def install() -> None:
    """Install cron entry (Linux / macOS)."""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
    existing = result.stdout if result.returncode == 0 else ""

    if MARKER in existing:
        typer.echo("Cron entry already installed.")
        return

    new_crontab = existing.rstrip("\n") + "\n" + CRON_LINE + "\n"
    proc = subprocess.run(["crontab", "-"], input=new_crontab, text=True, capture_output=True)
    if proc.returncode != 0:
        typer.echo(f"crontab install failed: {proc.stderr}", err=True)
        raise typer.Exit(1)
    typer.echo(f"Cron installed: {CRON_LINE}")
    typer.echo(f"Log file: {LOG_FILE}")


@app.command()
def remove() -> None:
    """Remove the cron entry."""
    result = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
    if result.returncode != 0 or MARKER not in result.stdout:
        typer.echo("No cron entry found.")
        return

    lines = [l for l in result.stdout.splitlines() if MARKER not in l]
    new_crontab = "\n".join(lines) + "\n"
    subprocess.run(["crontab", "-"], input=new_crontab, text=True, check=True)
    typer.echo("Cron entry removed.")


@app.command()
def windows() -> None:
    """Print PowerShell command to schedule on Windows Task Scheduler."""
    ps = (
        f'$action = New-ScheduledTaskAction -Execute "{PYTHON}" '
        f'-Argument "-m data.scrapers.snapshot_job --limit 5000" '
        f'-WorkingDirectory "{REPO_ROOT}"\n'
        f'$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Sunday -At 3am\n'
        f'Register-ScheduledTask -TaskName "gc-snapshot" -Action $action '
        f'-Trigger $trigger -RunLevel Highest'
    )
    typer.echo("Run this in PowerShell (Administrator):\n")
    typer.echo(ps)


if __name__ == "__main__":
    app()
