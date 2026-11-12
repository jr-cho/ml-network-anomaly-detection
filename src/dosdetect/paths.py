"""Repo paths, resolved from this file so they work from any cwd."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
RF_RESULTS_DIR = RESULTS_DIR / "random_forest"
