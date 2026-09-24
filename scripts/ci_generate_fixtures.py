"""Minimal CI fixture generator for SIH26138 Green Fleet Navigator.

Runs ONLY the two stages required to make the full pytest suite pass:

  Stage 1 — Generate synthetic voyage dataset (data/raw/voyages_sample.csv)
  Stage 2a — Train the three baseline ML models and write:
              - artifacts/models/*.pkl
              - artifacts/metrics/baseline_metrics.json

Everything else (prediction benchmark, compliance spot-check, fleet
scheduler, QPSO/PSO scalability sweep, Pareto front, 6-fuel scenario
analysis) is intentionally OMITTED.  Those stages are integration-level
benchmarks — they belong in the full pipeline run (run_full_pipeline.py),
not in a per-push CI gate.

Typical runtime (--rows 500, GitHub Actions ubuntu-latest):  ~25-35 s
Typical runtime (run_full_pipeline.py --rows 500):           ~2 m 30 s+

Usage
-----
  # CI (called by .github/workflows/ci.yml)
  python scripts/ci_generate_fixtures.py --rows 500

  # Faster local smoke-test
  python scripts/ci_generate_fixtures.py --rows 200

  # Full reproducible benchmark (unchanged)
  python scripts/run_full_pipeline.py --rows 10000
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path regardless of CWD
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from logging_config import configure_logging  # noqa: E402  (post-path-patch)
from src.prediction import train_all_models   # noqa: E402

logger = logging.getLogger("maritime_system")


def _ensure_dirs() -> None:
    for d in (
        "data/raw",
        "data/processed",
        "artifacts/models",
        "artifacts/metrics",
        "outputs/reports",
        "outputs/logs",
    ):
        Path(d).mkdir(parents=True, exist_ok=True)


def generate_fixtures(
    *,
    dataset_path: str = "data/raw/voyages_sample.csv",
    rows: int = 500,
    seed: int = 42,
    force: bool = False,
) -> None:
    """Generate the minimal set of artifacts required by the pytest suite."""
    configure_logging(level="INFO")
    _ensure_dirs()
    t_start = time.perf_counter()

    # ── Stage 1: synthetic voyage dataset ────────────────────────────────────
    data_file = Path(dataset_path)
    if force or not data_file.exists():
        logger.info("[CI-FIXTURE] Generating %d synthetic voyage records → %s", rows, dataset_path)
        from scripts.make_mock_dataset import generate_synthetic_voyages, write_csv
        records = generate_synthetic_voyages(num_rows=rows, seed=seed)
        write_csv(records, data_file)
        logger.info("[CI-FIXTURE] Dataset written: %s (%d rows)", data_file, len(records))
    else:
        logger.info("[CI-FIXTURE] Dataset already exists (%d bytes) — skipping generation", data_file.stat().st_size)

    # ── Stage 2a: train baseline models only ─────────────────────────────────
    logger.info("[CI-FIXTURE] Training baseline ML models (hist_gradient_boosting, random_forest, linear_regression)…")
    train_all_models(dataset_path=dataset_path, artifacts_dir="artifacts", random_state=seed)
    logger.info("[CI-FIXTURE] Model training complete")

    elapsed = time.perf_counter() - t_start
    logger.info("[CI-FIXTURE] Fixture generation finished in %.2f s", elapsed)

    # Sanity-check: assert required artifacts exist before handing off to pytest
    required = {
        "data/raw/voyages_sample.csv": Path(dataset_path),
        "artifacts/metrics/baseline_metrics.json": Path("artifacts/metrics/baseline_metrics.json"),
    }
    missing = [label for label, p in required.items() if not p.exists()]
    if missing:
        logger.error("[CI-FIXTURE] FATAL — required artifacts not found: %s", missing)
        sys.exit(1)

    # Confirm at least one .pkl exists
    pkl_files = list(Path("artifacts/models").glob("*.pkl"))
    if not pkl_files:
        logger.error("[CI-FIXTURE] FATAL — no .pkl model files found in artifacts/models/")
        sys.exit(1)

    logger.info(
        "[CI-FIXTURE] All required artifacts verified (%d model(s): %s)",
        len(pkl_files),
        [p.name for p in pkl_files],
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate minimal CI test fixtures for SIH26138 (data + model training only)."
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=500,
        help="Synthetic voyage rows to generate if dataset is absent (default: 500)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/raw/voyages_sample.csv",
        help="Path to raw voyage CSV (default: data/raw/voyages_sample.csv)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate dataset even if it already exists",
    )
    args = parser.parse_args()
    generate_fixtures(
        dataset_path=args.data,
        rows=args.rows,
        seed=args.seed,
        force=args.force,
    )


if __name__ == "__main__":
    main()
