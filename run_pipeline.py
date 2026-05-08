"""
Orchestrates the full data + training pipeline.

Usage:
    python run_pipeline.py [--skip-fetch]
"""

import argparse
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Nutritional Score Engine – pipeline")
    parser.add_argument(
        "--skip-fetch", action="store_true",
        help="Skip Open Food Facts fetch (use existing data/processed/foods.csv)",
    )
    parser.add_argument(
        "--mock-foods", action="store_true",
        help="Use built-in mock food dataset instead of Open Food Facts API",
    )
    args = parser.parse_args()

    t0 = time.time()

    # Step 1 – Food data
    if args.skip_fetch and Path("data/processed/foods.csv").exists():
        print("[pipeline] ── Step 1/5: SKIP (foods.csv already exists)")
    elif args.mock_foods:
        print("[pipeline] ── Step 1/5: Using built-in mock food dataset …")
        from src.data.mock_foods import build_mock_foods
        build_mock_foods()
    else:
        print("[pipeline] ── Step 1/5: Fetching food data from Open Food Facts …")
        from src.data.fetch_foods import fetch_foods
        foods_df = fetch_foods(use_mock_on_failure=True)
        if len(foods_df) < 10:
            print("[pipeline] ERROR: too few foods fetched. Check network access.")
            sys.exit(1)

    # Step 2 – Individuals
    print("\n[pipeline] ── Step 2/5: Generating synthetic individuals …")
    from src.data.generate_individuals import generate_individuals
    individuals_df = generate_individuals()

    # Step 3 – Pairs + heuristic scores
    print("\n[pipeline] ── Step 3/5: Generating individual × food pairs …")
    from src.data.generate_pairs import generate_pairs
    import pandas as pd
    pairs_df = generate_pairs(
        individuals_df=individuals_df,
        foods_df=pd.read_csv("data/processed/foods.csv"),
    )

    # Step 4 – Train MLP
    print("\n[pipeline] ── Step 4/5: Training MLP …")
    from src.model.train import train
    metrics = train()

    # Step 5 – Evaluate
    print("\n[pipeline] ── Step 5/5: Generating evaluation report …")
    from src.model.evaluate import evaluate
    evaluate()

    elapsed = time.time() - t0
    print(f"\n✓ Pipeline complete in {elapsed:.1f}s")
    print(f"  MAE={metrics['mae']}  RMSE={metrics['rmse']}  R²={metrics['r2']}")
    print("\nTo start the API:")
    print("  uvicorn src.api.main:app --reload")
    print("  Docs: http://127.0.0.1:8000/docs")


if __name__ == "__main__":
    main()
