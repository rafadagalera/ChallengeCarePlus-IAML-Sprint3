"""
Generates evaluation plots and a text report from saved training artefacts.
"""

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

MODELS_DIR    = Path(__file__).parents[2] / "data" / "models"
PROCESSED_DIR = Path(__file__).parents[2] / "data" / "processed"
REPORTS_DIR   = Path(__file__).parents[2] / "docs"


def evaluate() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    history  = joblib.load(MODELS_DIR / "training_history.pkl")
    results  = joblib.load(MODELS_DIR / "test_results.pkl")
    y_test   = results["y_test"]
    y_pred   = results["y_pred"]
    metrics  = results["metrics"]

    print("\n══════════════════════════════════════")
    print("         EVALUATION REPORT")
    print("══════════════════════════════════════")
    print(f"  MAE  : {metrics['mae']}")
    print(f"  RMSE : {metrics['rmse']}")
    print(f"  MSE  : {metrics['mse']}")
    print(f"  R²   : {metrics['r2']}")
    print("══════════════════════════════════════\n")

    # ── 1. Training loss curve ──────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].plot(history["train_loss"], label="Train MSE", linewidth=1.5)
    axes[0].plot(history["val_loss"],   label="Val MSE",   linewidth=1.5)
    axes[0].set_title("Training Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # ── 2. Predicted vs actual scatter ─────────
    axes[1].scatter(y_test, y_pred, alpha=0.3, s=6)
    lo, hi = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    axes[1].plot([lo, hi], [lo, hi], "r--", linewidth=1)
    axes[1].set_title(f"Predicted vs Actual  (R²={metrics['r2']})")
    axes[1].set_xlabel("Actual score")
    axes[1].set_ylabel("Predicted score")
    axes[1].grid(True, alpha=0.3)

    # ── 3. Residuals distribution ───────────────
    residuals = y_pred - y_test
    axes[2].hist(residuals, bins=60, color="steelblue", edgecolor="white", linewidth=0.3)
    axes[2].axvline(0, color="red", linestyle="--")
    axes[2].set_title(f"Residuals  (MAE={metrics['mae']})")
    axes[2].set_xlabel("Prediction error")
    axes[2].set_ylabel("Count")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = REPORTS_DIR / "evaluation_plots.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"[evaluate] plots saved to {plot_path}")

    # ── 4. Score distribution by food group ────
    pairs_df = pd.read_csv(PROCESSED_DIR / "pairs.csv")
    foods_df = pd.read_csv(PROCESSED_DIR / "foods.csv")
    merged   = pairs_df.merge(foods_df[["food_id", "food_group"]], on="food_id", how="left")

    fig, ax = plt.subplots(figsize=(10, 4))
    order = merged.groupby("food_group")["score"].median().sort_values(ascending=False).index
    sns.boxplot(data=merged, x="food_group", y="score", order=order,
                hue="food_group", palette="Set2", legend=False, ax=ax)
    ax.set_title("Score distribution by food group")
    ax.set_xlabel("")
    ax.set_ylabel("Nutritional score (0–10)")
    plt.xticks(rotation=30)
    plt.tight_layout()
    food_plot_path = REPORTS_DIR / "score_by_food_group.png"
    plt.savefig(food_plot_path, dpi=150)
    plt.close()
    print(f"[evaluate] food-group plot saved to {food_plot_path}")

    # ── 5. Score distribution by individual goal ─
    ind_df = pd.read_csv(PROCESSED_DIR / "individuals.csv")
    merged2 = pairs_df.merge(ind_df[["individual_id", "goal"]], on="individual_id", how="left")

    fig, ax = plt.subplots(figsize=(10, 4))
    order2 = merged2.groupby("goal")["score"].median().sort_values(ascending=False).index
    sns.boxplot(data=merged2, x="goal", y="score", order=order2,
                hue="goal", palette="Set3", legend=False, ax=ax)
    ax.set_title("Score distribution by individual goal")
    ax.set_xlabel("")
    ax.set_ylabel("Nutritional score (0–10)")
    plt.xticks(rotation=20)
    plt.tight_layout()
    goal_plot_path = REPORTS_DIR / "score_by_goal.png"
    plt.savefig(goal_plot_path, dpi=150)
    plt.close()
    print(f"[evaluate] goal plot saved to {goal_plot_path}")


if __name__ == "__main__":
    evaluate()
