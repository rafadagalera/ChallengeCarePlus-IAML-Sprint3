"""
Trains the MLPRegressor on the generated pairs and saves model artefacts.
"""

import warnings
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import ConvergenceWarning

from src.features.preprocessing import build_preprocessor, build_X, ALL_FEATURE_COLS

PROCESSED_DIR = Path(__file__).parents[2] / "data" / "processed"
MODELS_DIR    = Path(__file__).parents[2] / "data" / "models"


def train(
    pairs_path: Path | None = None,
    individuals_path: Path | None = None,
    foods_path: Path | None = None,
) -> dict:
    pairs_path       = pairs_path       or PROCESSED_DIR / "pairs.csv"
    individuals_path = individuals_path or PROCESSED_DIR / "individuals.csv"
    foods_path       = foods_path       or PROCESSED_DIR / "foods.csv"

    print("[train] Loading data …")
    pairs_df = pd.read_csv(pairs_path)
    ind_df   = pd.read_csv(individuals_path)
    foods_df = pd.read_csv(foods_path)

    merged = build_X(pairs_df, ind_df, foods_df)

    # Individual-based split: hold out 10 % of individuals for val, 10 % for test
    # This ensures the model is evaluated on completely unseen individuals.
    all_ind_ids = list(merged["individual_id"].unique())
    rng = np.random.default_rng(42)
    rng.shuffle(all_ind_ids)

    n = len(all_ind_ids)
    n_test = max(1, int(n * 0.10))
    n_val  = max(1, int(n * 0.10))

    test_ids = set(all_ind_ids[:n_test])
    val_ids  = set(all_ind_ids[n_test:n_test + n_val])
    train_ids = set(all_ind_ids[n_test + n_val:])

    mask_train = merged["individual_id"].isin(train_ids)
    mask_val   = merged["individual_id"].isin(val_ids)
    mask_test  = merged["individual_id"].isin(test_ids)

    train_df = merged[mask_train]
    val_df   = merged[mask_val]
    test_df  = merged[mask_test]

    print(f"  train: {len(train_df):,}  val: {len(val_df):,}  test: {len(test_df):,}")

    feature_cols = [c for c in ALL_FEATURE_COLS if c in merged.columns]
    X_train = train_df[feature_cols]
    y_train = train_df["score"].values
    X_val   = val_df[feature_cols]
    y_val   = val_df["score"].values
    X_test  = test_df[feature_cols]
    y_test  = test_df["score"].values

    print("[train] Fitting preprocessor …")
    preprocessor = build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)
    X_val_t   = preprocessor.transform(X_val)
    X_test_t  = preprocessor.transform(X_test)

    print(f"  feature matrix shape: {X_train_t.shape}")

    print("[train] Training MLP …")
    model = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        alpha=0.01,
        batch_size=256,
        learning_rate="adaptive",
        learning_rate_init=1e-3,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.0,   # we manage val split ourselves
        n_iter_no_change=15,
        tol=1e-4,
        random_state=42,
        verbose=False,
    )

    # Feed validation set via warm_start loop so we can track val loss
    best_val_loss = np.inf
    best_model_state = None
    patience = 15
    no_improve = 0
    history = {"train_loss": [], "val_loss": []}

    from sklearn.metrics import mean_squared_error
    model.set_params(max_iter=1, warm_start=True, early_stopping=False)

    for epoch in range(500):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(X_train_t, y_train)
        train_loss = mean_squared_error(y_train, model.predict(X_train_t))
        val_loss   = mean_squared_error(y_val,   model.predict(X_val_t))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_model_state = {
                "coefs_": [c.copy() for c in model.coefs_],
                "intercepts_": [b.copy() for b in model.intercepts_],
            }
            no_improve = 0
        else:
            no_improve += 1

        if (epoch + 1) % 50 == 0:
            print(f"  epoch {epoch+1:3d}  train_mse={train_loss:.4f}  val_mse={val_loss:.4f}")

        if no_improve >= patience:
            print(f"  early stop at epoch {epoch+1}")
            break

    # Restore best weights
    if best_model_state:
        model.coefs_       = best_model_state["coefs_"]
        model.intercepts_  = best_model_state["intercepts_"]

    # Final evaluation
    from sklearn.metrics import mean_absolute_error, r2_score
    y_pred = model.predict(X_test_t).clip(0, 10)
    metrics = {
        "mae":  round(float(mean_absolute_error(y_test, y_pred)), 4),
        "mse":  round(float(mean_squared_error(y_test, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
        "r2":   round(float(r2_score(y_test, y_pred)), 4),
    }
    print(f"\n[train] Test metrics → {metrics}")

    # Persist artefacts
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model,        MODELS_DIR / "mlp_model.pkl")
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.pkl")
    joblib.dump(history,      MODELS_DIR / "training_history.pkl")
    joblib.dump(
        {"y_test": y_test, "y_pred": y_pred, "metrics": metrics},
        MODELS_DIR / "test_results.pkl",
    )
    print(f"[train] Artefacts saved to {MODELS_DIR}/")
    return metrics


if __name__ == "__main__":
    train()
