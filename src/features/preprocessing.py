"""
Builds and exposes the sklearn ColumnTransformer that converts raw
individual + food feature rows into a numeric matrix for the MLP.
"""

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

# ──────────────────────────────────────────────
# Column groups — individual side
# ──────────────────────────────────────────────
IND_NUM = ["age", "total_cholesterol", "weight_kg", "height_cm", "bmi"]

IND_CAT = ["diet_type", "goal", "activity_level", "glycemic_condition", "hypertension"]

IND_BIN = [
    "allergy_gluten", "allergy_lactose", "allergy_nuts",
    "allergy_shellfish", "allergy_eggs", "allergy_soy",
    "restriction_low_sodium", "restriction_low_sugar", "restriction_low_fat",
    "restriction_high_protein", "restriction_low_carb",
]

# ──────────────────────────────────────────────
# Column groups — food side
# ──────────────────────────────────────────────
FOOD_NUM = [
    "energy_kcal_100g", "proteins_100g", "carbohydrates_100g",
    "fat_100g", "saturated_fat_100g", "fiber_100g",
    "sodium_mg_100g", "sugar_100g",
]

FOOD_CAT = ["food_group"]

FOOD_BIN = [
    "contains_gluten", "contains_lactose", "contains_nuts",
    "contains_shellfish", "contains_eggs", "contains_soy",
    "is_animal_product", "is_meat", "is_fish",
]

ALL_FEATURE_COLS = IND_NUM + IND_CAT + IND_BIN + FOOD_NUM + FOOD_CAT + FOOD_BIN


def build_preprocessor() -> ColumnTransformer:
    num_pipe = Pipeline([("scaler", StandardScaler())])
    cat_pipe = Pipeline([
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    return ColumnTransformer(
        transformers=[
            ("ind_num",  num_pipe, IND_NUM),
            ("food_num", num_pipe, FOOD_NUM),
            ("ind_cat",  cat_pipe, IND_CAT),
            ("food_cat", cat_pipe, FOOD_CAT),
            ("bin",      "passthrough", IND_BIN + FOOD_BIN),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_X(
    pairs_df: pd.DataFrame,
    individuals_df: pd.DataFrame,
    foods_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Merges pairs with individual and food feature tables.

    Returns:
        merged_df   : merged DataFrame with all raw columns
        X_cols      : list of column names used as features
        (so caller can call preprocessor.fit/transform on merged_df[X_cols])
    """
    merged = (
        pairs_df
        .merge(individuals_df, on="individual_id", how="left")
        .merge(foods_df, on="food_id", how="left")
    )
    return merged
