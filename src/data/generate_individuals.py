"""
Generates 1 000 synthetic but plausible individuals and saves individuals.csv.
"""

import numpy as np
import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path(__file__).parents[2] / "data" / "processed"

RNG = np.random.default_rng(42)

DIET_TYPES = ["omnivore", "vegetarian", "vegan", "keto", "pescatarian", "paleo"]
DIET_WEIGHTS = [0.50, 0.20, 0.10, 0.10, 0.08, 0.02]

GOALS = ["weight_loss", "muscle_gain", "maintenance", "health_improvement", "energy_boost"]
GOAL_WEIGHTS = [0.35, 0.20, 0.25, 0.15, 0.05]

ACTIVITY_LEVELS = ["sedentary", "lightly_active", "moderately_active", "very_active"]
ACTIVITY_WEIGHTS = [0.20, 0.35, 0.30, 0.15]

GLYCEMIC_CONDITIONS = ["none", "pre_diabetic", "type_2", "type_1"]
GLYCEMIC_WEIGHTS = [0.84, 0.10, 0.05, 0.01]

HYPERTENSION_STATES = ["none", "controlled", "uncontrolled"]
HYPERTENSION_WEIGHTS = [0.70, 0.20, 0.10]

ALLERGENS = ["gluten", "lactose", "nuts", "shellfish", "eggs", "soy"]
ALLERGEN_PROBS = [0.02, 0.15, 0.03, 0.04, 0.01, 0.02]

RESTRICTIONS = ["low_sodium", "low_sugar", "low_fat", "high_protein", "low_carb"]


def _weighted_choice(options: list, weights: list, size: int) -> np.ndarray:
    probs = np.array(weights) / np.sum(weights)
    return RNG.choice(options, size=size, p=probs)


def generate_individuals(n: int = 1000) -> pd.DataFrame:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    age = RNG.normal(40, 15, n).clip(18, 80).astype(int)
    height_cm = RNG.normal(170, 10, n).clip(150, 200).round(1)
    weight_kg = RNG.normal(75, 15, n).clip(45, 150).round(1)
    bmi = (weight_kg / (height_cm / 100) ** 2).round(1)

    # Cholesterol loosely correlated with age and BMI
    cholesterol_base = 160 + age * 0.7 + (bmi - 22) * 1.5
    total_cholesterol = (cholesterol_base + RNG.normal(0, 20, n)).clip(120, 320).round(0).astype(int)

    diet_type = _weighted_choice(DIET_TYPES, DIET_WEIGHTS, n)
    goal = _weighted_choice(GOALS, GOAL_WEIGHTS, n)
    activity_level = _weighted_choice(ACTIVITY_LEVELS, ACTIVITY_WEIGHTS, n)

    # Glycemic condition slightly more likely when high BMI
    glycemic_base_probs = np.tile(GLYCEMIC_WEIGHTS, (n, 1)).astype(float)
    high_bmi_mask = bmi > 30
    glycemic_base_probs[high_bmi_mask, 0] -= 0.15   # less 'none'
    glycemic_base_probs[high_bmi_mask, 1] += 0.10   # more pre_diabetic
    glycemic_base_probs[high_bmi_mask, 2] += 0.05   # more type_2
    glycemic_base_probs = np.clip(glycemic_base_probs, 0, 1)
    glycemic_base_probs /= glycemic_base_probs.sum(axis=1, keepdims=True)
    glycemic_condition = np.array([
        RNG.choice(GLYCEMIC_CONDITIONS, p=glycemic_base_probs[i]) for i in range(n)
    ])

    # Hypertension slightly more likely with high cholesterol and age
    hypertension_base_probs = np.tile(HYPERTENSION_WEIGHTS, (n, 1)).astype(float)
    high_risk_mask = (total_cholesterol > 240) | (age > 55)
    hypertension_base_probs[high_risk_mask, 0] -= 0.15
    hypertension_base_probs[high_risk_mask, 1] += 0.10
    hypertension_base_probs[high_risk_mask, 2] += 0.05
    hypertension_base_probs = np.clip(hypertension_base_probs, 0, 1)
    hypertension_base_probs /= hypertension_base_probs.sum(axis=1, keepdims=True)
    hypertension = np.array([
        RNG.choice(HYPERTENSION_STATES, p=hypertension_base_probs[i]) for i in range(n)
    ])

    # Allergens: independent Bernoulli per allergen
    allergen_matrix = RNG.random((n, len(ALLERGENS))) < ALLERGEN_PROBS

    # Restrictions: correlated with health conditions
    has_hypertension = hypertension != "none"
    has_glycemic = glycemic_condition != "none"
    high_cholesterol = total_cholesterol > 200
    muscle_goal = goal == "muscle_gain"
    keto_diet = diet_type == "keto"

    restriction_matrix = np.zeros((n, len(RESTRICTIONS)), dtype=int)
    restriction_matrix[:, 0] = (  # low_sodium
        (has_hypertension & (RNG.random(n) < 0.70)) |
        (~has_hypertension & (RNG.random(n) < 0.05))
    ).astype(int)
    restriction_matrix[:, 1] = (  # low_sugar
        (has_glycemic & (RNG.random(n) < 0.80)) |
        (~has_glycemic & (RNG.random(n) < 0.08))
    ).astype(int)
    restriction_matrix[:, 2] = (  # low_fat
        (high_cholesterol & (RNG.random(n) < 0.40)) |
        (~high_cholesterol & (RNG.random(n) < 0.08))
    ).astype(int)
    restriction_matrix[:, 3] = (  # high_protein
        (muscle_goal & (RNG.random(n) < 0.75)) |
        (~muscle_goal & (RNG.random(n) < 0.10))
    ).astype(int)
    restriction_matrix[:, 4] = (  # low_carb
        (keto_diet & (RNG.random(n) < 0.90)) |
        (~keto_diet & (RNG.random(n) < 0.05))
    ).astype(int)

    # Assemble DataFrame
    df = pd.DataFrame({
        "individual_id": [f"ind_{i:04d}" for i in range(n)],
        "age": age,
        "diet_type": diet_type,
        "total_cholesterol": total_cholesterol,
        "weight_kg": weight_kg,
        "height_cm": height_cm,
        "bmi": bmi,
        "goal": goal,
        "activity_level": activity_level,
        "glycemic_condition": glycemic_condition,
        "hypertension": hypertension,
    })

    for i, allergen in enumerate(ALLERGENS):
        df[f"allergy_{allergen}"] = allergen_matrix[:, i].astype(int)

    for i, restriction in enumerate(RESTRICTIONS):
        df[f"restriction_{restriction}"] = restriction_matrix[:, i]

    out_path = PROCESSED_DIR / "individuals.csv"
    df.to_csv(out_path, index=False)
    print(f"[generate_individuals] saved {len(df)} individuals to {out_path}")
    return df


if __name__ == "__main__":
    generate_individuals()
