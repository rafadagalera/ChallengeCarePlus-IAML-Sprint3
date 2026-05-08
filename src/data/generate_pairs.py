"""
Generates individual × food pairs with a plausible heuristic nutritional score.

Score logic (0–10):
  Start at 10.0
  1. Hard allergen constraint  → 0 immediately
  2. Diet incompatibility      → penalty up to –4
  3. Goal alignment            → bonus/penalty up to ±2
  4. Health conditions         → penalty up to –3
  5. Explicit restrictions     → penalty up to –1.5 each
  + Gaussian noise σ=0.3
  → clip to [0, 10]
"""

import numpy as np
import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path(__file__).parents[2] / "data" / "processed"
RNG = np.random.default_rng(42)


# ──────────────────────────────────────────────
# Score components
# ──────────────────────────────────────────────

def _allergen_penalty(ind: pd.Series, food: pd.Series) -> float:
    allergens = ["gluten", "lactose", "nuts", "shellfish", "eggs", "soy"]
    for a in allergens:
        if ind.get(f"allergy_{a}", 0) and food.get(f"contains_{a}", 0):
            return -100.0  # hard block → will become 0 after clip
    return 0.0


def _diet_penalty(ind: pd.Series, food: pd.Series) -> float:
    diet = ind["diet_type"]
    if diet == "vegan" and food["is_animal_product"]:
        return -4.0
    if diet == "vegetarian" and (food["is_meat"] or food["is_fish"]):
        return -3.0
    if diet == "pescatarian" and food["is_meat"]:
        return -2.0
    if diet == "keto":
        carbs = food["carbohydrates_100g"]
        if carbs > 10:
            return -min(3.0, (carbs - 10) * 0.10)
    if diet == "paleo" and food["food_group"] in ("dairy", "grain", "legume", "processed"):
        return -2.0
    return 0.0


def _goal_bonus(ind: pd.Series, food: pd.Series) -> float:
    goal = ind["goal"]
    energy = food["energy_kcal_100g"]
    protein = food["proteins_100g"]
    fiber = food["fiber_100g"]
    sugar = food["sugar_100g"]
    bonus = 0.0

    if goal == "weight_loss":
        if energy > 400:
            bonus -= 1.5
        elif energy < 200:
            bonus += 0.5
        if fiber > 5:
            bonus += 0.5
        if protein > 15:
            bonus += 0.5
        if sugar > 15:
            bonus -= 0.5
    elif goal == "muscle_gain":
        if protein >= 20:
            bonus += 2.0
        elif protein >= 10:
            bonus += 0.8
        if energy > 250:
            bonus += 0.5
    elif goal == "health_improvement":
        if fiber > 5:
            bonus += 1.0
        if sugar < 5:
            bonus += 0.5
        if food["food_group"] in ("vegetable", "fruit", "legume"):
            bonus += 0.5
    elif goal == "energy_boost":
        if 200 <= energy <= 350:
            bonus += 1.0
        elif energy < 50:
            bonus -= 0.5
    # maintenance: neutral
    return bonus


def _health_penalty(ind: pd.Series, food: pd.Series) -> float:
    sodium = food["sodium_mg_100g"]
    sugar = food["sugar_100g"]
    sat_fat = food["saturated_fat_100g"]
    cholesterol = ind["total_cholesterol"]
    penalty = 0.0

    hyp = ind["hypertension"]
    if hyp == "uncontrolled":
        if sodium > 400:
            penalty += 3.0
        elif sodium > 200:
            penalty += 1.5
    elif hyp == "controlled":
        if sodium > 400:
            penalty += 2.0
        elif sodium > 200:
            penalty += 0.8

    glycemic = ind["glycemic_condition"]
    if glycemic in ("type_1", "type_2", "pre_diabetic"):
        if sugar > 15:
            penalty += 2.0
        elif sugar > 8:
            penalty += 1.0

    if cholesterol > 240:
        if sat_fat > 7:
            penalty += 1.5
        elif sat_fat > 4:
            penalty += 0.8

    return penalty


def _restriction_penalty(ind: pd.Series, food: pd.Series) -> float:
    penalty = 0.0
    if ind.get("restriction_low_sodium", 0) and food["sodium_mg_100g"] > 200:
        penalty += 1.0
    if ind.get("restriction_low_sugar", 0) and food["sugar_100g"] > 5:
        penalty += 1.0
    if ind.get("restriction_low_fat", 0) and food["fat_100g"] > 15:
        penalty += 1.0
    if ind.get("restriction_high_protein", 0) and food["proteins_100g"] < 10:
        penalty += 1.0
    if ind.get("restriction_low_carb", 0) and food["carbohydrates_100g"] > 20:
        penalty += 1.5
    return penalty


def compute_score(ind: pd.Series, food: pd.Series, noise_std: float = 0.3) -> float:
    allergen = _allergen_penalty(ind, food)
    if allergen < 0:
        return 0.0

    score = (
        10.0
        + _diet_penalty(ind, food)
        + _goal_bonus(ind, food)
        - _health_penalty(ind, food)
        - _restriction_penalty(ind, food)
        + RNG.normal(0, noise_std)
    )
    return float(np.clip(score, 0, 10))


# ──────────────────────────────────────────────
# Pair generation
# ──────────────────────────────────────────────

def generate_pairs(
    individuals_df: pd.DataFrame | None = None,
    foods_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if individuals_df is None:
        individuals_df = pd.read_csv(PROCESSED_DIR / "individuals.csv")
    if foods_df is None:
        foods_df = pd.read_csv(PROCESSED_DIR / "foods.csv")

    print(f"[generate_pairs] {len(individuals_df)} individuals × {len(foods_df)} foods "
          f"= {len(individuals_df) * len(foods_df):,} pairs")

    rows = []
    for _, ind in individuals_df.iterrows():
        for _, food in foods_df.iterrows():
            score = compute_score(ind, food)
            rows.append({
                "individual_id": ind["individual_id"],
                "food_id": food["food_id"],
                "score": round(score, 3),
            })

    pairs_df = pd.DataFrame(rows)
    out_path = PROCESSED_DIR / "pairs.csv"
    pairs_df.to_csv(out_path, index=False)
    print(f"[generate_pairs] saved {len(pairs_df):,} pairs to {out_path}")
    print(f"  score distribution: mean={pairs_df['score'].mean():.2f}, "
          f"std={pairs_df['score'].std():.2f}, "
          f"min={pairs_df['score'].min():.2f}, max={pairs_df['score'].max():.2f}")
    return pairs_df


if __name__ == "__main__":
    generate_pairs()
