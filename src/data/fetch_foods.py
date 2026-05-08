"""
Fetches food data from the Open Food Facts public API and saves a clean
foods.csv to data/processed/.
"""

import json
import time
import requests
import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = Path(__file__).parents[2] / "data" / "raw"
PROCESSED_DIR = Path(__file__).parents[2] / "data" / "processed"

OFF_SEARCH = "https://world.openfoodfacts.org/cgi/search.pl"
FIELDS = "code,product_name,nutriments,allergens_tags,food_groups_tags"
PAGE_SIZE = 8
SLEEP = 0.6          # seconds between requests
MAX_RETRIES = 3      # retry on 5xx with exponential backoff
HEADERS = {"User-Agent": "NutritionalScoreEngine/1.0 (educational; github.com/bcr)"}

# Search terms grouped by food_group label
SEARCH_CATEGORIES: dict[str, list[str]] = {
    "grain":     ["brown rice", "white pasta", "whole wheat bread", "rolled oats",
                  "quinoa", "corn tortilla", "barley flour", "buckwheat"],
    "legume":    ["cooked lentils", "chickpeas canned", "black beans", "kidney beans",
                  "edamame", "green peas", "split peas"],
    "vegetable": ["broccoli", "spinach", "carrots", "tomatoes", "bell pepper",
                  "kale", "sweet potato", "cucumber", "zucchini"],
    "fruit":     ["apple", "banana", "orange", "strawberry", "mango",
                  "grape", "watermelon", "blueberry", "avocado"],
    "dairy":     ["whole milk", "greek yogurt", "cheddar cheese",
                  "cream cheese", "butter", "skimmed milk"],
    "meat":      ["chicken breast", "ground beef", "pork chop",
                  "turkey breast", "lamb chop", "beef steak"],
    "fish":      ["salmon fillet", "canned tuna", "sardines in oil",
                  "cod fillet", "shrimp", "tilapia fillet"],
    "egg":       ["whole egg", "egg white liquid"],
    "nut":       ["almonds", "walnuts", "cashews", "peanuts",
                  "pistachios", "brazil nuts", "sunflower seeds"],
    "processed": ["pizza margherita", "potato chips", "chocolate bar",
                  "instant noodles", "hot dog sausage", "french fries frozen"],
    "beverage":  ["orange juice", "cola drink", "green tea unsweetened",
                  "beer lager", "coconut water", "whole milk"],
    "snack":     ["granola bar", "protein bar chocolate", "whole grain crackers",
                  "microwave popcorn", "rice cakes"],
}

ANIMAL_GROUPS = {"dairy", "meat", "fish", "egg"}

# Allergen tag prefixes in OFF
ALLERGEN_MAP = {
    "contains_gluten":    ["en:gluten", "en:wheat", "en:rye", "en:barley", "en:oats"],
    "contains_lactose":   ["en:milk", "en:lactose"],
    "contains_nuts":      ["en:nuts", "en:almonds", "en:walnuts", "en:cashews",
                           "en:pistachios", "en:hazelnuts", "en:pecans"],
    "contains_shellfish": ["en:crustaceans", "en:molluscs", "en:shellfish",
                           "en:shrimps", "en:lobster", "en:crab", "en:oysters"],
    "contains_eggs":      ["en:eggs", "en:egg"],
    "contains_soy":       ["en:soybeans", "en:soy", "en:soya"],
}

NUTRIMENT_KEYS = {
    "energy_kcal_100g":     "energy-kcal_100g",
    "proteins_100g":        "proteins_100g",
    "carbohydrates_100g":   "carbohydrates_100g",
    "fat_100g":             "fat_100g",
    "saturated_fat_100g":   "saturated-fat_100g",
    "fiber_100g":           "fiber_100g",
    "sodium_100g":          "sodium_100g",
    "sugar_100g":           "sugars_100g",
}


def _search(term: str) -> list[dict]:
    params = {
        "action": "process",
        "json": 1,
        "search_simple": 1,
        "search_terms": term,
        "page_size": PAGE_SIZE,
        "fields": FIELDS,
        "sort_by": "unique_scans_n",
    }
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(OFF_SEARCH, params=params, headers=HEADERS, timeout=15)
            r.raise_for_status()
            return r.json().get("products", [])
        except requests.exceptions.HTTPError as exc:
            if r.status_code in (429, 503) and attempt < MAX_RETRIES - 1:
                backoff = SLEEP * (2 ** attempt)
                print(f"  [retry {attempt+1}/{MAX_RETRIES}] '{term}' → {r.status_code}, waiting {backoff:.1f}s")
                time.sleep(backoff)
            else:
                print(f"  [warn] search '{term}' failed: {exc}")
                return []
        except Exception as exc:
            print(f"  [warn] search '{term}' failed: {exc}")
            return []
    return []


def _extract(product: dict, food_group: str) -> dict | None:
    name = (product.get("product_name") or "").strip()
    if not name:
        return None

    nutriments = product.get("nutriments", {})
    row: dict = {
        "food_id": product.get("code", ""),
        "product_name": name,
        "food_group": food_group,
    }

    for col, key in NUTRIMENT_KEYS.items():
        val = nutriments.get(key)
        try:
            row[col] = float(val) if val is not None else np.nan
        except (ValueError, TypeError):
            row[col] = np.nan

    allergens_tags = product.get("allergens_tags") or []
    for col, tags in ALLERGEN_MAP.items():
        row[col] = int(any(t in allergens_tags for t in tags))

    # Override some allergens from food_group when OFF tags are sparse
    if food_group == "dairy":
        row["contains_lactose"] = 1
    if food_group in ("grain",):
        row["contains_gluten"] = 1
    if food_group == "nut":
        row["contains_nuts"] = 1
    if food_group == "egg":
        row["contains_eggs"] = 1
    if food_group == "fish" and "shrimp" in name.lower():
        row["contains_shellfish"] = 1

    row["is_animal_product"] = int(food_group in ANIMAL_GROUPS)
    row["is_meat"] = int(food_group == "meat")
    row["is_fish"] = int(food_group == "fish")

    # Must have at least calories to be useful
    if np.isnan(row["energy_kcal_100g"]):
        return None

    return row


def fetch_foods(target_per_category: int = 25, use_mock_on_failure: bool = True) -> pd.DataFrame:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    seen_ids: set[str] = set()

    for group, terms in SEARCH_CATEGORIES.items():
        group_rows: list[dict] = []
        print(f"[fetch] {group} …")
        for term in terms:
            if len(group_rows) >= target_per_category:
                break
            products = _search(term)
            time.sleep(SLEEP)
            for p in products:
                fid = p.get("code", "")
                if fid in seen_ids:
                    continue
                row = _extract(p, group)
                if row:
                    group_rows.append(row)
                    seen_ids.add(fid)
                    if len(group_rows) >= target_per_category:
                        break
        print(f"  → {len(group_rows)} alimentos coletados")
        all_rows.extend(group_rows)

    df = pd.DataFrame(all_rows)

    # Save raw snapshot
    raw_path = RAW_DIR / "foods_raw.json"
    df.to_json(raw_path, orient="records", force_ascii=False, indent=2)
    print(f"\n[raw] saved {len(df)} products to {raw_path}")

    # Impute missing nutriment values with group median
    num_cols = list(NUTRIMENT_KEYS.keys())
    for col in num_cols:
        group_medians = df.groupby("food_group")[col].transform("median")
        global_median = df[col].median()
        df[col] = df[col].fillna(group_medians).fillna(global_median).round(2)

    # Clip to physiologically plausible ranges
    df["energy_kcal_100g"] = df["energy_kcal_100g"].clip(0, 900)
    df["sodium_100g"] = df["sodium_100g"].clip(0, 5)        # g → g
    df["proteins_100g"] = df["proteins_100g"].clip(0, 100)
    df["fat_100g"] = df["fat_100g"].clip(0, 100)
    df["carbohydrates_100g"] = df["carbohydrates_100g"].clip(0, 100)
    df["fiber_100g"] = df["fiber_100g"].clip(0, 30)
    df["sugar_100g"] = df["sugar_100g"].clip(0, 100)
    df["saturated_fat_100g"] = df["saturated_fat_100g"].clip(0, 60)

    # Sodium in OFF is stored in g/100g; convert to mg/100g for intuitive thresholds
    df["sodium_mg_100g"] = (df["sodium_100g"] * 1000).round(1)
    df = df.drop(columns=["sodium_100g"])

    # Reset index, assign sequential food_id if missing
    df = df.reset_index(drop=True)
    df["food_id"] = df["food_id"].replace("", pd.NA)
    df["food_id"] = df["food_id"].fillna(
        pd.Series([f"local_{i}" for i in range(len(df))])
    )

    if len(df) < 10:
        if use_mock_on_failure:
            print("[fetch] Too few foods from API — falling back to built-in mock dataset.")
            from src.data.mock_foods import build_mock_foods
            return build_mock_foods()
        else:
            raise RuntimeError("Too few foods fetched and mock fallback is disabled.")

    out_path = PROCESSED_DIR / "foods.csv"
    df.to_csv(out_path, index=False)
    print(f"[processed] saved {len(df)} foods to {out_path}\n")
    return df


if __name__ == "__main__":
    fetch_foods()
