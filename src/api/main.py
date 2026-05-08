"""
FastAPI nutritional score engine.

Endpoints
---------
GET  /health
GET  /foods
POST /individuals
GET  /individuals/{individual_id}
GET  /individuals/{individual_id}/top-foods
POST /score
"""

import uuid
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from src.features.preprocessing import (
    ALL_FEATURE_COLS,
    IND_NUM, IND_CAT, IND_BIN,
    FOOD_NUM, FOOD_CAT, FOOD_BIN,
)
from src.data.generate_pairs import compute_score

MODELS_DIR    = Path(__file__).parents[2] / "data" / "models"
PROCESSED_DIR = Path(__file__).parents[2] / "data" / "processed"

app = FastAPI(
    title="Nutritional Score Engine",
    description="Score de compatibilidade nutricional (0–10) para pares indivíduo × alimento.",
    version="0.1.0",
)

# ─────────────────────────────────────────────
# Startup: load model, preprocessor, foods
# ─────────────────────────────────────────────
_model        = None
_preprocessor = None
_foods_df: pd.DataFrame | None = None
_individuals: dict[str, dict] = {}   # in-memory store keyed by UUID


@app.on_event("startup")
def _load_artefacts() -> None:
    global _model, _preprocessor, _foods_df
    try:
        _model        = joblib.load(MODELS_DIR / "mlp_model.pkl")
        _preprocessor = joblib.load(MODELS_DIR / "preprocessor.pkl")
        _foods_df     = pd.read_csv(PROCESSED_DIR / "foods.csv")
        print(f"[api] model loaded | {len(_foods_df)} foods available")
    except FileNotFoundError as exc:
        print(f"[api] WARNING – artefact not found: {exc}. Run run_pipeline.py first.")


# ─────────────────────────────────────────────
# Pydantic schemas
# ─────────────────────────────────────────────

class IndividualIn(BaseModel):
    name: str = Field(..., example="João Silva")
    age: int = Field(..., ge=18, le=110, example=35)
    diet_type: str = Field(..., example="vegetarian")
    allergies: list[str] = Field(default_factory=list, example=["lactose"])
    total_cholesterol: int = Field(..., ge=100, le=400, example=210)
    weight_kg: float = Field(..., gt=30, le=300, example=70.0)
    height_cm: float = Field(..., gt=100, le=250, example=175.0)
    restrictions: list[str] = Field(default_factory=list, example=["low_sugar"])
    goal: str = Field(..., example="weight_loss")
    activity_level: str = Field(..., example="moderately_active")
    glycemic_condition: str = Field(default="none", example="pre_diabetic")
    hypertension: str = Field(default="none", example="none")

    @field_validator("diet_type")
    @classmethod
    def validate_diet(cls, v: str) -> str:
        valid = {"omnivore", "vegetarian", "vegan", "keto", "pescatarian", "paleo"}
        if v not in valid:
            raise ValueError(f"diet_type must be one of {valid}")
        return v

    @field_validator("goal")
    @classmethod
    def validate_goal(cls, v: str) -> str:
        valid = {"weight_loss", "muscle_gain", "maintenance", "health_improvement", "energy_boost"}
        if v not in valid:
            raise ValueError(f"goal must be one of {valid}")
        return v

    @field_validator("activity_level")
    @classmethod
    def validate_activity(cls, v: str) -> str:
        valid = {"sedentary", "lightly_active", "moderately_active", "very_active"}
        if v not in valid:
            raise ValueError(f"activity_level must be one of {valid}")
        return v

    @field_validator("glycemic_condition")
    @classmethod
    def validate_glycemic(cls, v: str) -> str:
        valid = {"none", "pre_diabetic", "type_1", "type_2"}
        if v not in valid:
            raise ValueError(f"glycemic_condition must be one of {valid}")
        return v

    @field_validator("hypertension")
    @classmethod
    def validate_hypertension(cls, v: str) -> str:
        valid = {"none", "controlled", "uncontrolled"}
        if v not in valid:
            raise ValueError(f"hypertension must be one of {valid}")
        return v


class IndividualOut(BaseModel):
    individual_id: str
    name: str
    bmi: float


class ScoreRequest(BaseModel):
    individual_id: str
    food_id: str


class ScoreBreakdown(BaseModel):
    allergen_safe: bool
    diet_compatible: bool
    goal_alignment: str
    health_flags: list[str]
    heuristic_reference: float


class ScoreResponse(BaseModel):
    individual_id: str
    food_id: str
    food_name: str
    score: float
    breakdown: ScoreBreakdown


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

VALID_ALLERGENS   = {"gluten", "lactose", "nuts", "shellfish", "eggs", "soy"}
VALID_RESTRICTIONS = {"low_sodium", "low_sugar", "low_fat", "high_protein", "low_carb"}


def _individual_to_series(data: dict) -> pd.Series:
    """Converts stored individual dict into a feature Series for scoring."""
    row: dict[str, Any] = {}
    for col in IND_NUM:
        row[col] = data.get(col, 0)
    for col in IND_CAT:
        row[col] = data.get(col, "none")
    for col in IND_BIN:
        row[col] = data.get(col, 0)
    return pd.Series(row)


def _food_row_to_series(food_row: pd.Series) -> pd.Series:
    """Ensures a food DataFrame row has all expected feature columns."""
    row: dict[str, Any] = {}
    for col in FOOD_NUM + FOOD_CAT + FOOD_BIN:
        row[col] = food_row.get(col, 0)
    return pd.Series(row)


def _build_breakdown(
    ind: pd.Series,
    food: pd.Series,
    heuristic_score: float,
) -> ScoreBreakdown:
    allergens = [a for a in ["gluten", "lactose", "nuts", "shellfish", "eggs", "soy"]
                 if ind.get(f"allergy_{a}", 0) and food.get(f"contains_{a}", 0)]
    allergen_safe = len(allergens) == 0

    diet = ind.get("diet_type", "")
    diet_compat = True
    if diet == "vegan" and food.get("is_animal_product", 0):
        diet_compat = False
    elif diet in ("vegetarian",) and (food.get("is_meat", 0) or food.get("is_fish", 0)):
        diet_compat = False
    elif diet == "pescatarian" and food.get("is_meat", 0):
        diet_compat = False

    goal = ind.get("goal", "")
    protein = food.get("proteins_100g", 0)
    energy  = food.get("energy_kcal_100g", 0)
    fiber   = food.get("fiber_100g", 0)
    if goal == "muscle_gain" and protein >= 20:
        goal_alignment = "high"
    elif goal == "weight_loss" and energy < 200 and fiber > 3:
        goal_alignment = "high"
    elif goal == "health_improvement" and food.get("food_group") in ("vegetable", "fruit", "legume"):
        goal_alignment = "high"
    elif heuristic_score >= 7:
        goal_alignment = "moderate"
    elif heuristic_score >= 4:
        goal_alignment = "low"
    else:
        goal_alignment = "poor"

    flags: list[str] = []
    sodium = food.get("sodium_mg_100g", 0)
    sugar  = food.get("sugar_100g", 0)
    sat_fat = food.get("saturated_fat_100g", 0)
    if ind.get("hypertension", "none") != "none" and sodium > 200:
        flags.append(f"sodium: caution ({sodium:.0f} mg/100g)")
    if ind.get("glycemic_condition", "none") != "none" and sugar > 8:
        flags.append(f"sugar: caution ({sugar:.1f} g/100g)")
    if ind.get("total_cholesterol", 0) > 240 and sat_fat > 5:
        flags.append(f"saturated fat: caution ({sat_fat:.1f} g/100g)")

    return ScoreBreakdown(
        allergen_safe=allergen_safe,
        diet_compatible=diet_compat,
        goal_alignment=goal_alignment,
        health_flags=flags,
        heuristic_reference=round(heuristic_score, 2),
    )


def _mlp_score(ind_series: pd.Series, food_row: pd.Series) -> float:
    """Runs the trained MLP on a single pair."""
    if _model is None or _preprocessor is None:
        return -1.0

    feature_cols = [c for c in ALL_FEATURE_COLS if c in list(ind_series.index) + list(food_row.index)]
    combined = pd.concat([ind_series, food_row]).to_frame().T
    combined = combined.reindex(columns=ALL_FEATURE_COLS, fill_value=0)

    X = _preprocessor.transform(combined)
    pred = float(_model.predict(X)[0])
    return round(float(np.clip(pred, 0, 10)), 2)


def _get_ind_series(individual_id: str) -> pd.Series:
    data = _individuals.get(individual_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Individual not found")
    return _individual_to_series(data)


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.get("/health", tags=["system"])
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "foods_loaded": _foods_df is not None,
        "n_foods": len(_foods_df) if _foods_df is not None else 0,
        "n_individuals_registered": len(_individuals),
    }


@app.get("/foods", tags=["foods"])
def list_foods(limit: int = 100, food_group: str | None = None) -> list[dict]:
    if _foods_df is None:
        raise HTTPException(status_code=503, detail="Foods not loaded. Run pipeline first.")
    df = _foods_df
    if food_group:
        df = df[df["food_group"] == food_group]
    cols = ["food_id", "product_name", "food_group",
            "energy_kcal_100g", "proteins_100g", "carbohydrates_100g", "fat_100g"]
    return df[cols].head(limit).to_dict(orient="records")


@app.post("/individuals", response_model=IndividualOut, status_code=201, tags=["individuals"])
def create_individual(body: IndividualIn) -> IndividualOut:
    individual_id = str(uuid.uuid4())
    bmi = round(body.weight_kg / (body.height_cm / 100) ** 2, 1)

    data: dict[str, Any] = {
        "individual_id": individual_id,
        "name": body.name,
        "age": body.age,
        "diet_type": body.diet_type,
        "total_cholesterol": body.total_cholesterol,
        "weight_kg": body.weight_kg,
        "height_cm": body.height_cm,
        "bmi": bmi,
        "goal": body.goal,
        "activity_level": body.activity_level,
        "glycemic_condition": body.glycemic_condition,
        "hypertension": body.hypertension,
    }

    for allergen in VALID_ALLERGENS:
        data[f"allergy_{allergen}"] = int(allergen in body.allergies)
    for restriction in VALID_RESTRICTIONS:
        data[f"restriction_{restriction}"] = int(restriction in body.restrictions)

    _individuals[individual_id] = data
    return IndividualOut(individual_id=individual_id, name=body.name, bmi=bmi)


@app.get("/individuals/{individual_id}", tags=["individuals"])
def get_individual(individual_id: str) -> dict:
    data = _individuals.get(individual_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Individual not found")
    return data


@app.get("/individuals/{individual_id}/top-foods", tags=["individuals"])
def top_foods(individual_id: str, limit: int = 10) -> dict:
    if _foods_df is None:
        raise HTTPException(status_code=503, detail="Foods not loaded. Run pipeline first.")

    ind_data   = _individuals.get(individual_id)
    if ind_data is None:
        raise HTTPException(status_code=404, detail="Individual not found")

    ind_series = _individual_to_series(ind_data)
    ind_pd     = pd.Series(ind_data)  # for heuristic (uses raw field names)

    results = []
    for _, food_row in _foods_df.iterrows():
        food_series = _food_row_to_series(food_row)
        heuristic   = compute_score(ind_pd, food_row, noise_std=0.0)
        mlp_s       = _mlp_score(ind_series, food_series)
        score       = mlp_s if mlp_s >= 0 else heuristic
        breakdown   = _build_breakdown(ind_series, food_series, heuristic)
        results.append({
            "food_id":   str(food_row["food_id"]),
            "food_name": str(food_row["product_name"]),
            "food_group": str(food_row["food_group"]),
            "score":     score,
            "breakdown": breakdown.model_dump(),
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return {
        "individual_id": individual_id,
        "name": ind_data.get("name", ""),
        "top_foods": results[:limit],
    }


@app.post("/score", response_model=ScoreResponse, tags=["score"])
def score_pair(body: ScoreRequest) -> ScoreResponse:
    if _foods_df is None:
        raise HTTPException(status_code=503, detail="Foods not loaded. Run pipeline first.")

    ind_data = _individuals.get(body.individual_id)
    if ind_data is None:
        raise HTTPException(status_code=404, detail="Individual not found")

    food_rows = _foods_df[_foods_df["food_id"].astype(str) == body.food_id]
    if food_rows.empty:
        raise HTTPException(status_code=404, detail="Food not found")

    food_row    = food_rows.iloc[0]
    ind_series  = _individual_to_series(ind_data)
    food_series = _food_row_to_series(food_row)
    ind_pd      = pd.Series(ind_data)

    heuristic   = compute_score(ind_pd, food_row, noise_std=0.0)
    mlp_s       = _mlp_score(ind_series, food_series)
    final_score = mlp_s if mlp_s >= 0 else heuristic

    breakdown = _build_breakdown(ind_series, food_series, heuristic)

    return ScoreResponse(
        individual_id=body.individual_id,
        food_id=body.food_id,
        food_name=str(food_row["product_name"]),
        score=final_score,
        breakdown=breakdown,
    )
