import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "steel_price_data.csv"
MODEL_PATH = ROOT / "models" / "steel_price_model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"
TARGET = "steel_price_inr_ton"

NUMERIC_FEATURES = [
    "iron_ore_price_usd_ton",
    "coal_price_usd_ton",
    "energy_cost_usd_kwh",
    "usd_inr_rate",
    "demand_index",
    "production_volume_million_ton",
    "inventory_level_million_ton",
]
CATEGORICAL_FEATURES = ["season"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run `python3 data/generate_sample_data.py` first."
        )
    return pd.read_csv(path)


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    model = RandomForestRegressor(
        n_estimators=120,
        max_depth=10,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=1,
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def train() -> dict[str, float]:
    df = load_data()
    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")

    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    metrics = {
        "mean_absolute_error_inr_per_ton": round(mean_absolute_error(y_test, predictions), 2),
        "r2_score": round(r2_score(y_test, predictions), 4),
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    metrics = train()
    print("Model trained successfully")
    print(json.dumps(metrics, indent=2))
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
