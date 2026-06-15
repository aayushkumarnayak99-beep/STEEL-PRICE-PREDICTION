import argparse
from pathlib import Path

import joblib
import pandas as pd

from .train_model import FEATURES, MODEL_PATH


def load_model(model_path: Path = MODEL_PATH):
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run `python3 -m src.steel_price_prediction.train_model` first."
        )
    return joblib.load(model_path)


def predict_price(input_values: dict) -> float:
    model = load_model()
    row = pd.DataFrame([input_values], columns=FEATURES)
    return float(model.predict(row)[0])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict steel price in INR per metric ton.")
    parser.add_argument("--iron-ore-price", type=float, required=True)
    parser.add_argument("--coal-price", type=float, required=True)
    parser.add_argument("--energy-cost", type=float, required=True)
    parser.add_argument("--usd-inr", type=float, required=True)
    parser.add_argument("--demand-index", type=float, required=True)
    parser.add_argument("--production-volume", type=float, required=True)
    parser.add_argument("--inventory-level", type=float, required=True)
    parser.add_argument(
        "--season",
        choices=["winter", "summer", "monsoon", "festival"],
        required=True,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_values = {
        "iron_ore_price_usd_ton": args.iron_ore_price,
        "coal_price_usd_ton": args.coal_price,
        "energy_cost_usd_kwh": args.energy_cost,
        "usd_inr_rate": args.usd_inr,
        "demand_index": args.demand_index,
        "production_volume_million_ton": args.production_volume,
        "inventory_level_million_ton": args.inventory_level,
        "season": args.season,
    }
    price = predict_price(input_values)
    print(f"Predicted steel price: INR {price:,.2f} per metric ton")


if __name__ == "__main__":
    main()
