from pathlib import Path

import numpy as np
import pandas as pd


DATA_PATH = Path(__file__).resolve().parent / "steel_price_data.csv"
SEASONS = ["winter", "summer", "monsoon", "festival"]


def build_dataset(rows: int = 240, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=rows, freq="MS")
    trend = np.linspace(0, 160, rows)
    seasonal_wave = 35 * np.sin(np.arange(rows) * 2 * np.pi / 12)

    iron_ore_price = rng.normal(115, 18, rows) + trend * 0.08 + seasonal_wave * 0.08
    coal_price = rng.normal(170, 35, rows) + trend * 0.12
    energy_cost = rng.normal(0.115, 0.018, rows) + trend * 0.00008
    usd_inr = rng.normal(76.5, 3.2, rows) + trend * 0.035
    demand_index = rng.normal(100, 9, rows) + seasonal_wave * 0.18 + trend * 0.035
    production_volume = rng.normal(9.5, 1.1, rows) + trend * 0.004
    inventory_level = rng.normal(5.2, 0.9, rows) - seasonal_wave * 0.012
    season = [SEASONS[(date.month - 1) // 3] for date in dates]

    season_effect = {
        "winter": 40,
        "summer": -15,
        "monsoon": 25,
        "festival": 65,
    }

    price = (
        39000
        + iron_ore_price * 96
        + coal_price * 42
        + energy_cost * 68000
        + usd_inr * 155
        + demand_index * 135
        - production_volume * 370
        - inventory_level * 620
        + np.array([season_effect[item] for item in season])
        + trend * 28
        + rng.normal(0, 850, rows)
    )

    return pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "iron_ore_price_usd_ton": iron_ore_price.round(2),
            "coal_price_usd_ton": coal_price.round(2),
            "energy_cost_usd_kwh": energy_cost.round(3),
            "usd_inr_rate": usd_inr.round(2),
            "demand_index": demand_index.round(2),
            "production_volume_million_ton": production_volume.round(2),
            "inventory_level_million_ton": inventory_level.round(2),
            "season": season,
            "steel_price_inr_ton": price.round(2),
        }
    )


def main() -> None:
    df = build_dataset()
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"Saved {len(df)} rows to {DATA_PATH}")


if __name__ == "__main__":
    main()
