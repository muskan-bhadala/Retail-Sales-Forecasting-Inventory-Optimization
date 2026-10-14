
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "raw" / "retail_sales.csv"


def generate_dataset(seed: int = 42) -> pd.DataFrame:
    """Generate reproducible simulated retail sales."""
    rng = np.random.default_rng(seed)

    products = [
        ("P001", "Beverages", 30.0, 18.0, 3, 500, 45.0),
        ("P002", "Snacks", 20.0, 11.0, 2, 400, 35.0),
        ("P003", "Household", 120.0, 75.0, 5, 180, 70.0),
        ("P004", "Personal Care", 85.0, 48.0, 4, 220, 55.0),
        ("P005", "Staples", 65.0, 42.0, 3, 300, 40.0),
    ]
    stores = ["S001", "S002", "S003"]
    dates = pd.date_range("2024-01-01", periods=730, freq="D")

    records = []

    for store_index, store_id in enumerate(stores):
        for item_id, category, price, cost, lead, base, order_cost in products:
            for day_index, date in enumerate(dates):
                weekday_effect = (
                    1.20 if date.dayofweek in (5, 6) else 0.95
                )
                trend = 1 + 0.0003 * day_index
                yearly_season = (
                    1 + 0.15 * np.sin(2 * np.pi * day_index / 365.25)
                )
                store_effect = 0.90 + 0.10 * store_index
                on_promo = int(rng.random() < 0.12)
                promo_effect = 1.30 if on_promo else 1.0

                expected = (
                    base / 7
                    * weekday_effect
                    * trend
                    * yearly_season
                    * store_effect
                    * promo_effect
                )
                qty_sold = int(rng.poisson(max(expected, 0.1)))

                discount_pct = (
                    float(rng.choice([0.10, 0.15, 0.20]))
                    if on_promo
                    else 0.0
                )

                records.append({
                    "date": date,
                    "store_id": store_id,
                    "item_id": item_id,
                    "category": category,
                    "qty_sold": qty_sold,
                    "price": round(price * (1 - discount_pct), 2),
                    "on_promo": on_promo,
                    "discount_pct": discount_pct,
                    "stock_on_hand": max(
                        0, int(expected * rng.uniform(3, 12))
                    ),
                    # The first version simulates observable, uncensored
                    # sales. A realistic stockout-censoring experiment
                    # can be added as a separate simulation.
                    "stockout_flag": 0,
                    "lead_time_days": lead,
                    "unit_cost": cost,
                    "ordering_cost": order_cost,
                    "holding_cost_rate": 0.20,
                })

    return pd.DataFrame(records)


if __name__ == "__main__":
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    data = generate_dataset()
    data.to_csv(OUTPUT, index=False)
    print(f"Dataset saved: {OUTPUT}")
    print(f"Rows: {len(data):,}")
    print(f"Stores: {data['store_id'].nunique()}")
    print(f"Products: {data['item_id'].nunique()}")
    print(data.head().to_string(index=False))
