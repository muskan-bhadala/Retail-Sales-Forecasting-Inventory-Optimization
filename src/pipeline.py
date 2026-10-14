"""
Retail Sales Forecasting & Inventory Optimization Pipeline

Run from the project root:
    python src/pipeline.py

Expected input:
    data/raw/retail_sales.csv

Main outputs:
    outputs/sales_forecast.csv
    outputs/inventory_recommendations.csv
    outputs/model_metrics.csv
    outputs/actual_vs_predicted.png
    outputs/category_sales.png
    outputs/sales_trend.png
    outputs/sales_forecast.png
    models/retail_sales_model.joblib
"""

from pathlib import Path
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data" / "raw" / "retail_sales.csv"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"

TARGET = "qty_sold"
GROUP_COLUMNS = ["store_id", "item_id"]
FORECAST_DAYS = 30
TEST_DAYS = 30
RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "store_code",
    "item_code",
    "day_of_year",
    "day_of_week",
    "day_of_month",
    "month",
    "week_of_year",
    "is_weekend",
    "on_promo",
    "discount_pct",
    "price",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_28",
    "rolling_std_7",
    "rolling_std_14",
    "rolling_std_28",
]


def ensure_directories():
    """Create folders used by the pipeline."""
    for folder in (OUTPUT_DIR, MODEL_DIR, ROOT / "data" / "processed"):
        folder.mkdir(parents=True, exist_ok=True)


def load_and_clean_data(path=RAW_FILE):
    """Load the retail dataset, validate required fields, and clean basic issues."""
    if not path.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {path}\n"
            "Generate it first with: python src/generate_data.py"
        )

    df = pd.read_csv(path)
    required = ["date", "store_id", "item_id", TARGET]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "store_id", "item_id", TARGET]).copy()

    for column in [
        TARGET,
        "price",
        "on_promo",
        "discount_pct",
        "stock_on_hand",
        "lead_time_days",
        "unit_cost",
        "ordering_cost",
        "holding_cost_rate",
    ]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df[TARGET] = df[TARGET].clip(lower=0)
    for column in ["price", "stock_on_hand", "lead_time_days", "unit_cost",
                   "ordering_cost", "holding_cost_rate", "discount_pct"]:
        if column in df.columns:
            df[column] = df[column].fillna(df[column].median() if df[column].notna().any() else 0)

    if "on_promo" in df.columns:
        df["on_promo"] = df["on_promo"].fillna(0).astype(int).clip(0, 1)
    else:
        df["on_promo"] = 0

    if "discount_pct" not in df.columns:
        df["discount_pct"] = 0.0
    if "price" not in df.columns:
        df["price"] = 0.0

    # Keep one record per store-product-date to prevent duplicate time points.
    df = (
        df.sort_values(["store_id", "item_id", "date"])
        .drop_duplicates(["store_id", "item_id", "date"], keep="last")
        .reset_index(drop=True)
    )

    if df.empty:
        raise ValueError("No valid rows remain after cleaning the dataset.")

    df["store_id"] = df["store_id"].astype(str)
    df["item_id"] = df["item_id"].astype(str)

    print(f"Clean dataset: {len(df):,} rows")
    return df


def build_features(df):
    """Create leakage-safe calendar, lag, and rolling features."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(GROUP_COLUMNS + ["date"]).reset_index(drop=True)

    # Stable categorical codes are sufficient for this local model.
    df["store_code"] = df["store_id"].astype("category").cat.codes.astype(int)
    df["item_code"] = df["item_id"].astype("category").cat.codes.astype(int)

    df["day_of_year"] = df["date"].dt.dayofyear
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    grouped = df.groupby(GROUP_COLUMNS, sort=False)[TARGET]
    for lag in (1, 7, 14, 28):
        df[f"lag_{lag}"] = grouped.shift(lag)

    for window in (7, 14, 28):
        df[f"rolling_mean_{window}"] = grouped.transform(
            lambda series, w=window: series.shift(1).rolling(w, min_periods=1).mean()
        )
        df[f"rolling_std_{window}"] = grouped.transform(
            lambda series, w=window: series.shift(1).rolling(w, min_periods=2).std()
        )

    lag_columns = [f"lag_{lag}" for lag in (1, 7, 14, 28)]
    mean_columns = [f"rolling_mean_{window}" for window in (7, 14, 28)]
    std_columns = [f"rolling_std_{window}" for window in (7, 14, 28)]

    df[lag_columns + mean_columns] = df[lag_columns + mean_columns].fillna(0)
    df[std_columns] = df[std_columns].fillna(0)
    return df


def calculate_inventory_policy(
    forecast,
    residual_std,
    on_hand,
    lead_time,
    service_level=0.95,
    annual_demand=None,
    ordering_cost=50,
    unit_cost=20,
    holding_rate=0.20,
):
    """
    Calculate replenishment policy from a forecast.

    residual_std is the standard deviation of daily forecast errors.
    The lead-time demand uncertainty approximation is residual_std * sqrt(lead_time).
    """
    if not 0 < service_level < 1:
        raise ValueError("service_level must be strictly between 0 and 1.")
    if lead_time is None or float(lead_time) < 0:
        raise ValueError("lead_time must be non-negative.")

    forecast_values = np.asarray(list(forecast), dtype=float)
    forecast_values = np.clip(forecast_values, 0, None)
    if forecast_values.size == 0:
        forecast_values = np.array([0.0])

    lead_time = float(lead_time)
    on_hand = max(float(on_hand or 0), 0)
    residual_std = max(float(residual_std or 0), 0)
    mean_daily_demand = float(np.mean(forecast_values))
    mean_lead_time_demand = mean_daily_demand * lead_time
    z_score = float(norm.ppf(service_level))
    safety_stock = z_score * residual_std * np.sqrt(lead_time)
    reorder_point = mean_lead_time_demand + safety_stock

    if annual_demand is None:
        annual_demand = mean_daily_demand * 365
    annual_demand = max(float(annual_demand or 0), 0)
    ordering_cost = max(float(ordering_cost or 0), 0)
    unit_cost = max(float(unit_cost or 0), 0)
    holding_rate = max(float(holding_rate or 0), 0)
    annual_holding_cost = unit_cost * holding_rate

    if annual_holding_cost > 0:
        eoq = float(np.sqrt((2 * annual_demand * ordering_cost) / annual_holding_cost))
    else:
        eoq = 0.0

    recommended_order_qty = max(0, int(np.ceil(max(reorder_point - on_hand, 0))))
    if on_hand <= reorder_point:
        stock_status = "REORDER"
        if eoq > recommended_order_qty:
            recommended_order_qty = int(np.ceil(eoq))
    elif on_hand <= reorder_point * 1.25:
        stock_status = "LOW"
    else:
        stock_status = "SUFFICIENT"

    return {
        "mean_lead_time_demand": round(float(mean_lead_time_demand), 2),
        "safety_stock": round(float(safety_stock), 2),
        "reorder_point": round(float(reorder_point), 2),
        "eoq": round(float(eoq), 2),
        "recommended_order_qty": int(recommended_order_qty),
        "stock_status": stock_status,
    }


def make_model():
    """Create a robust baseline model for tabular time-series features."""
    return RandomForestRegressor(
        n_estimators=160,
        max_depth=14,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def evaluate_model(model, test_df, feature_columns):
    """Return test predictions and standard regression metrics."""
    X_test = test_df[feature_columns]
    y_test = test_df[TARGET].astype(float)
    predictions = np.clip(model.predict(X_test), 0, None)

    metrics = {
        "model": "RandomForestRegressor",
        "test_rows": int(len(test_df)),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "r2": float(r2_score(y_test, predictions)) if len(test_df) > 1 else float("nan"),
    }
    return predictions, metrics


def make_future_feature_row(
    store_id, item_id, date, history, store_code, item_code, promo, discount, price
):
    """Build one future row using only known history/predictions."""
    series = np.asarray(history, dtype=float)
    row = {
        "store_code": store_code,
        "item_code": item_code,
        "day_of_year": date.dayofyear,
        "day_of_week": date.dayofweek,
        "day_of_month": date.day,
        "month": date.month,
        "week_of_year": int(date.isocalendar().week),
        "is_weekend": int(date.dayofweek >= 5),
        "on_promo": int(promo),
        "discount_pct": float(discount),
        "price": float(price),
    }

    for lag in (1, 7, 14, 28):
        row[f"lag_{lag}"] = float(series[-lag]) if len(series) >= lag else 0.0

    for window in (7, 14, 28):
        tail = series[-window:]
        row[f"rolling_mean_{window}"] = float(np.mean(tail)) if len(tail) else 0.0
        row[f"rolling_std_{window}"] = float(np.std(tail, ddof=1)) if len(tail) > 1 else 0.0

    return row


def forecast_future(model, df, feature_columns, forecast_days=FORECAST_DAYS):
    """Recursively forecast daily demand for each store-product pair."""
    rows = []
    latest = df.sort_values(GROUP_COLUMNS + ["date"]).copy()
    store_codes = latest.groupby("store_id")["store_code"].first().to_dict()
    item_codes = latest.groupby("item_id")["item_code"].first().to_dict()

    for (store_id, item_id), group in latest.groupby(GROUP_COLUMNS, sort=True):
        group = group.sort_values("date")
        history = group[TARGET].astype(float).tolist()
        last_date = group["date"].max()
        latest_row = group.iloc[-1]

        promo = int(latest_row.get("on_promo", 0))
        discount = float(latest_row.get("discount_pct", 0.0))
        price = float(latest_row.get("price", 0.0))

        for offset in range(1, forecast_days + 1):
            future_date = last_date + pd.Timedelta(days=offset)
            feature_row = make_future_feature_row(
                store_id,
                item_id,
                future_date,
                history,
                int(store_codes[store_id]),
                int(item_codes[item_id]),
                promo,
                discount,
                price,
            )
            feature_frame = pd.DataFrame([feature_row], columns=feature_columns)
            prediction = max(0.0, float(model.predict(feature_frame)[0]))
            rows.append({
                "date": future_date,
                "store_id": store_id,
                "item_id": item_id,
                "category": latest_row.get("category", "Unknown"),
                "forecast_qty": round(prediction, 2),
                "price": price,
            })
            history.append(prediction)

    return pd.DataFrame(rows)


def create_inventory_recommendations(df, forecasts, residual_std):
    """Aggregate forecasts and calculate inventory recommendations."""
    results = []
    group_cols = GROUP_COLUMNS

    for (store_id, item_id), forecast_group in forecasts.groupby(group_cols, sort=True):
        history_group = df[
            (df["store_id"] == store_id) & (df["item_id"] == item_id)
        ].sort_values("date")
        latest = history_group.iloc[-1]

        forecast_7 = forecast_group.head(7)["forecast_qty"].sum()
        forecast_30 = forecast_group.head(30)["forecast_qty"].sum()
        daily_forecast = forecast_group["forecast_qty"].tolist()

        lead_time = float(latest.get("lead_time_days", 7) or 7)
        on_hand = float(latest.get("stock_on_hand", 0) or 0)
        unit_cost = float(latest.get("unit_cost", 20) or 20)
        ordering_cost = float(latest.get("ordering_cost", 50) or 50)
        holding_rate = float(latest.get("holding_cost_rate", 0.20) or 0.20)
        annual_demand = float(history_group[TARGET].mean() * 365)

        policy = calculate_inventory_policy(
            forecast=daily_forecast,
            residual_std=residual_std,
            on_hand=on_hand,
            lead_time=lead_time,
            service_level=0.95,
            annual_demand=annual_demand,
            ordering_cost=ordering_cost,
            unit_cost=unit_cost,
            holding_rate=holding_rate,
        )

        results.append({
            "store_id": store_id,
            "item_id": item_id,
            "category": latest.get("category", "Unknown"),
            "on_hand": round(on_hand, 2),
            "forecast_next_7_days": round(float(forecast_7), 2),
            "forecast_next_30_days": round(float(forecast_30), 2),
            "lead_time_days": lead_time,
            "mean_lead_time_demand": policy["mean_lead_time_demand"],
            "safety_stock": policy["safety_stock"],
            "reorder_point": policy["reorder_point"],
            "eoq": policy["eoq"],
            "recommended_order_qty": policy["recommended_order_qty"],
            "stock_status": policy["stock_status"],
            "unit_cost": unit_cost,
        })

    return pd.DataFrame(results).sort_values(
        ["stock_status", "recommended_order_qty"],
        ascending=[True, False],
    ).reset_index(drop=True)


def save_charts(df, test_df, test_predictions, forecasts):
    """Create portfolio-friendly PNG charts."""
    plt.figure(figsize=(10, 5))
    daily = df.groupby("date", as_index=False)[TARGET].sum()
    plt.plot(daily["date"], daily[TARGET], linewidth=1.2)
    plt.title("Daily Retail Sales Across All Stores and Products")
    plt.xlabel("Date")
    plt.ylabel("Units Sold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sales_trend.png", dpi=150)
    plt.close()

    if "category" in df.columns:
        plt.figure(figsize=(9, 5))
        category_sales = df.groupby("category")[TARGET].sum().sort_values(ascending=False)
        category_sales.plot(kind="bar")
        plt.title("Total Sales by Category")
        plt.xlabel("Category")
        plt.ylabel("Units Sold")
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "category_sales.png", dpi=150)
        plt.close()

    plt.figure(figsize=(7, 7))
    actual = test_df[TARGET].to_numpy(dtype=float)
    plt.scatter(actual, test_predictions, alpha=0.35)
    max_value = max(float(np.max(actual)), float(np.max(test_predictions)), 1.0)
    plt.plot([0, max_value], [0, max_value], linestyle="--")
    plt.title("Actual vs Predicted Demand (Holdout Test Set)")
    plt.xlabel("Actual Units Sold")
    plt.ylabel("Predicted Units Sold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "actual_vs_predicted.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    forecast_daily = forecasts.groupby("date", as_index=False)["forecast_qty"].sum()
    plt.plot(forecast_daily["date"], forecast_daily["forecast_qty"], marker="o", markersize=2)
    plt.title(f"Total Demand Forecast for the Next {FORECAST_DAYS} Days")
    plt.xlabel("Date")
    plt.ylabel("Forecast Units")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sales_forecast.png", dpi=150)
    plt.close()


def main():
    ensure_directories()
    df = load_and_clean_data()

    # Save cleaned source data for reproducibility.
    df.to_csv(ROOT / "data" / "processed" / "retail_sales_clean.csv", index=False)

    featured = build_features(df)
    feature_columns = [column for column in NUMERIC_FEATURES if column in featured.columns]
    if not feature_columns:
        raise ValueError("No model features were created. Check build_features().")

    # Use the last TEST_DAYS dates as a chronological holdout.
    unique_dates = np.sort(featured["date"].unique())
    if len(unique_dates) < 2:
        raise ValueError("At least two distinct dates are required to train and evaluate.")

    test_days = min(TEST_DAYS, max(1, len(unique_dates) // 5))
    cutoff = pd.Timestamp(unique_dates[-test_days])
    train_df = featured[featured["date"] < cutoff].copy()
    test_df = featured[featured["date"] >= cutoff].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Chronological train/test split produced an empty partition.")

    # Feature construction uses shifted lags/rolls. Initial rows are retained with
    # zero-filled history so the entire chronology remains usable.
    model = make_model()
    model.fit(train_df[feature_columns], train_df[TARGET].astype(float))
    test_predictions, metrics = evaluate_model(model, test_df, feature_columns)

    # Estimate forecast uncertainty from the holdout errors.
    residuals = test_df[TARGET].to_numpy(dtype=float) - test_predictions
    residual_std = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0
    metrics["residual_std"] = residual_std
    metrics["train_rows"] = int(len(train_df))
    metrics["test_start_date"] = str(pd.Timestamp(cutoff).date())
    metrics["test_end_date"] = str(pd.Timestamp(test_df["date"].max()).date())

    # Refit on all historical observations before producing future forecasts.
    final_model = make_model()
    final_model.fit(featured[feature_columns], featured[TARGET].astype(float))
    joblib.dump(
        {"model": final_model, "feature_columns": feature_columns},
        MODEL_DIR / "retail_sales_model.joblib",
    )

    forecasts = forecast_future(final_model, featured, feature_columns, FORECAST_DAYS)
    recommendations = create_inventory_recommendations(df, forecasts, residual_std)

    # Export predictions and metrics.
    forecasts.to_csv(OUTPUT_DIR / "sales_forecast.csv", index=False)
    recommendations.to_csv(OUTPUT_DIR / "inventory_recommendations.csv", index=False)
    pd.DataFrame([metrics]).to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)

    # Holdout predictions are saved for auditability and charting.
    holdout_output = test_df[["date", "store_id", "item_id", TARGET]].copy()
    holdout_output["predicted_qty"] = test_predictions
    holdout_output.to_csv(OUTPUT_DIR / "holdout_predictions.csv", index=False)

    save_charts(df, test_df, test_predictions, forecasts)

    print("\nPipeline completed successfully.")
    print(f"Training rows: {len(train_df):,}")
    print(f"Testing rows:  {len(test_df):,}")
    print(f"MAE:  {metrics['mae']:.3f}")
    print(f"RMSE: {metrics['rmse']:.3f}")
    print(f"R²:   {metrics['r2']:.3f}")
    print(f"Forecast rows: {len(forecasts):,}")
    print(f"Inventory recommendations: {len(recommendations):,}")
    print(f"Outputs saved in: {OUTPUT_DIR}")
    print(f"Model saved to: {MODEL_DIR / 'retail_sales_model.joblib'}")


if __name__ == "__main__":
    main()
