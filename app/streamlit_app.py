
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
IMAGES = ROOT / "images"

st.set_page_config(
    page_title="Retail Forecast & Inventory",
    page_icon="📦",
    layout="wide",
)

st.title("Retail Sales Forecasting & Inventory Optimization")
st.caption(
    "Simulated retail data | Machine learning | Replenishment decision support"
)

forecast_path = OUTPUTS / "sales_forecast.csv"
inventory_path = OUTPUTS / "inventory_recommendations.csv"
metrics_path = OUTPUTS / "model_metrics.csv"

if not all(p.exists() for p in (forecast_path, inventory_path, metrics_path)):
    st.warning("Run the forecasting pipeline before opening this dashboard.")
    st.code(
        "python src/generate_data.py\n"
        "python src/pipeline.py"
    )
    st.stop()

forecasts = pd.read_csv(forecast_path, parse_dates=["date"])
inventory = pd.read_csv(inventory_path)
metrics = pd.read_csv(metrics_path)

st.sidebar.header("Filters")
stores = sorted(inventory["store_id"].unique())
products = sorted(inventory["item_id"].unique())

selected_store = st.sidebar.selectbox("Store", stores)
selected_product = st.sidebar.selectbox("Product", products)

filtered = inventory[
    (inventory["store_id"] == selected_store)
    & (inventory["item_id"] == selected_product)
]

forecast_filtered = forecasts[
    (forecasts["store_id"] == selected_store)
    & (forecasts["item_id"] == selected_product)
].sort_values("date")

if filtered.empty:
    st.error("No inventory recommendation found for the selection.")
    st.stop()

item = filtered.iloc[0]

st.subheader(f"{selected_store} / {selected_product}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Current stock", f"{int(item['on_hand'])} units")
c2.metric("30-day forecast", f"{item['forecast_next_30_days']:.0f} units")
c3.metric("Reorder point", f"{item['reorder_point']:.1f} units")
c4.metric("Recommended order", f"{int(item['recommended_order_qty'])} units")

if item["stock_status"] == "REORDER":
    st.error("Replenishment alert: current stock is at or below the reorder point.")
else:
    st.success("Current stock is above the calculated reorder point.")

st.subheader("Future sales forecast")
st.line_chart(
    forecast_filtered.set_index("date")["forecast_qty"],
    y_label="Predicted units per day",
)


st.subheader("Inventory recommendation")
st.dataframe(
    filtered[[
        "store_id", "item_id", "category", "on_hand",
        "forecast_next_7_days", "forecast_next_30_days",
        "mean_lead_time_demand", "safety_stock", "reorder_point",
        "eoq", "recommended_order_qty", "stock_status",
    ]],
    use_container_width=True,
)


st.download_button(
    label="Download inventory recommendations CSV",
    data=inventory.to_csv(index=False).encode("utf-8"),
    file_name="inventory_recommendations.csv",
    mime="text/csv",
    key="download_inventory_recommendations",
)




st.download_button(
    "Download sales forecasts CSV",
    data=forecasts.to_csv(index=False).encode("utf-8"),
    file_name="sales_forecast.csv",
    mime="text/csv",
)

st.subheader("Forecast model evaluation")
st.dataframe(metrics.round(3), use_container_width=True)

for image_name, title in [
    ("sales_trend.png", "Historical sales trend"),
    ("category_sales.png", "Sales by category"),
    ("actual_vs_predicted.png", "Actual vs predicted sales"),
    ("sales_forecast.png", "Example 30-day forecast"),
]:
    image_path = IMAGES / image_name
    if image_path.exists():
        st.subheader(title)
        st.image(str(image_path), use_container_width=True)

st.caption(
    "Educational prototype. Data and inventory recommendations are simulated; "
    "validate forecasts and stock policies before operational use."
)
