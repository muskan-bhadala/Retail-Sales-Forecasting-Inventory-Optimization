# Retail Sales Forecasting & Inventory Optimization System

A Python-based machine learning project that forecasts retail product demand and generates inventory recommendations to support data-driven replenishment decisions.

## Project Overview

Retail businesses need to predict future demand while maintaining sufficient inventory and minimizing overstocking. This project combines sales data analysis, machine learning, forecasting, and inventory optimization in an interactive Streamlit dashboard.

## Features

- Data cleaning and preprocessing
- Exploratory data analysis (EDA)
- Time-based and lag-based feature engineering
- Random Forest regression for sales forecasting
- Chronological training and testing
- Actual-versus-predicted comparison
- Future sales forecast visualization
- Reorder point and safety stock calculations
- Economic Order Quantity (EOQ) calculation
- Inventory recommendations by store and product
- Interactive Streamlit dashboard
- CSV downloads for forecast and inventory results

## Technology Stack

- **Language:** Python
- **Data processing:** Pandas, NumPy
- **Machine learning:** Scikit-learn
- **Visualization:** Matplotlib, Seaborn, Streamlit
- **Statistical calculations:** SciPy
- **Model persistence:** Joblib

## Model Evaluation

The completed pipeline reported the following results on its evaluation data:

| Metric | Result |
|---|---:|
| Training rows | 10,500 |
| Testing rows | 450 |
| Mean Absolute Error (MAE) | 6.588 |
| Root Mean Squared Error (RMSE) | 8.436 |
| R² score | 0.877 |

These metrics describe the current evaluation run. Performance on new data may differ.

## Project Structure

```text
Retail-Sales-Forecasting-Inventory-Optimization/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── raw/
│   └── processed/
├── images/
├── models/
├── outputs/
├── src/
│   └── pipeline.py
├── screenshots/
├── .gitignore
├── README.md
└── requirements.txt
```

Some folders may be created automatically when the pipeline runs. Adjust this structure to match the files included in the repository.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/Retail-Sales-Forecasting-Inventory-Optimization.git
cd Retail-Sales-Forecasting-Inventory-Optimization
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Prepare the dataset

Place the required CSV dataset at:

```text
data/raw/retail_sales.csv
```

Use the dataset's actual required column names and format.

### 5. Run the forecasting pipeline

```bash
python src/pipeline.py
```

The pipeline generates sales forecasts, evaluation metrics, inventory recommendations, visualizations, and a saved model.

### 6. Launch the dashboard

```bash
python -m streamlit run app/streamlit_app.py
```

Open the local URL shown in the terminal, usually `http://localhost:8501`.

## Generated Outputs

- `outputs/model_metrics.csv` — model evaluation metrics
- `outputs/holdout_predictions.csv` — evaluation predictions
- `outputs/sales_forecast.csv` — future demand forecasts
- `outputs/inventory_recommendations.csv` — inventory planning recommendations
- `outputs/actual_vs_predicted.png` — prediction comparison
- `outputs/sales_forecast.png` — forecast visualization
- `models/retail_sales_model.joblib` — serialized trained model

## Inventory Optimization

The system calculates inventory indicators such as safety stock, reorder point, and Economic Order Quantity (EOQ). Recommendations depend on the data and the inventory assumptions used by the pipeline.

These calculations support planning and should be validated against actual supplier lead times, ordering costs, holding costs, and business requirements before operational use.

## Limitations

- Forecast quality depends on dataset quality and historical patterns.
- Future promotions, price changes, holidays, and supply disruptions may affect demand.
- Inventory recommendations depend on the assumptions and parameters configured in the code.
- This project is intended for learning, demonstration, and decision support.

## Future Enhancements

- Compare Random Forest against additional forecasting models.
- Add configurable forecast horizons and service levels.
- Integrate live sales and inventory data.
- Add automated model monitoring.
- Deploy the dashboard to a cloud platform.

## License

Add your chosen open-source license and its corresponding license file before distributing this project.
