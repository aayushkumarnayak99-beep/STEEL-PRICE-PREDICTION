# Steel Price Prediction Project

A compact machine learning project that predicts steel price per metric ton from market and cost indicators such as iron ore price, coal price, energy cost, USD/INR exchange rate, demand index, production volume, inventory, and season.

The project includes:

- Synthetic but realistic sample steel market data
- A reproducible model training pipeline
- A saved scikit-learn regression model
- Command-line prediction support
- A Streamlit web app for interactive predictions

## Project Structure

```text
.
├── app/
│   └── streamlit_app.py
├── data/
│   └── generate_sample_data.py
├── models/
├── src/
│   └── steel_price_prediction/
│       ├── __init__.py
│       ├── predict.py
│       └── train_model.py
├── tests/
│   └── test_prediction_pipeline.py
├── requirements.txt
└── README.md
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Generate Sample Data

```bash
python3 data/generate_sample_data.py
```

This creates `data/steel_price_data.csv`.

## Train the Model

```bash
python3 -m src.steel_price_prediction.train_model
```

The trained model is saved at `models/steel_price_model.joblib`, and metrics are saved at `models/metrics.json`.

## Predict From the Command Line

```bash
python3 -m src.steel_price_prediction.predict \
  --iron-ore-price 122 \
  --coal-price 188 \
  --energy-cost 0.12 \
  --usd-inr 83.4 \
  --demand-index 104 \
  --production-volume 9.8 \
  --inventory-level 5.1 \
  --season monsoon
```

## Run the Web App

```bash
python3 -m streamlit run app/streamlit_app.py --server.fileWatcherType none
```

## Notes

This is a learning and demo project. The included data is generated locally and should be replaced with real steel market history before using the model for business decisions. Good real-world features may include regional steel indices, import/export duties, freight rates, scrap prices, capacity utilization, order-book data, macroeconomic indicators, and lagged prices.
