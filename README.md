# Walmart Sales Forecasting
Project Overview

Forecast weekly sales for Walmart stores using historical sales data. The project uses time-based features, lagged sales, rolling averages, and holiday indicators to predict weekly sales with XGBoost.


Dataset

Kaggle Walmart Sales Forecast dataset: Link

Includes four CSV files:

train.csv – Historical weekly sales

test.csv – Test dataset for predictions

features.csv – Additional features like temperature, CPI, fuel price

stores.csv – Store-level information

Features Used

Time Features: day, month, year, dayofweek, quarter

Holiday Indicator: IsHoliday

Lag Features: lag_1, lag_2

Rolling Averages: rolling_mean_4, rolling_mean_12

Tools & Libraries

Python 3.12

Pandas, NumPy

Matplotlib

Scikit-learn

XGBoost

Statsmodels

Project Structure

Walmart-Sales-Forecasting/
│
├── salesforecasting.py
├── train.csv
├── test.csv
├── features.csv
├── stores.csv
├── figures/
│   ├── task7_1.png
│   └── task7_2.png
└── README.md

License

Educational / Personal Use. Dataset is from Kaggle under their dataset license.

Store & Department
