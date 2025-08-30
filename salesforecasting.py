# ===============================
# Walmart Sales Forecasting
# Using train.csv and features.csv
# ===============================

# 1️⃣ Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from statsmodels.tsa.seasonal import seasonal_decompose

# 2️⃣ Load train.csv
train = pd.read_csv('train.csv', parse_dates=['Date'])
print("Columns in train.csv:", train.columns)
print(train.head())

# 3️⃣ Feature Engineering
train['day'] = train['Date'].dt.day
train['month'] = train['Date'].dt.month
train['year'] = train['Date'].dt.year
train['dayofweek'] = train['Date'].dt.dayofweek
train['quarter'] = train['Date'].dt.quarter

# Convert IsHoliday to int (False=0, True=1)
train['IsHoliday'] = train['IsHoliday'].astype(int)

# Sort by Store, Dept, Date
train = train.sort_values(['Store','Dept','Date'])

# Lag features per Store & Dept
train['lag_1'] = train.groupby(['Store','Dept'])['Weekly_Sales'].shift(1)
train['lag_2'] = train.groupby(['Store','Dept'])['Weekly_Sales'].shift(2)

# Rolling averages per Store & Dept
train['rolling_mean_4'] = train.groupby(['Store','Dept'])['Weekly_Sales'].transform(lambda x: x.rolling(4).mean())
train['rolling_mean_12'] = train.groupby(['Store','Dept'])['Weekly_Sales'].transform(lambda x: x.rolling(12).mean())

# Drop rows with missing values
train.dropna(inplace=True)

# Optional: Seasonal decomposition for one store example
sample_store = train[train['Store']==1].groupby('Date')['Weekly_Sales'].sum()
decomposition = seasonal_decompose(sample_store, model='multiplicative', period=52)
decomposition.plot()
plt.show()

# 4️⃣ Prepare Features and Target
features = ['Store','Dept','day','month','year','dayofweek','quarter','IsHoliday',
            'lag_1','lag_2','rolling_mean_4','rolling_mean_12']
X = train[features]
y = train['Weekly_Sales']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5️⃣ Train-Test Split (time-series aware)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

# 6️⃣ Initialize XGBoost Model
model = XGBRegressor(n_estimators=200, learning_rate=0.1, random_state=42, tree_method='hist')

# 7️⃣ Train the Model
model.fit(X_train, y_train)

# 8️⃣ Make Predictions
y_pred = model.predict(X_test)

# 9️⃣ Evaluate Model
print("\nXGBoost Model Performance:")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"R²: {r2_score(y_test, y_pred):.2f}")

# 10️⃣ Plot Actual vs Predicted Sales (for the test set)
plt.figure(figsize=(12,6))
plt.plot(y_test.index, y_test, label='Actual Sales', color='blue')
plt.plot(y_test.index, y_pred, label='Predicted Sales', color='red', linestyle='--')
plt.title('Actual vs Predicted Weekly Sales')
plt.xlabel('Index')
plt.ylabel('Weekly Sales')
plt.legend()
plt.grid(True)
plt.show()
