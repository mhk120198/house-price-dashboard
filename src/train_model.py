import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

# 1. Load Data
df = pd.read_csv('data/housing.csv')

# 2. Split Features (X) and Target (y)
X = df[['SquareFeet', 'Bedrooms', 'Age']]
y = df['Price']

# 3. Train Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# 4. Save Model to models/ folder
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/house_price_model.joblib')

print("Model trained and saved to models/house_price_model.joblib successfully!")