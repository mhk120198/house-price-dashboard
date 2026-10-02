import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Page Title & Config
st.set_page_config(page_title="House Price Predictor", layout="wide")
st.title("🏡 House Price Prediction Dashboard")
st.write("An end-to-end Machine Learning web application.")

# Load Trained Model
model = joblib.load('models/house_price_model.joblib')

# Sidebar for User Inputs
st.sidebar.header("Filter House Parameters")
sqft = st.sidebar.slider("Square Feet", min_value=800, max_value=4000, value=1800, step=50)
bedrooms = st.sidebar.selectbox("Bedrooms", options=[1, 2, 3, 4, 5], index=2)
age = st.sidebar.slider("Property Age (Years)", min_value=0, max_value=30, value=5)

# Display Input Overview
st.subheader("Property Characteristics")
col1, col2, col3 = st.columns(3)
col1.metric("Area Size", f"{sqft} sq ft")
col2.metric("Bedrooms", bedrooms)
col3.metric("Building Age", f"{age} years")

# Predict Price
input_data = pd.DataFrame([[sqft, bedrooms, age]], columns=['SquareFeet', 'Bedrooms', 'Age'])
predicted_price = model.predict(input_data)[0]

st.markdown("---")
st.subheader("Estimated Market Value")
st.success(f"### Estimated Price: **${predicted_price:,.2f}**")

# Show Historical Data Reference
st.subheader("Historical Training Data")
df = pd.read_csv('data/housing.csv')
st.dataframe(df)
st.line_chart(df.set_index('SquareFeet')['Price'])