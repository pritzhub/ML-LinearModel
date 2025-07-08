import pandas as pd
import streamlit as st
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import os

def run_model():
    st.title("Car Price Prediction App")

    Year = st.number_input("Manufacture Year", min_value=1990, max_value=2025, value=2015)
    Kilometer = st.number_input("Kilometers Driven", value=50000)
    FuelType = st.selectbox("Fuel Type", ['Diesel', 'Petrol', 'LPG', 'Electric', 'CNG'])
    Transmission = st.selectbox("Transmission", ['Manual', 'Automatic'])
    Owner = st.selectbox("Owner", ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'])
    SellerType = st.selectbox("Seller Type", ['Dealer', 'Individual', 'Trustmark Dealer'])

    input_dict = {
        'Year': Year,
        'Kilometer': Kilometer
    }

    for col in model_columns:
        if col.startswith('FuelType_') and f'FuelType_{FuelType}' == col:
            input_dict[col] = 1
        elif col.startswith('Transmission_') and f'Transmission_{Transmission}' == col:
            input_dict[col] = 1
        elif col.startswith('Owner_') and f'Owner_{Owner}' == col:
            input_dict[col] = 1
        elif col.startswith('SellerType_') and f'SellerType_{SellerType}' == col:
            input_dict[col] = 1

    input_df = pd.DataFrame([input_dict])
    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[model_columns]

    if st.button("Predict Price"):
        prediction = model.predict(input_df)[0]
        st.success(f"Estimated Car Price: ₹ {int(prediction):,}")
        
if __name__ == "__main__":
    if not os.path.exists("model/car_price_model.pkl") or not os.path.exists("model/model_columns.pkl"):
        st.write("Model does not exists")
    else:
        model = joblib.load("model/car_price_model.pkl")
        model_columns = joblib.load("model/model_columns.pkl")
        run_model()