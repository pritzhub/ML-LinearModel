import pandas as pd
import streamlit as st
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import os
#import time

#@st.cache_data
def train_model():
    df = pd.read_csv("data/car_details.csv")
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    df = df[['Year', 'Price', 'Kilometer', 'FuelType', 'SellerType', 'Transmission', 'Owner']]
    df = pd.get_dummies(df, drop_first=True)

    X = df.drop('Price', axis=1)
    y = df['Price']
    st.write("Training the model")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    st.write("Saving the model")
    current_directory = os.getcwd()
    joblib.dump(model, "model/car_price_model.pkl")
    joblib.dump(X.columns.tolist(), "model/model_columns.pkl")
    st.write("Model saved, please verify")

    return model, X.columns.tolist()

def get_yes_no_input(prompt_message):
    """
    Prompts the user with a yes/no question and returns True for 'yes' or False for 'no'.
    Handles case-insensitivity and invalid input by re-prompting.
    """
    
    while True:
        user_response = input(f"{prompt_message} (yes/no): ").strip().lower()
        if user_response in ['yes', 'y']:
            return True
        elif user_response in ['no', 'n']:
            return False
        else:
            st.write("Invalid input. Please enter 'yes' or 'no'.")
            
if __name__ == "__main__":
    if not os.path.exists("model/car_price_model.pkl") or not os.path.exists("model/model_columns.pkl"):
        st.write("Model doesn't exist in the folder, creating and training the model")
        model, model_columns = train_model()
        st.write("Model creation completed, please verify")
    else:
        if get_yes_no_input("Model Exists, do you want to recreate and retrain the model?"):
            file_path = "model/car_price_model.pkl" # Replace with the actual file path
            if os.path.exists(file_path):
                os.remove(file_path)
            file_path = "model/model_columns.pkl" # Replace with the actual file path
            if os.path.exists(file_path):
                os.remove(file_path)
                
            st.write("Existing model have been removed. Recreating ...")
            #time.sleep(5)
            model, model_columns = train_model()
            st.write("Model creation completed, please verify")
        else:
            # Load the model
            model = joblib.load("model/car_price_model.pkl")
            model_columns = joblib.load("model/model_columns.pkl")
            st.write("Loaded the model for the further processing..")
            
