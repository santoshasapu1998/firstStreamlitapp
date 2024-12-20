import streamlit as st
import pandas as pd
import numpy as np
import joblib  # To load trained model
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load model and preprocessing objects
model = joblib.load("trained_model.pkl")
scaler = joblib.load("scaler.pkl")  # Save and load StandardScaler used for training
encoder = joblib.load("encoder.pkl")  # Save and load OneHotEncoder used for training

# Load the dataset to calculate min/max for sliders
df = pd.read_csv("training_data.csv")

# Initialize Streamlit app
st.title("Market Share Prediction")

# Create sliders for numeric features
st.subheader("Input Numeric Features")
numeric_features = ['carrier_cnt', 'price_gap_40', 'med_ded_std', 'med_moop_std']
numeric_input = {}
for feature in numeric_features:
    min_val = df[feature].min()
    max_val = df[feature].max()
    numeric_input[feature] = st.slider(
        f"{feature}", float(min_val), float(max_val), float((min_val + max_val) / 2)
    )

# Create dropdowns/sliders for categorical features
st.subheader("Input Categorical Features")
categorical_features = ['geo_member_state', 'geo_member_rating_area', 'Metal_level', 
                        'competitive_position', 'expansion_label', 'zero_ded_label']
categorical_input = {}
for feature in categorical_features:
    unique_values = df[feature].unique()
    categorical_input[feature] = st.selectbox(f"Select {feature}", unique_values)

# Process the input data
# Convert numeric inputs to DataFrame and scale
numeric_df = pd.DataFrame([numeric_input])
numeric_df = pd.DataFrame(scaler.transform(numeric_df), columns=numeric_features)

# One-hot encode categorical inputs
categorical_df = pd.DataFrame([categorical_input])
encoded_categorical_df = pd.DataFrame(
    encoder.transform(categorical_df).toarray(),
    columns=encoder.get_feature_names_out(categorical_features)
)

# Combine scaled numeric and encoded categorical data
input_data = pd.concat([numeric_df, encoded_categorical_df], axis=1)

# Make prediction
market_share_prediction = model.predict(input_data)[0]
predicted_market_share = np.exp(market_share_prediction)  # Reverse log transformation

# Display Prediction
st.subheader("Predicted Market Share")
st.write(f"Predicted Market Share: {predicted_market_share:.2f}")
