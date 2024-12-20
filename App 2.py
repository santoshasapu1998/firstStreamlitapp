import streamlit as st
import pandas as pd
import numpy as np
import joblib  # To load trained model
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import plotly.graph_objects as go

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

# Prepare for line chart predictions
price_gap_values = np.arange(-35, 36, 5)  # Define price_gap_40 range (-35 to 35 in steps of 5)
predictions = []

# Make predictions for each price gap value
for price_gap in price_gap_values:
    input_data['price_gap_40'] = scaler.transform(pd.DataFrame({'price_gap_40': [price_gap]}))['price_gap_40'].values
    market_share_prediction = model.predict(input_data)[0]
    predicted_market_share = np.exp(market_share_prediction)  # Reverse log transformation
    predictions.append(predicted_market_share)

# Create Plotly line chart with regions
fig = go.Figure()

# Plot the predicted market share
fig.add_trace(go.Scatter(x=price_gap_values, y=predictions, mode='lines', name='Market Share', line=dict(color='blue', width=2)))

# Add shaded regions for the price gap categories
fig.add_vrect(x0=-35, x1=0, fillcolor="green", opacity=0.2, line_width=0, annotation_text="Best", annotation_position="bottom center")
fig.add_vrect(x0=0, x1=10, fillcolor="yellow", opacity=0.2, line_width=0, annotation_text="Favorable", annotation_position="bottom center")
fig.add_vrect(x0=10, x1=20, fillcolor="orange", opacity=0.2, line_width=0, annotation_text="Competitive", annotation_position="bottom center")
fig.add_vrect(x0=20, x1=35, fillcolor="red", opacity=0.2, line_width=0, annotation_text="Challenged", annotation_position="bottom center")

# Customize chart layout
fig.update_layout(
    title="Predicted Market Share vs Price Gap 40",
    xaxis_title="Price Gap ($)",
    yaxis_title="Predicted Market Share",
    template="plotly_white",
    xaxis=dict(range=[-35, 35])
)

# Show chart in Streamlit
st.plotly_chart(fig)
