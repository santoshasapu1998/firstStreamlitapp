import streamlit as st
import pandas as pd
import numpy as np
import joblib  # To load trained model
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import plotly.express as px

# Load model and preprocessing objects
best_model = joblib.load("trained_model.pkl")  # Replace model with best_model
sc = joblib.load("scaler.pkl")  # Replace scaler with sc
encoder = joblib.load("encoder.pkl")  # Save and load OneHotEncoder used for training

# Load the dataset to calculate min/max for sliders
df_final = pd.read_csv("training_data.csv")  # Replace df with df_final

# Initialize Streamlit app
st.title("Market Share Prediction")

# Create sliders for numeric features excluding `price_gap_40`
st.subheader("Input Numeric Features")
numeric_features = ['carrier_cnt', 'med_ded_std', 'med_moop_std']  # Excludes `price_gap_40`
numeric_input = {}
for feature in numeric_features:
    min_val = df_final[feature].min()  # Use df_final
    max_val = df_final[feature].max()  # Use df_final
    numeric_input[feature] = st.slider(
        f"{feature}", float(min_val), float(max_val), float((min_val + max_val) / 2)
    )

# Create dropdowns/sliders for categorical features
st.subheader("Input Categorical Features")
categorical_features = ['geo_member_state', 'geo_member_rating_area', 'Metal_level', 
                        'competitive_position', 'expansion_label', 'zero_ded_label']
categorical_input = {}
for feature in categorical_features:
    unique_values = df_final[feature].unique()  # Use df_final
    categorical_input[feature] = st.selectbox(f"Select {feature}", unique_values)

# Process the input data for predictions
# Create a copy of numeric and categorical input
base_numeric_input = numeric_input.copy()

# Add `price_gap_40` placeholder
numeric_features_with_price_gap = ['carrier_cnt', 'price_gap_40', 'med_ded_std', 'med_moop_std']
base_numeric_input['price_gap_40'] = 0  # Initial placeholder for price_gap_40

# Generate predictions for varying `price_gap_40` values
price_gap_values = list(range(-35, 36, 5))  # Generate range -35 to 35 with step of 5
predictions = []

for price_gap in price_gap_values:
    # Add the varying price_gap_40 value to input
    base_numeric_input['price_gap_40'] = price_gap
    
    # Scale numeric features including `price_gap_40`
    numeric_df = pd.DataFrame([base_numeric_input], columns=numeric_features_with_price_gap)  # Maintain order
    numeric_df = pd.DataFrame(
        sc.transform(numeric_df),  # Use sc instead of scaler
        columns=numeric_features_with_price_gap
    )
    
    # One-hot encode categorical features
    categorical_df = pd.DataFrame([categorical_input])
    encoded_categorical_df = pd.DataFrame(
        encoder.transform(categorical_df).toarray(),
        columns=encoder.get_feature_names_out(categorical_features)
    )
    
    # Combine numeric and categorical data
    input_data = pd.concat([numeric_df, encoded_categorical_df], axis=1)
    
    # Predict market share using best_model
    market_share_prediction = best_model.predict(input_data)[0]  # Use best_model
    predicted_market_share = np.exp(market_share_prediction)  # Reverse log transformation
    predictions.append(predicted_market_share)

# Create DataFrame for visualization
data = pd.DataFrame({"Price Gap ($)": price_gap_values, "Predicted Market Share": predictions})

# Plot using Plotly
fig = px.line(
    data,
    x="Price Gap ($)",
    y="Predicted Market Share",
    title="Predicted Market Share vs Price Gap",
    labels={"Price Gap ($)": "Price Gap ($)", "Predicted Market Share": "Market Share"},
    hover_data={"Price Gap ($)": True, "Predicted Market Share": ":.2f"},  # Add hover info
)

# Highlight different regions with color-coded ranges
fig.add_vrect(x0=-35, x1=0, fillcolor="lightgreen", opacity=0.2, line_width=0, annotation_text="Best")
fig.add_vrect(x0=0, x1=10, fillcolor="yellow", opacity=0.2, line_width=0, annotation_text="Favourable")
fig.add_vrect(x0=10, x1=20, fillcolor="orange", opacity=0.2, line_width=0, annotation_text="Competitive")
fig.add_vrect(x0=20, x1=35, fillcolor="red", opacity=0.2, line_width=0, annotation_text="Challenged")

# Display the chart
st.plotly_chart(fig)

# Explanation for regions
st.subheader("Region Descriptions")
st.markdown("""
- **Best (-35 to 0):** Strong market position, favorable dynamics.
- **Favorable (0 to 10):** Slight advantage, good potential.
- **Competitive (10 to 20):** Average standing, competitive field.
- **Challenged (20+):** Significant challenges, weaker market position.
""")
