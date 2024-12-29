import streamlit as st
import pandas as pd
import numpy as np
import joblib  # To load trained model
import matplotlib.pyplot as plt

# Load model and preprocessing objects
best_model = joblib.load("trained_model.pkl")  # Use best_model
sc = joblib.load("scaler.pkl")  # Use sc
encoder = joblib.load("encoder.pkl")  # Use encoder

# Load test dataset
df_final = pd.read_csv("X_test.csv")  # Ensure your test data is saved as `X_test.csv`

# Initialize Streamlit app
st.title("Market Share Prediction: Filtered Analysis")

# User input for filtering test data
st.subheader("Filter Test Data")

# Add "All" option for `geo_member_state` and `competitive_position`
geo_member_states = ["All"] + df_final["geo_member_state"].unique().tolist()
competitive_positions = ["All"] + df_final["competitive_position"].unique().tolist()

geo_member_state = st.selectbox("Select Geo Member State", geo_member_states)
competitive_position = st.selectbox("Select Competitive Position", competitive_positions)

# Apply filters
filtered_df = df_final.copy()

if geo_member_state != "All":
    filtered_df = filtered_df[filtered_df["geo_member_state"] == geo_member_state]

if competitive_position != "All":
    filtered_df = filtered_df[filtered_df["competitive_position"] == competitive_position]

if filtered_df.empty:
    st.warning("No matching data for the selected filters. Try different values.")
else:
    # Predict on the filtered data
    numeric_features = ['carrier_cnt', 'price_gap_40', 'med_ded_std', 'med_moop_std']
    categorical_features = ['geo_member_state', 'geo_member_rating_area', 'Metal_level', 
                            'competitive_position', 'expansion_label', 'zero_ded_label']

    # Split filtered data into numeric and categorical parts
    numeric_df = filtered_df[numeric_features].copy()
    categorical_df = filtered_df[categorical_features].copy()

    # Scale numeric features
    numeric_scaled = pd.DataFrame(sc.transform(numeric_df), columns=numeric_features)

    # One-hot encode categorical features
    encoded_categorical_df = pd.DataFrame(
        encoder.transform(categorical_df).toarray(),
        columns=encoder.get_feature_names_out(categorical_features)
    )

    # Combine scaled numeric and encoded categorical data
    input_data = pd.concat([numeric_scaled, encoded_categorical_df], axis=1)

    # Predict market share
    market_share_predictions = best_model.predict(input_data)
    filtered_df["predicted_market_share"] = np.exp(market_share_predictions)  # Reverse log transformation

    # Generate the chart
    st.subheader(f"Predicted Market Share vs Price Gap for Filters: {geo_member_state} | {competitive_position}")
    fig, ax = plt.subplots(figsize=(10, 6))

    # Sort by `price_gap_40` for better visualization
    filtered_df = filtered_df.sort_values(by="price_gap_40")

    # Plot predictions
    ax.plot(filtered_df["price_gap_40"], filtered_df["predicted_market_share"], 
            label="Predicted Market Share", color="black", linewidth=2)

    # Highlight regions with different colors
    ax.axvspan(-35, 0, color="lightgreen", alpha=0.5, label="Best")
    ax.axvspan(0, 10, color="yellow", alpha=0.5, label="Favourable")
    ax.axvspan(10, 20, color="orange", alpha=0.5, label="Competitive")
    ax.axvspan(20, 35, color="red", alpha=0.5, label="Challenged")

    # Add labels and title
    ax.set_xlabel("Price Gap ($)", fontsize=12)
    ax.set_ylabel("Predicted Market Share", fontsize=12)
    ax.set_title("Market Share vs Price Gap", fontsize=14)
    ax.legend(title="Regions")
    st.pyplot(fig)

    # Show filtered data
    st.subheader("Filtered Data")
    st.write(filtered_df[["geo_member_state", "competitive_position", "price_gap_40", "predicted_market_share"]])

    # Explanation for regions
    st.subheader("Region Descriptions")
    st.markdown("""
    - **Best (-35 to 0):** Strong market position, favorable dynamics.
    - **Favorable (0 to 10):** Slight advantage, good potential.
    - **Competitive (10 to 20):** Average standing, competitive field.
    - **Challenged (20+):** Significant challenges, weaker market position.
    """)
