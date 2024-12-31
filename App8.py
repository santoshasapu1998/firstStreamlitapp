import streamlit as st
import pandas as pd
import numpy as np
import joblib  # To load trained model
import plotly.express as px  # For interactive plots

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

# Further filter for `price_gap_40` in the range -30 to +30
filtered_df = filtered_df[(filtered_df["price_gap_40"] >= -30) & (filtered_df["price_gap_40"] <= 30)]

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

    # Generate the chart with Plotly
    st.subheader(f"Predicted Market Share vs Price Gap for Filters: {geo_member_state} | {competitive_position}")

    # Sort by `price_gap_40` for better visualization
    filtered_df = filtered_df.sort_values(by="price_gap_40")

    # Create interactive plot using Plotly
    fig = px.line(
        filtered_df, 
        x="price_gap_40", 
        y="predicted_market_share",
        title="Predicted Market Share vs Price Gap",
        labels={"price_gap_40": "Price Gap ($)", "predicted_market_share": "Predicted Market Share"},
        markers=True,
        template="plotly_white",
        hover_data={"price_gap_40": True, "predicted_market_share": True}
    )

    # Highlight regions using vertical shading
    fig.add_vrect(x0=-30, x1=0, fillcolor="lightgreen", opacity=0.3, line_width=0, annotation_text="Best")
    fig.add_vrect(x0=0, x1=10, fillcolor="yellow", opacity=0.3, line_width=0, annotation_text="Favourable")
    fig.add_vrect(x0=10, x1=20, fillcolor="orange", opacity=0.3, line_width=0, annotation_text="Competitive")
    fig.add_vrect(x0=20, x1=30, fillcolor="red", opacity=0.3, line_width=0, annotation_text="Challenged")

    # Display the interactive chart
    st.plotly_chart(fig, use_container_width=True)

    # Show filtered data
    st.subheader("Filtered Data")
    st.write(filtered_df[["geo_member_state", "competitive_position", "price_gap_40", "predicted_market_share"]])

    # Explanation for regions
    st.subheader("Region Descriptions")
    st.markdown("""
    - **Best (-30 to 0):** Strong market position, favorable dynamics.
    - **Favorable (0 to 10):** Slight advantage, good potential.
    - **Competitive (10 to 20):** Average standing, competitive field.
    - **Challenged (20 to 30):** Significant challenges, weaker market position.
    """)
