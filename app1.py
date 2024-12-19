import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib  # For loading the trained model

# Load trained model
model = joblib.load('trained_model.pkl')  # Replace with your model file

# Sample data with many counties
data = {
    'county': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],  # Add more counties
    'price_gap': np.random.uniform(1.0, 5.0, 100),
    'market_size': np.random.uniform(100, 500, 100),
    'actual_market_share': np.random.uniform(10, 50, 100),
}
df = pd.DataFrame(data)

# Streamlit app
st.title("Market Share Prediction and Visualization")

# Sidebar for county search
selected_county = st.selectbox(
    "Search for a County", 
    options=sorted(df['county'].unique()),  # Make it searchable
    key="county_select"
)

# Filter data by selected county
filtered_data = df[df['county'] == selected_county]

# Sidebar for price_gap adjustment
price_gap_range = st.slider("Select Price Gap Range", 
                            min_value=float(df['price_gap'].min()), 
                            max_value=float(df['price_gap'].max()), 
                            value=(float(df['price_gap'].min()), float(df['price_gap'].max())))
filtered_data = filtered_data[
    (filtered_data['price_gap'] >= price_gap_range[0]) &
    (filtered_data['price_gap'] <= price_gap_range[1])
]

# Make predictions using the model
if 'market_size' in filtered_data.columns:  # Ensure feature exists
    X = filtered_data[['price_gap', 'market_size']]
    filtered_data['predicted_market_share'] = model.predict(X)

# Display filtered data
st.write(f"Data for County: {selected_county}")
st.write(filtered_data)

# Visualization
chart = alt.Chart(filtered_data).mark_line().encode(
    x='price_gap',
    y=alt.Y('actual_market_share', title='Market Share'),
    color=alt.value('blue'),
    tooltip=['price_gap', 'actual_market_share']
).properties(
    title=f"Actual Market Share (County: {selected_county})"
)

pred_chart = alt.Chart(filtered_data).mark_line(strokeDash=[5, 5]).encode(
    x='price_gap',
    y=alt.Y('predicted_market_share', title='Market Share'),
    color=alt.value('red'),
    tooltip=['price_gap', 'predicted_market_share']
).properties(
    title=f"Predicted Market Share (County: {selected_county})"
)

st.altair_chart(chart + pred_chart, use_container_width=True)
