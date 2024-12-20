import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib  # To load your trained model

# Load your trained model
model = joblib.load("path_to_your_trained_model.pkl")

# Define the input features for the model
input_features = ['fips_code', 'price_gap_40']  # Extend as per your model

# Streamlit App
st.title("Market Share Prediction App")

# Sidebar Inputs
fips_code = st.number_input("Enter FIPS Code (numeric value):", min_value=0.0, step=1.0, value=1.0)
price_range = st.slider(
    "Select Price Gap 40 Range",
    -35.0, 35.0, (-35.0, 35.0), step=5.0
)

# Generate a range of `price_gap_40` values within the selected range with step=5
price_gap_values = np.arange(price_range[0], price_range[1] + 5, 5)

# Prepare data for prediction
input_data = pd.DataFrame({'fips_code': [fips_code] * len(price_gap_values),
                           'price_gap_40': price_gap_values})

# Predict market share
market_share_predictions = model.predict(input_data)

# Visualization
st.subheader(f"Predicted Market Share vs Price Gap 40 for FIPS Code: {fips_code}")

# Define regions and their colors
regions = {
    'Best': [-35, 0, 'green'],
    'Favorable': [0, 10, 'blue'],
    'Competitive': [10, 20, 'orange'],
    'Challenged': [20, 35, 'red']
}

# Create the plot
plt.figure(figsize=(12, 8))

# Add different regions
for region_name, (start, end, color) in regions.items():
    region_mask = (price_gap_values >= start) & (price_gap_values <= end)
    plt.plot(price_gap_values[region_mask],
             market_share_predictions[region_mask],
             label=f"{region_name} ({start} to {end})",
             color=color)

# Add labels and grid
plt.xlabel('Price Gap 40')
plt.ylabel('Market Share')
plt.title('Predicted Market Share vs Price Gap 40')
plt.legend()
plt.grid(True)

# Show the chart in Streamlit
st.pyplot(plt)

# Display Region Descriptions
st.markdown("### Regions Description")
st.markdown("""
- **Best (-35 to 0):** Indicated in Green.
- **Favorable (0 to 10):** Indicated in Blue.
- **Competitive (10 to 20):** Indicated in Orange.
- **Challenged (20 to 35):** Indicated in Red.
""")
