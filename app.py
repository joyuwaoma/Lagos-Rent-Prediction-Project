import streamlit as st
import pandas as pd
from load_model import model  # import trained pipeline

# App title
st.title("🏠 Lagos Rent Prediction App")
st.divider()
st.write("Predict monthly house rent in Lagos based on property features.")
st.divider()

# Input options
property_options = ["Mini flat", "Flat", "Duplex", "Bungalow", "Other"]
location_options = ["Ikeja", "Lekki", "Yaba", "Ikoyi", "Ajah", "Surulere", "Maryland", "Oshodi"]

# User inputs
property_type = st.selectbox("Property Type", property_options)
bedrooms = st.number_input("Number of Bedrooms", min_value=0, value=1)
bathrooms = st.number_input("Number of Bathrooms", min_value=0, value=1)
toilets = st.number_input("Number of Toilets", min_value=0, value=1)
newly_built = st.radio("Newly Built?", ["Yes", "No"])
location = st.selectbox("City", location_options)
neighborhood = st.text_input("Neighborhood", "")

# Convert Newly Built to numeric
newly_built_numeric = 1 if newly_built == "Yes" else 0

# Prepare input DataFrame
X_input = pd.DataFrame([{
    "Property_type_extracted": property_type,
    "City": location,
    "Neighborhood": neighborhood,
    "Bedrooms": float(bedrooms),
    "Bathrooms": float(bathrooms),
    "Toilets": float(toilets),
    "Newly_Built": float(newly_built_numeric)
}])

st.divider()

# Predict
if st.button("Predict!"):
    try:
        prediction = model.predict(X_input)[0]
        st.success(f"Predicted Monthly Rent: ₦{prediction:,.0f}")
        st.balloons()
    except Exception as e:
        st.error(f"Prediction failed: {e}")
else:
    st.info("Enter all values and click Predict!")
