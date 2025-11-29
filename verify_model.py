import joblib
import pandas as pd

# --- Load the saved pipeline ---
MODEL_FILENAME = "rfmodel_v2.pkl"
pipeline = joblib.load(MODEL_FILENAME)
print(f"Pipeline loaded successfully: {type(pipeline)}")

# --- Create a small test input (same format as Streamlit inputs) ---
test_input = pd.DataFrame([{
    "Property_type_extracted": "Flat",
    "City": "Lekki",
    "Neighborhood": "Ajah",
    "Bedrooms": 3.0,
    "Bathrooms": 2.0,
    "Toilets": 2.0,
    "Newly Built": 1.0
}])

# --- Make a test prediction ---
prediction = pipeline.predict(test_input)[0]
print(f"Test prediction successful: ₦{prediction:,.0f}")
