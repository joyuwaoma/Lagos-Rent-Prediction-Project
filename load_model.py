import os
import pickle
import joblib

# Get base directory safely
try:
    BASE_DIR = os.path.dirname(__file__)
except NameError:
    # Fallback if __file__ is not defined (e.g., interactive shell, Streamlit)
    BASE_DIR = os.getcwd()

MODEL_FILENAME = "rfmodel.pkl"
MODEL_PATH = os.path.join(BASE_DIR, MODEL_FILENAME)

# Load trained model
try:
    # Try joblib first
    try:
        model = joblib.load(MODEL_PATH)
        print(f"Model loaded successfully with joblib from {MODEL_PATH}")
    except Exception:
        # Fallback to pickle if joblib fails
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        print(f"Model loaded successfully with pickle from {MODEL_PATH}")

except Exception as e:
    raise RuntimeError(f"Failed to load model from {MODEL_PATH}: {e}")
