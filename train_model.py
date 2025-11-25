import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
import joblib

# Load dataset
df = pd.read_csv('/Users/mac/Downloads/lagos-rent.csv')
df.columns = df.columns.str.strip()

# Extract property type
def extract_property_type(x):
    x = str(x).lower()
    if "mini" in x:
        return "Mini flat"
    elif "flat" in x:
        return "Flat"
    elif "duplex" in x:
        return "Duplex"
    elif "bungalow" in x:
        return "Bungalow"
    else:
        return "Other"

df['Property_type_extracted'] = df['Title'].apply(extract_property_type)

# Clean numeric columns
def extract_number(x):
    try:
        num = pd.Series(str(x)).str.extract(r"(\d+)")[0]
        return float(num.iloc[0])
    except:
        return 0.0

df["Bedrooms"] = df["Bedrooms"].apply(extract_number)
df["Bathrooms"] = df["Bathrooms"].apply(extract_number)
df["Toilets"] = df["Toilets"].apply(extract_number)

# Newly Built → 1/0
df["Newly_Built"] = df["Newly Built"].astype(str).str.lower().map({"yes": 1, "no": 0}).fillna(0)

# Clean Price
df['Price'] = df['Price'].astype(str).str.replace(',', '').str.extract(r'(\d+)')[0].astype(float)

# Features & target
categorical_cols = ['City', 'Neighborhood', 'Property_type_extracted']
numeric_cols = ['Bedrooms', 'Bathrooms', 'Toilets', 'Newly_Built']
X = df[categorical_cols + numeric_cols]
y = df["Price"]

# Preprocessing pipeline
numeric_transformer = Pipeline([('imputer', SimpleImputer(strategy='median'))])
categorical_transformer = Pipeline([('encoder', OneHotEncoder(handle_unknown='ignore'))])
preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_cols),
    ('cat', categorical_transformer, categorical_cols)
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
print(f"R²: {r2_score(y_test, y_pred):.3f}, RMSE: {mean_squared_error(y_test, y_pred, squared=False):.2f}")

# Save pipeline
joblib.dump(pipeline, "rfmodel.pkl")
print("Pipeline trained and saved!")