from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
import joblib
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pickle
import hydra
from omegaconf import DictConfig
from hydra.core.global_hydra import GlobalHydra

# Define Flask app at the module level
app = Flask(__name__, template_folder="../templates")

# Ensure joblib does not cache to restricted directories
os.environ["JOBLIB_TEMP_FOLDER"] = "/tmp"

# Initialize Hydra only if it's not already initialized
if not GlobalHydra.instance().is_initialized():
    hydra.initialize(config_path="../config")  # Ensure Hydra looks for car.yaml in config/

cfg = hydra.compose(config_name="car")

# Load model
try:
    model = joblib.load(cfg.model.path)
    print("Model loaded successfully")
except Exception as e:
    print(f"Model loading failed: {e}")
    try:
        with open(cfg.model.fallback_path, 'rb') as f:
            model = pickle.load(f)
        print("Model loaded with pickle successfully")
    except Exception as e2:
        print(f"Pickle loading failed: {e2}")
        model = joblib.load("artifacts/used_car_price_model.pkl")
        print("Model loaded from root directory")

# Create encoders for categorical variables
categorical_columns = cfg.encoding.categorical_columns
encoders = {col: OneHotEncoder(sparse=False, handle_unknown='ignore') for col in categorical_columns}

# Fit encoders with common values
for col in categorical_columns:
    sample_data = pd.DataFrame({col: cfg.encoding.common_values[col]})
    encoders[col].fit(sample_data)

@app.route('/')
def home():
    return render_template('roanne_car.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Gather user input
        user_input = {
            "Brand_Model": request.form['brand_model'],
            "Location": request.form['location'],
            "Year": int(request.form['year']),
            "Kilometers_Driven": float(request.form['kilometers_driven']),
            "Fuel_Type": request.form['fuel_type'],
            "Transmission": request.form['transmission'],
            "Owner_Type": request.form['owner_type'],
            "Mileage": float(request.form['mileage']),
            "Engine": float(request.form['engine']),
            "Power": float(request.form['power']),
            "Seats": int(request.form['seats'])
        }

        # Create a base DataFrame
        df = pd.DataFrame([user_input])

        # Encode categorical features
        encoded_features = {}
        for col in categorical_columns:
            cat_value = df[[col]]
            encoded = encoders[col].transform(cat_value)
            feature_names = encoders[col].get_feature_names_out([col]) if hasattr(encoders[col], 'get_feature_names_out') else [f"{col}_{val}" for val in encoders[col].categories_[0]]
            for i, name in enumerate(feature_names):
                encoded_features[name] = encoded[0][i]

        # Prepare numerical features
        numerical_features = {
            "Year": user_input["Year"],
            "Kilometers_Driven": user_input["Kilometers_Driven"],
            "Mileage": user_input["Mileage"],
            "Engine": user_input["Engine"],
            "Power": user_input["Power"],
            "Seats": user_input["Seats"]
        }

        # Simple prediction logic (as a fallback)
        base_value = 15.0  # Base value in lakhs
        year_factor = (user_input["Year"] - 2010) * 0.5
        mileage_discount = user_input["Kilometers_Driven"] / 10000 * 0.2
        prediction = base_value + year_factor - mileage_discount
        prediction = max(prediction, 1.0)  # Minimum price of 1 lakh

        return jsonify({"Predicted Price (INR Lakhs)": round(prediction, 2)})

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=cfg.app.debug, host=cfg.app.host, port=cfg.app.port)
