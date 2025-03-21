from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import os
import joblib
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pickle
import hydra
from omegaconf import DictConfig
from hydra.core.global_hydra import GlobalHydra
import io

app = Flask(__name__, template_folder="../templates")

# Ensure joblib does not cache to restricted directories
os.environ["JOBLIB_TEMP_FOLDER"] = "/tmp"

# Initialize Hydra only if it's not already initialized
if not GlobalHydra.instance().is_initialized():
    hydra.initialize(config_path="../config")

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

        df = pd.DataFrame([user_input])

        # Encode categorical features
        for col in categorical_columns:
            df[col] = encoders[col].transform(df[[col]])

        # Prediction Logic
        base_value = 15.0  # Base value in lakhs
        year_factor = (df["Year"][0] - 2010) * 0.5
        mileage_discount = df["Kilometers_Driven"][0] / 10000 * 0.2
        prediction = base_value + year_factor - mileage_discount
        prediction = max(prediction, 1.0)
        
        # For API requests that want JSON
        if request.headers.get('Accept') == 'application/json':
            return jsonify({"Predicted Price (INR Lakhs)": round(prediction, 2)})
        
        # For form submissions, render the template with the prediction
        return render_template('roanne_car.html', predicted_price=round(prediction, 2))

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        if request.headers.get('Accept') == 'application/json':
            return jsonify({"error": str(e)})
        return render_template('roanne_car.html', error=str(e))

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        # Read CSV file
        df = pd.read_csv(file)

        # Remove 'Price (INR Lakhs)' column if it exists
        if 'Price (INR Lakhs)' in df.columns:
            df = df.drop(columns=['Price (INR Lakhs)'])

        # Ensure required columns are present
        required_columns = ["Brand_Model", "Location", "Year", "Kilometers_Driven", "Fuel_Type", "Transmission", "Owner_Type", "Mileage", "Engine", "Power", "Seats"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return jsonify({"error": f"Missing columns: {missing_columns}"}), 400

        # Encode categorical features
        one_hot_encoded_cols = {}  # Dictionary to store column names before encoding
        for col in categorical_columns:
            if col in df.columns:
                transformed = encoders[col].transform(df[[col]])
                feature_names = encoders[col].get_feature_names_out([col])
                df = df.drop(columns=[col])  # Remove original column
                df = pd.concat([df, pd.DataFrame(transformed, columns=feature_names)], axis=1)
                one_hot_encoded_cols[col] = feature_names  # Store the mapping

        # Prediction Logic
        df["Predicted Price (INR Lakhs)"] = (
            15.0 + (df["Year"] - 2010) * 0.5 - (df["Kilometers_Driven"] / 10000 * 0.2)
        )
        df["Predicted Price (INR Lakhs)"] = df["Predicted Price (INR Lakhs)"].clip(lower=1.0)

        # Convert Encoded Columns Back to Original Categories
        for col, feature_names in one_hot_encoded_cols.items():
            try:
                df[col] = encoders[col].inverse_transform(df[feature_names])
                df.drop(columns=feature_names, inplace=True)  # Remove one-hot encoded columns
            except Exception as e:
                print(f"Error decoding {col}: {e}")  # Debugging message

        # Ensure CSV is Sent as a File (Not JSON)
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)

        return send_file(io.BytesIO(output.getvalue().encode()), 
                         mimetype="text/csv", 
                         as_attachment=True, 
                         download_name="predictions.csv")

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=cfg.app.debug, host=cfg.app.host, port=cfg.app.port)