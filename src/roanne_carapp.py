from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
import joblib
from pycaret.regression import load_model

app = Flask(__name__, template_folder="../templates")

# Ensure joblib does not cache to restricted directories
os.environ["JOBLIB_TEMP_FOLDER"] = "/tmp"

# Load model - choose one method, not both
try:
    # First try with PyCaret's load_model
    model = load_model("artifacts/used_car_price_model", verbose=False)
    model.memory = None  # Prevent caching
except Exception as e:
    print(f"PyCaret load failed: {e}")
    # Fall back to joblib if PyCaret fails
    model = joblib.load("artifacts/used_car_price_model.joblib")
    model.memory = None  # Disable memory caching

@app.route('/')
def home():
    return render_template('roanne_car.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = {
            "Mileage": float(request.form['mileage']),
            "Year": int(request.form['year']),
            "Brand": request.form['brand'],
        }

        df = pd.DataFrame([features])

        # Make a prediction
        prediction = model.predict(df)[0]

        return jsonify({"Predicted Price (INR Lakhs)": round(prediction, 2)})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)