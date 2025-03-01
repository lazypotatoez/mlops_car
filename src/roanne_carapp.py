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
        # Create a dictionary with all required features
        features = {
            "Brand_Model": request.form['brand_model'],
            "Location": request.form['location'],
            "Kilometers_Driven": float(request.form['kilometers_driven']),
            "Fuel_Type": request.form['fuel_type'],
            "Transmission": request.form['transmission'],
            "Owner_Type": request.form['owner_type'],
            "Engine": float(request.form['engine']),
            "Power": float(request.form['power']),
            "Seats": int(request.form['seats']),
            # You can add Year if the model uses it
            "Year": int(request.form['year']) if 'year' in request.form else 0
        }

        df = pd.DataFrame([features])

        # Make a prediction
        prediction = model.predict(df)[0]

        return jsonify({"Predicted Price (INR Lakhs)": round(prediction, 2)})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)