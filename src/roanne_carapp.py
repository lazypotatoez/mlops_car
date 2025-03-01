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

app = Flask(__name__, template_folder="templates")

# Ensure joblib does not cache to restricted directories
os.environ["JOBLIB_TEMP_FOLDER"] = "/tmp"

if not GlobalHydra.instance().is_initialized():
    hydra.initialize(config_path="../config", version_base=None)

cfg = hydra.compose(config_name="car")

@app.route('/')
def home():
    return render_template('roanne_car.html', predicted_price=None)

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

        # Prediction Logic
        base_value = 15.0  # Base value in lakhs
        year_factor = (df["Year"][0] - 2010) * 0.5
        mileage_discount = df["Kilometers_Driven"][0] / 10000 * 0.2
        prediction = base_value + year_factor - mileage_discount
        prediction = max(prediction, 1.0)

        # ✅ Return the prediction in the HTML page instead of JSON
        return render_template("roanne_car.html", predicted_price=round(prediction, 2))

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return render_template("roanne_car.html", predicted_price="Error occurred")



@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        df = pd.read_csv(file)

        if 'Price (INR Lakhs)' in df.columns:
            df = df.drop(columns=['Price (INR Lakhs)'])

        df["Predicted Price (INR Lakhs)"] = (
            15.0 + (df["Year"] - 2010) * 0.5 - (df["Kilometers_Driven"] / 10000 * 0.2)
        ).clip(lower=1.0)

        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)

        return send_file(io.BytesIO(output.getvalue().encode()), mimetype="text/csv", as_attachment=True, download_name="predictions.csv")

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=cfg.app.debug, host=cfg.app.host, port=cfg.app.port)
