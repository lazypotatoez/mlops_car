from flask import Flask, render_template, request, jsonify
import pandas as pd
from pycaret.regression import load_model

app = Flask(__name__, template_folder="../templates")

# Load the trained model properly using PyCaret
model = load_model("artifacts/used_car_price_model")  # Ensure the correct path

# Define route for homepage
@app.route('/')
def home():
    return render_template('roanne_car.html')

# Define prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form submission
        features = {
            "Mileage": float(request.form['mileage']),
            "Year": int(request.form['year']),
            "Brand": request.form['brand'],  # Example categorical feature
        }

        #  Convert data to a DataFrame (Ensure the order of columns matches training)
        df = pd.DataFrame([features])

        #  Make a prediction using PyCaret model
        prediction = model.predict(df)[0]

        return jsonify({"Predicted Price (INR Lakhs)": round(prediction, 2)})

    except Exception as e:
        return jsonify({"error": str(e)})

# Ensure app runs properly on Render deployment
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=10000, debug=True)
