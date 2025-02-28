from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import os

app = Flask(__name__, template_folder="../templates")

# Load the trained model
model_path = "artifacts/used_car_price_model.pkl"

with open(model_path, "rb") as file:
    model = pickle.load(file)

# Define route for homepage
@app.route('/')
def home():
    return render_template('roanne_car.html')

# Define prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        features = {
            "Mileage": float(request.form['mileage']),
            "Year": int(request.form['year']),
            "Brand": request.form['brand'],  # Example categorical feature
            # Add more features here based on your dataset
        }

        # Convert data to a DataFrame (Ensure the order of columns matches training)
        df = pd.DataFrame([features])

        # Make a prediction
        prediction = model.predict(df)[0]

        return jsonify({"Predicted Price (INR Lakhs)": round(prediction, 2)})

    except Exception as e:
        return jsonify({"error": str(e)})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
