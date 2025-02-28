from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__, template_folder="../templates")

# Load model using joblib instead of PyCaret's load_model
model = joblib.load("artifacts/used_car_price_model.joblib")

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
