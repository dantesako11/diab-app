import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load the model and scaler
with open("diabetes_model2.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("transf1.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Home route to render HTML form
@app.route("/")
def home():
    return render_template("index.html")

# Predict route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form data
        data = request.form
        required_columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
                            "BMI", "DiabetesPedigreeFunction", "Age"]

        # Convert form data to DataFrame
        input_data = pd.DataFrame([{key: float(data[key]) for key in required_columns}])

        # Scale and predict
        scaled_data = scaler.transform(input_data)
        prediction = model.predict(scaled_data)

        result = "Diabetes" if prediction[0] == 1 else "No Diabetes"
        return render_template("index.html", prediction=result)

    except Exception as e:
        return render_template("index.html", prediction=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
