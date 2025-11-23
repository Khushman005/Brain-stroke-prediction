# app.py
from flask import Flask, render_template, request
import pickle
import numpy as np
import csv

app = Flask(__name__)

# Load the model
model = pickle.load(open("model/stroke_model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect additional patient info
    patient_name = request.form['patient_name']
    father_name = request.form['father_name']
    occupation = request.form['occupation']
    full_address = request.form['full_address']

    # Collect input features
    input_features = [
        request.form['gender'],
        float(request.form['age']),
        int(request.form['hypertension']),
        int(request.form['heart_disease']),
        request.form['ever_married'],
        request.form['work_type'],
        request.form['residence_type'],
        float(request.form['avg_glucose_level']),
        float(request.form['bmi']),
        request.form['smoking_status']
    ]

    # Encoding same as training
    le_dict = {
        'gender': {'Male': 1, 'Female': 0, 'Other': 2},
        'ever_married': {'Yes': 1, 'No': 0},
        # LabelEncoder encodes categories in alphabetical order during training
        # ['Govt_job', 'Never_worked', 'Private', 'Self-employed', 'children'] -> [0,1,2,3,4]
        'work_type': {'Govt_job': 0, 'Never_worked': 1, 'Private': 2, 'Self-employed': 3, 'children': 4},
        'residence_type': {'Urban': 1, 'Rural': 0},
        # ['Unknown', 'formerly smoked', 'never smoked', 'smokes'] -> [0,1,2,3]
        'smoking_status': {'Unknown': 0, 'formerly smoked': 1, 'never smoked': 2, 'smokes': 3}
    }

    input_encoded = [
        le_dict['gender'][input_features[0]],
        input_features[1],
        input_features[2],
        input_features[3],
        le_dict['ever_married'][input_features[4]],
        le_dict['work_type'][input_features[5]],
        le_dict['residence_type'][input_features[6]],
        input_features[7],
        input_features[8],
        le_dict['smoking_status'][input_features[9]]
    ]

    proba = float(model.predict_proba([input_encoded])[0][1])
    threshold = 0.35  # tuneable threshold due to class imbalance
    prediction = 1 if proba >= threshold else 0
    result = (f"High Risk of Stroke ({proba*100:.1f}% chance)" if prediction == 1
              else f"Low Risk of Stroke ({proba*100:.1f}% chance)")

    # Save all info + encoded inputs + probability + decision to CSV
    with open('patient_predictions.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            patient_name, father_name, occupation, full_address,
            *input_encoded, f"{proba:.6f}", prediction, result
        ])

    return render_template("result.html",
                           prediction_result=result,
                           patient_name=patient_name,
                           father_name=father_name,
                           occupation=occupation,
                           full_address=full_address)


if __name__ == "__main__":
    app.run(debug=True)
