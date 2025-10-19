from flask import Flask, render_template, request
from flask_cors import CORS, cross_origin
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
cors = CORS(app)

# Load model and dataset
model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))
car = pd.read_csv('Cleaned_Car_data.csv')


@app.route('/', methods=['GET', 'POST'])
def index():
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    years = sorted(car['year'].unique(), reverse=True)
    fuel_types = car['fuel_type'].unique()

    companies.insert(0, 'Select Company')
    return render_template('index.html', companies=companies, car_models=car_models, years=years, fuel_types=fuel_types)


@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    try:
        company = request.form.get('company')
        car_model = request.form.get('car_models')
        year = request.form.get('year')
        fuel_type = request.form.get('fuel_type')
        driven = request.form.get('kilo_driven')

        # Validate numeric inputs
        if not year.isdigit() or not driven.isdigit():
            return "Please enter valid numbers for Year and Kilometres Driven."

        year = int(year)
        driven = int(driven)

        # Prepare data for prediction
        data = pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                            data=[[car_model, company, year, driven, fuel_type]])

        # Make prediction
        prediction = model.predict(data)[0]

        # Clip negative predictions to 0
        prediction = max(0, prediction)

        # Format with ₹ and commas
        prediction_int = int(prediction)
        return str(prediction_int)

    except Exception as e:
        print("Prediction error:", e)
        return "Error in prediction. Please check your input values."


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
