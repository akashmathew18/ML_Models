from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import numpy as np
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Load Models
try:
    slr_model = joblib.load("Models/SLR_Model.pkl")
    mlr_model = joblib.load("Models/MLR_Model.pkl")
except:
    slr_model = None
    mlr_model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/slr')
def slr_page():
    return render_template('slr.html')

@app.route('/mlr')
def mlr_page():
    return render_template('mlr.html')

# Train SLR and MLR Models
@app.route('/train', methods=['POST'])
def train_models():
    global slr_model, mlr_model
    
    # Load CSVs
    slr_data = pd.read_csv("data/slr_data.csv")
    mlr_data = pd.read_csv("data/mlr_data.csv")
    
    # Train SLR Model
    slr_model = LinearRegression()
    slr_model.fit(slr_data[['training_hours']], slr_data['performance_score'])
    joblib.dump(slr_model, "models/slr_model.pkl")
    
    # Train MLR Model
    mlr_model = LinearRegression()
    mlr_model.fit(mlr_data[['training_hours', 'fitness_level', 'past_performance']], mlr_data['performance_score'])
    joblib.dump(mlr_model, "models/mlr_model.pkl")
    
    return jsonify({"message": "Models trained successfully!"})

# Predict SLR
@app.route('/predict_slr', methods=['POST'])
def predict_slr():
    global slr_model
    if not slr_model:
        return jsonify({"error": "SLR model not trained yet!"})
    
    training_hours = float(request.form['training_hours'])
    prediction = slr_model.predict([[training_hours]])[0]
    
    return render_template('slr.html', prediction=prediction)

# Predict MLR
@app.route('/predict_mlr', methods=['POST'])
def predict_mlr():
    global mlr_model
    if not mlr_model:
        return jsonify({"error": "MLR model not trained yet!"})
    
    training_hours = float(request.form['training_hours'])
    fitness_level = float(request.form['fitness_level'])
    past_performance = float(request.form['past_performance'])
    
    prediction = mlr_model.predict([[training_hours, fitness_level, past_performance]])[0]
    
    return render_template('mlr.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
