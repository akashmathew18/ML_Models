from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Dictionary to store loaded models
models = {}

def load_models():
    """Load all models at startup"""
    try:
        # Load SLR and MLR models
        models['slr'] = joblib.load("Models/SLR_Model.pkl")
        models['mlr'] = joblib.load("Models/MLR_Model.pkl")
        print(" SLR and MLR models loaded successfully")
        
        try:
            # Load Logistic model
            with open("Models/Logistic_Model.pkl", "rb") as f:
                models['logistic'] = pickle.load(f)
            print(" Logistic Regression model loaded successfully")
        except Exception as e:
            print(f" Error loading Logistic Regression model: {str(e)}")
        
        try:
            # Load Polynomial model
            with open("Models/Polynomial_Model.pkl", "rb") as f:
                models['polynomial'] = pickle.load(f)
            print(" Polynomial model loaded successfully")
        except Exception as e:
            print(f" Error loading Polynomial model: {str(e)}")
        
        try:
            # Load KNN model and label encoder
            with open("Models/kNN_Model.pkl", "rb") as f:
                models['knn'] = pickle.load(f)
            with open("Models/label_encoder.pkl", "rb") as f:
                models['label_encoder'] = pickle.load(f)
            print(" KNN model and label encoder loaded successfully")
        except Exception as e:
            print(f" Error loading KNN model: {str(e)}")
        
        return True
    except Exception as e:
        print(f" Error in load_models: {str(e)}")
        return False

# Load models at startup
models_loaded = load_models()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/slr')
def slr_page():
    return render_template('slr.html')

@app.route('/mlr')
def mlr_page():
    return render_template('mlr.html')

@app.route('/pr')
def pr_page():
    return render_template('PR.html')

@app.route('/lr')
def lr_page():
    return render_template('LR.html')

@app.route('/knn')
def knn_page():
    return render_template('knn.html')

@app.route('/predict_slr', methods=['POST'])
def predict_slr():
    if not models.get('slr'):
        return render_template('slr.html', error="SLR model not loaded")
    
    try:
        training_hours = float(request.form['training_hours'])
        prediction = models['slr'].predict([[training_hours]])[0]
        return render_template('slr.html', prediction=f"Predicted Performance Score: {prediction:.2f}")
    except Exception as e:
        return render_template('slr.html', error=str(e))

@app.route('/predict_mlr', methods=['POST'])
def predict_mlr():
    if not models.get('mlr'):
        return render_template('mlr.html', error="MLR model not loaded")
    
    try:
        training_hours = float(request.form['training_hours'])
        fitness_level = float(request.form['fitness_level'])
        past_performance = float(request.form['past_performance'])
        prediction = models['mlr'].predict([[training_hours, fitness_level, past_performance]])[0]
        return render_template('mlr.html', prediction=f"Predicted Performance Score: {prediction:.2f}")
    except Exception as e:
        return render_template('mlr.html', error=str(e))

@app.route('/predict_logistic', methods=['POST'])
def predict_logistic():
    if not models.get('logistic'):
        print("Error: Logistic model not found in models dict")
        return jsonify({"error": "Logistic Regression model not loaded"})
    
    try:
        print("Received form data:", dict(request.form))
        
        # Extract features
        form_fields = ['matches_played', 'goals_scored', 'assists', 'performance_score', 
                      'fitness_score', 'training_attendance', 'coach_rating']
        
        # Check if all fields are present
        missing_fields = [field for field in form_fields if field not in request.form]
        if missing_fields:
            error_msg = f"Missing required fields: {', '.join(missing_fields)}"
            print("Error:", error_msg)
            return jsonify({"error": error_msg})
        
        # Create features array
        features = []
        for field in form_fields:
            try:
                value = float(request.form[field])
                features.append(value)
            except ValueError as e:
                error_msg = f"Invalid value for {field}: {request.form[field]}"
                print("Error:", error_msg)
                return jsonify({"error": error_msg})
        
        features = np.array([features])
        print("Features array shape:", features.shape)
        print("Features array:", features)
        
        # Make prediction
        prediction = models['logistic'].predict(features)[0]
        probability = models['logistic'].predict_proba(features)[0]
        confidence = probability[1] if prediction == 1 else probability[0]
        result = "Selected" if prediction == 1 else "Not Selected"
        
        print("Prediction successful:", result)
        return jsonify({
            "prediction": f"Player {result} (Confidence: {confidence:.2f})"
        })
    except Exception as e:
        print("Error in logistic prediction:", str(e))
        print("Error type:", type(e).__name__)
        import traceback
        print("Traceback:", traceback.format_exc())
        return jsonify({"error": f"An error occurred: {str(e)}"})

@app.route('/predict_polynomial', methods=['POST'])
def predict_polynomial():
    if not models.get('polynomial'):
        return render_template('PR.html', error="Polynomial model not loaded")
    
    try:
        features = np.array([[
            float(request.form['matches_played']),
            float(request.form['training_hours']),
            float(request.form['fitness_score']),
            float(request.form['coach_rating']),
            float(request.form['age'])
        ]])
        
        prediction = models['polynomial'].predict(features)[0]
        return render_template('PR.html', 
                             prediction=f"Predicted Performance Score: {prediction:.2f}")
    except Exception as e:
        return render_template('PR.html', error=str(e))

@app.route('/predict_knn', methods=['POST'])
def predict_knn():
    if not models.get('knn') or not models.get('label_encoder'):
        print("Error: KNN model or label encoder not found")
        return jsonify({"error": "KNN model or label encoder not loaded"})
    
    try:
        print("Received form data:", dict(request.form))
        
        # Extract features
        form_fields = ['height_cm', 'weight_kg', 'stamina', 'jumping', 'speed', 
                      'passing', 'shooting', 'defense', 'goalkeeping']
        
        # Check if all fields are present
        missing_fields = [field for field in form_fields if field not in request.form]
        if missing_fields:
            error_msg = f"Missing required fields: {', '.join(missing_fields)}"
            print("Error:", error_msg)
            return jsonify({"error": error_msg})
        
        # Create features array
        features = []
        for field in form_fields:
            try:
                value = float(request.form[field])
                features.append(value)
            except ValueError as e:
                error_msg = f"Invalid value for {field}: {request.form[field]}"
                print("Error:", error_msg)
                return jsonify({"error": error_msg})
        
        features = np.array([features])
        print("Features array shape:", features.shape)
        print("Features array:", features)
        
        # Make prediction
        prediction = models['knn'].predict(features)[0]
        position = models['label_encoder'].inverse_transform([prediction])[0]
        
        print("Prediction successful:", position)
        return jsonify({
            "prediction": f"Predicted Position: {position}"
        })
    except Exception as e:
        print("Error in KNN prediction:", str(e))
        print("Error type:", type(e).__name__)
        import traceback
        print("Traceback:", traceback.format_exc())
        return jsonify({"error": f"An error occurred: {str(e)}"})

if __name__ == '__main__':
    app.run(debug=True)