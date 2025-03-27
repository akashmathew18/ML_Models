import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier

# Train Multiple Linear Regression (MLR) Model
df_mlr = pd.read_csv("data/mlr_data.csv")
X_mlr = df_mlr.iloc[:, :-1]  # All columns except last as independent variables
Y_mlr = df_mlr.iloc[:, -1]   # Last column as the target variable

mlr_model = LinearRegression()
mlr_model.fit(X_mlr, Y_mlr)

with open("Models/MLR_Model.pkl", "wb") as file:
    pickle.dump(mlr_model, file)

print(" MLR Model trained and saved as MLR_Model.pkl!")

# Train Simple Linear Regression (SLR) Model
df_slr = pd.read_csv("data/slr_data.csv")
X_slr = df_slr.iloc[:, :-1]  # Only one independent variable (must be 2D)
Y_slr = df_slr.iloc[:, -1]   # Target variable

slr_model = LinearRegression()
slr_model.fit(X_slr, Y_slr)

with open("Models/SLR_Model.pkl", "wb") as file:
    pickle.dump(slr_model, file)

print(" SLR Model trained and saved as SLR_Model.pkl!")


#Train Polynomial Regression Model
data = pd.read_csv("data/PR_data.csv")

# Features and target
X = data[['Matches_Played', 'Training_Hours', 'Fitness_Score', 'Coach_Rating', 'Age']]
y = data['Performance_Score']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create Polynomial Regression model
degree = 2  # You can adjust this for better performance
model = make_pipeline(PolynomialFeatures(degree), LinearRegression())

# Train model
model.fit(X_train, y_train)

# Save model
with open("Models/Polynomial_Model.pkl", "wb") as f:
    pickle.dump(model, f)

print(" Polynomial Regression Model trained and saved as Polynomial_Model.pkl!")





# Train Logistic Regression Model
data = pd.read_csv("data/LR_data.csv")

# Features and target
X = data[['Matches_Played', 'Goals_Scored', 'Assists', 'Performance_Score', 'Fitness_Score', 'Training_Attendance', 'Coach_Rating']]
y = data['Player_Selected']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save model
with open("Models/Logistic_Model.pkl", "wb") as f:
    pickle.dump(model, f)

print(" Logistic Regression Model trained and saved as Logistic_Model.pkl!")


#Train kNN Model
data = pd.read_csv("data/kNN_data.csv")

# Features and target
X = data[['Height_cm', 'Weight_kg', 'Stamina_%', 'Jumping_Ability', 'Speed', 'Passing_Accuracy',
          'Shooting_Power', 'Defensive_Skills', 'Goalkeeping_Reflexes']]
y = data['Position']

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train kNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Save model and label encoder
with open("Models/kNN_Model.pkl", "wb") as f:
    pickle.dump(knn, f)

with open("Models/label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print(" kNN Model trained and saved as kNN_Model.pkl!")