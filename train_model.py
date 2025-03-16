import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

# Train Multiple Linear Regression (MLR) Model
df_mlr = pd.read_csv("data/mlr_data.csv")
X_mlr = df_mlr.iloc[:, :-1]  # All columns except last as independent variables
Y_mlr = df_mlr.iloc[:, -1]   # Last column as the target variable

mlr_model = LinearRegression()
mlr_model.fit(X_mlr, Y_mlr)

with open("Models/MLR_Model.pkl", "wb") as file:
    pickle.dump(mlr_model, file)

print("✅ MLR Model trained and saved as MLR_Model.pkl!")

# Train Simple Linear Regression (SLR) Model
df_slr = pd.read_csv("data/slr_data.csv")
X_slr = df_slr.iloc[:, :-1]  # Only one independent variable (must be 2D)
Y_slr = df_slr.iloc[:, -1]   # Target variable

slr_model = LinearRegression()
slr_model.fit(X_slr, Y_slr)

with open("Models/SLR_Model.pkl", "wb") as file:
    pickle.dump(slr_model, file)

print("✅ SLR Model trained and saved as SLR_Model.pkl!")
