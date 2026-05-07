import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# ===============================
# Step 9A: Random Forest Regression
# ===============================

# ---- Load datasets (UNSCALED) ----
X_train = pd.read_csv("X_train.csv")
X_test  = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").values.ravel()
y_test  = pd.read_csv("y_test.csv").values.ravel()

# ---- Evaluation function ----
def evaluate_model(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    smape = np.mean(
        2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))
    ) * 100
    r2 = r2_score(y_true, y_pred)
    mbe = np.mean(y_true - y_pred)

    return {
        "RMSE": rmse,
        "MAE": mae,
        "MAPE (%)": mape,
        "sMAPE (%)": smape,
        "R2": r2,
        "MBE": mbe
    }

# ---- Initialize Random Forest ----
rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

# ---- Train model ----
rf_model.fit(X_train, y_train)

# ---- Predict ----
y_pred = rf_model.predict(X_test)

# ---- Evaluate ----
metrics = evaluate_model(y_test, y_pred)

# ---- Save model ----
joblib.dump(rf_model, "random_forest_model.pkl")

# ---- Print results ----
print("Random Forest Evaluation Results")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")
