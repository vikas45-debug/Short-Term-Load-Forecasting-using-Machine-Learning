import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# ===============================
# Step 9C: XGBoost Regression
# ===============================

# ---- Load datasets (UNSCALED) ----
X_train = pd.read_csv("X_train.csv")
X_test  = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").values.ravel()
y_test  = pd.read_csv("y_test.csv").values.ravel()

# ---- Evaluation function (same as RF & SVR) ----
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

# ---- Initialize XGBoost ----
xgb_model = XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)

# ---- Train ----
xgb_model.fit(X_train, y_train)

# ---- Predict ----
y_pred = xgb_model.predict(X_test)

# ---- Evaluate ----
metrics = evaluate_model(y_test, y_pred)

# ---- Save model ----
joblib.dump(xgb_model, "xgboost_model.pkl")

# ---- Print results ----
print("XGBoost Evaluation Results")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")
