import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# ===============================
# Step 9D: CatBoost Regression
# ===============================

# ---- Load datasets (UNSCALED) ----
X_train = pd.read_csv("X_train.csv")
X_test  = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").values.ravel()
y_test  = pd.read_csv("y_test.csv").values.ravel()

# ---- Evaluation function (same for all models) ----
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

# ---- Initialize CatBoost ----
cat_model = CatBoostRegressor(
    iterations=800,
    depth=8,
    learning_rate=0.05,
    loss_function="RMSE",
    random_seed=42,
    verbose=False
)

# ---- Train ----
cat_model.fit(X_train, y_train)

# ---- Predict ----
y_pred = cat_model.predict(X_test)

# ---- Evaluate ----
metrics = evaluate_model(y_test, y_pred)

# ---- Save model ----
joblib.dump(cat_model, "catboost_model.pkl")

# ---- Print results ----
print("CatBoost Evaluation Results")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")
