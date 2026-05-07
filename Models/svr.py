import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# ===============================
# Step 9B: Support Vector Regression
# ===============================

# ---- Load SCALED features (SVR only) ----
X_train = pd.read_csv("X_train_scaled_svr.csv")
X_test  = pd.read_csv("X_test_scaled_svr.csv")

y_train = pd.read_csv("y_train.csv").values.ravel()
y_test  = pd.read_csv("y_test.csv").values.ravel()

# ---- Evaluation function (same as RF) ----
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

# ---- Initialize SVR ----
svr_model = SVR(
    kernel="rbf",
    C=100,
    epsilon=0.1,
    gamma="scale"
)

# ---- Train ----
svr_model.fit(X_train, y_train)

# ---- Predict ----
y_pred = svr_model.predict(X_test)

# ---- Evaluate ----
metrics = evaluate_model(y_test, y_pred)

# ---- Save model ----
joblib.dump(svr_model, "svr_model.pkl")

# ---- Print results ----
print("SVR Evaluation Results")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")
