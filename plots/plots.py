import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# ===============================
# CONFIGURATION
# ===============================

PLOTS_DIR = "plots"

MODELS = {
    "RandomForest": {
        "model_file": "random_forest_model.pkl",
        "scaled": False
    },
    "SVR": {
        "model_file": "svr_model.pkl",
        "scaled": True
    },
    "XGBoost": {
        "model_file": "xgboost_model.pkl",
        "scaled": False
    },
    "CatBoost": {
        "model_file": "catboost_model.pkl",
        "scaled": False
    }
}

# ===============================
# LOAD DATA
# ===============================

X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv").values.ravel()

X_test_scaled = pd.read_csv("X_test_scaled_svr.csv")

# ===============================
# CREATE PLOTS DIRECTORY STRUCTURE
# ===============================

os.makedirs(PLOTS_DIR, exist_ok=True)

for model_name in MODELS:
    os.makedirs(os.path.join(PLOTS_DIR, model_name), exist_ok=True)

# ===============================
# PLOTTING FUNCTION
# ===============================

def generate_plots(model_name, y_true, y_pred):
    model_dir = os.path.join(PLOTS_DIR, model_name)
    residuals = y_true - y_pred
    abs_error = np.abs(residuals)

    # -------- 1. Actual vs Predicted (Full) --------
    plt.figure()
    plt.plot(y_true, label="Actual")
    plt.plot(y_pred, label="Predicted")
    plt.xlabel("Time (Hours)")
    plt.ylabel("Load")
    plt.title(f"Actual vs Predicted Load – {model_name}")
    plt.legend()
    plt.savefig(f"{model_dir}/actual_vs_predicted_full.png")
    plt.close()

    # -------- 2. Actual vs Predicted (Zoomed) --------
    plt.figure()
    plt.plot(y_true[:300], label="Actual")
    plt.plot(y_pred[:300], label="Predicted")
    plt.xlabel("Time (Hours)")
    plt.ylabel("Load")
    plt.title(f"Actual vs Predicted (First 300 Hours) – {model_name}")
    plt.legend()
    plt.savefig(f"{model_dir}/actual_vs_predicted_zoomed.png")
    plt.close()

    # -------- 3. Residuals vs Time --------
    plt.figure()
    plt.plot(residuals)
    plt.xlabel("Time (Hours)")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.title(f"Residuals Over Time – {model_name}")
    plt.savefig(f"{model_dir}/residuals_vs_time.png")
    plt.close()

    # -------- 4. Residual Distribution --------
    plt.figure()
    plt.hist(residuals, bins=50)
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.title(f"Residual Distribution – {model_name}")
    plt.savefig(f"{model_dir}/residual_histogram.png")
    plt.close()

    # -------- 5. Absolute Error vs Time --------
    plt.figure()
    plt.plot(abs_error)
    plt.xlabel("Time (Hours)")
    plt.ylabel("Absolute Error")
    plt.title(f"Absolute Error Over Time – {model_name}")
    plt.savefig(f"{model_dir}/absolute_error_vs_time.png")
    plt.close()

    # -------- 6. Actual Load vs Error --------
    plt.figure()
    plt.scatter(y_true, residuals, alpha=0.3)
    plt.xlabel("Actual Load")
    plt.ylabel("Prediction Error")
    plt.title(f"Actual Load vs Error – {model_name}")
    plt.savefig(f"{model_dir}/actual_vs_error_scatter.png")
    plt.close()

# ===============================
# RUN FOR EACH MODEL
# ===============================

for model_name, config in MODELS.items():
    print(f"Generating plots for {model_name}...")

    model = joblib.load(config["model_file"])

    if config["scaled"]:
        y_pred = model.predict(X_test_scaled)
    else:
        y_pred = model.predict(X_test)

    generate_plots(model_name, y_test, y_pred)

print("\nAll plots generated successfully!")
print("Check the 'plots/' directory.")
