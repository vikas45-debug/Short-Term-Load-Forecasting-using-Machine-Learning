# ==========================================
# EDA for Short-Term Load Forecasting (STLF)
# Input: Step 1 Processed Data
# Outputs: 8 EDA plots saved in same directory
# ==========================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf

sns.set(style="whitegrid")

# ------------------------------------------
# Load Step 1 processed dataset
# ------------------------------------------
input_path = "step1_datetime_processed.csv"
df = pd.read_csv(input_path, parse_dates=["time"])
df = df.sort_values("time").reset_index(drop=True)

# ------------------------------------------
# Output directory (same as Step 1 file)
# ------------------------------------------
output_dir = os.path.dirname(input_path)
if output_dir == "":
    output_dir = "."

# ------------------------------------------
# Basic time features for EDA ONLY
# ------------------------------------------
df["hour"] = df["time"].dt.hour
df["day_of_week"] = df["time"].dt.dayofweek

# ==========================================
# PLOT 1: Load vs Time
# ==========================================
plt.figure(figsize=(12, 4))
plt.plot(df["time"], df["load"])
plt.title("Load vs Time")
plt.xlabel("Time")
plt.ylabel("Load")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "eda_1_load_vs_time.png"), dpi=300)
plt.show()

# ==========================================
# PLOT 2: Load Distribution
# ==========================================
plt.figure(figsize=(6, 4))
sns.histplot(df["load"], bins=50, kde=True)
plt.title("Distribution of Load")
plt.xlabel("Load")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "eda_2_load_distribution.png"), dpi=300)
plt.show()

# ==========================================
# PLOT 3: Average Load by Hour of Day
# ==========================================
hourly_avg = df.groupby("hour")["load"].mean()

plt.figure(figsize=(7, 4))
hourly_avg.plot(marker="o")
plt.title("Average Load by Hour of Day")
plt.xlabel("Hour")
plt.ylabel("Average Load")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "eda_3_avg_load_by_hour.png"), dpi=300)
plt.show()

# ==========================================
# PLOT 4: Average Load by Day of Week
# ==========================================
dow_avg = df.groupby("day_of_week")["load"].mean()

plt.figure(figsize=(7, 4))
dow_avg.plot(kind="bar")
plt.title("Average Load by Day of Week")
plt.xlabel("Day (0 = Monday, 6 = Sunday)")
plt.ylabel("Average Load")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "eda_4_avg_load_by_dow.png"), dpi=300)
plt.show()

# ==========================================
# PLOT 5: Load vs Temperature
# ==========================================
plt.figure(figsize=(6, 4))
sns.scatterplot(
    x=df["temperature"],
    y=df["load"],
    alpha=0.4
)
plt.title("Load vs Temperature")
plt.xlabel("Temperature")
plt.ylabel("Load")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "eda_5_load_vs_temperature.png"), dpi=300)
plt.show()

# ==========================================
# PLOT 6: Load vs Previous Hour Load (Lag-1)
# ==========================================
df["load_lag_1"] = df["load"].shift(1)

plt.figure(figsize=(6, 4))
sns.scatterplot(
    x=df["load_lag_1"],
    y=df["load"],
    alpha=0.4
)
plt.title("Current Load vs Previous Hour Load")
plt.xlabel("Load at t-1")
plt.ylabel("Load at t")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "eda_6_load_vs_lag1.png"), dpi=300)
plt.show()

# ==========================================
# PLOT 7: Autocorrelation Function (ACF)
# ==========================================
plt.figure(figsize=(8, 4))
plot_acf(df["load"].dropna(), lags=48)
plt.title("Autocorrelation of Load")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "eda_7_acf.png"), dpi=300)
plt.show()

# ==========================================
# PLOT 8: Correlation Heatmap
# ==========================================
corr_features = df[["load", "temperature", "hour", "day_of_week"]]

plt.figure(figsize=(6, 5))
sns.heatmap(
    corr_features.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Correlation Heatmap of Key Variables")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "eda_8_correlation_heatmap.png"), dpi=300)
plt.show()

print("EDA completed successfully. All plots saved in Step 1 directory.")
