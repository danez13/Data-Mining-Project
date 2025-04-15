import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Plot 1: Decision Tree Accuracy
data = pd.read_csv("t11_DTall.csv")

plt.figure(figsize=(10, 6))
plt.plot(data["min_samples_leaf"], data["train accuracy"], label='Training Accuracy', marker='o')
plt.plot(data["min_samples_leaf"], data["validation accuracy"], label='Validation Accuracy', marker='s')
plt.plot(data["min_samples_leaf"], data["test accuracy"], label='Test Accuracy', marker='^')
plt.title('Decision Tree Accuracy vs. min_samples_leaf')
plt.xlabel('min_samples_leaf')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Plot 2: Random Forest Test Accuracy
data = pd.read_csv("t11_RFall.csv")
# Get min and max max_features
min_feat = data["max_features"].min()
max_feat = data["max_features"].max()

# Filter for min and max max_features
df_min = data[data["max_features"] == min_feat]
df_max = data[data["max_features"] == max_feat]
plt.figure(figsize=(10, 6))
plt.plot(df_min["n_estimators"], df_min["test accuracy"], marker='o', label=f'max_features = {min_feat}')
plt.plot(df_max["n_estimators"], df_max["test accuracy"], marker='s', label=f'max_features = {max_feat}')
plt.title('Random Forest Test Accuracy vs n_estimators')
plt.xlabel('n_estimators')
plt.ylabel('Test Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Show both plots
plt.show()






