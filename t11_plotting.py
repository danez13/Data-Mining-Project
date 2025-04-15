import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# --- Load Dataset ---
df = pd.read_csv('product_data.csv')  # Make sure this file is in the same directory

# --- Identify Features and Target ---
target_column = 'f_FPro_class'  # Target column

if target_column not in df.columns:
    raise ValueError(f"Column '{target_column}' not found in dataset. Available columns: {df.columns.tolist()}")

# Drop columns that are not useful for modeling
drop_cols = ['original_ID', 'name', 'store', 'brand', 'food category']
df = df.drop(columns=drop_cols, errors='ignore')

# Separate features and target
X = df.drop(columns=[target_column])
y = df[target_column]

# Encode categorical target if necessary
if y.dtype == 'object':
    y = LabelEncoder().fit_transform(y)

# One-hot encode any remaining categorical features
X = pd.get_dummies(X, drop_first=True)

# --- Split the data ---
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

# ==============================
#  Plot 1: Decision Tree
# ==============================

min_samples_leaf_values = range(1, 21)
train_accuracies = []
val_accuracies = []
test_accuracies = []

for min_samples_leaf in min_samples_leaf_values:
    model = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf, random_state=42)
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_val, val_pred)
    test_acc = accuracy_score(y_test, test_pred)

    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)
    test_accuracies.append(test_acc)

# ==============================
#  Plot 2: Random Forest
# ==============================

n_estimators_list = range(10, 110, 10)
max_features_min = 1
max_features_max = 'sqrt'

rf_test_accuracies_min = []
rf_test_accuracies_max = []

for n in n_estimators_list:
    # Model with min max_features
    rf_min = RandomForestClassifier(n_estimators=n, max_features=max_features_min, random_state=42)
    rf_min.fit(X_train, y_train)
    pred_min = rf_min.predict(X_test)
    acc_min = accuracy_score(y_test, pred_min)
    rf_test_accuracies_min.append(acc_min)

    # Model with max max_features
    rf_max = RandomForestClassifier(n_estimators=n, max_features=max_features_max, random_state=42)
    rf_max.fit(X_train, y_train)
    pred_max = rf_max.predict(X_test)
    acc_max = accuracy_score(y_test, pred_max)
    rf_test_accuracies_max.append(acc_max)

# ==============================
# Show both plots
# ==============================

# Plot 1: Decision Tree Accuracy
plt.figure(figsize=(10, 6))
plt.plot(min_samples_leaf_values, train_accuracies, label='Training Accuracy', marker='o')
plt.plot(min_samples_leaf_values, val_accuracies, label='Validation Accuracy', marker='s')
plt.plot(min_samples_leaf_values, test_accuracies, label='Test Accuracy', marker='^')
plt.title('Decision Tree Accuracy vs. min_samples_leaf')
plt.xlabel('min_samples_leaf')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Plot 2: Random Forest Test Accuracy
plt.figure(figsize=(10, 6))
plt.plot(n_estimators_list, rf_test_accuracies_min, marker='o', label=f'max_features = {max_features_min}')
plt.plot(n_estimators_list, rf_test_accuracies_max, marker='s', label=f'max_features = {max_features_max}')
plt.title('Random Forest Test Accuracy vs n_estimators')
plt.xlabel('n_estimators')
plt.ylabel('Test Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Show both plots
plt.show()






