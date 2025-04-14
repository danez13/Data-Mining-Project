import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
df = pd.read_csv('t11_cleaned_data.csv')

# Transform f_FPro_class into binary label
df['label'] = df['f_FPro_class'].apply(lambda x: 0 if x == 3 else 1)

# Drop irrelevant columns (optional for baseline)
drop_cols = ['original_ID', 'name', 'brand']  # these are unique per item
df = df.drop(columns=drop_cols, errors='ignore')

# Separate features and labels
X = df.drop(columns=['f_FPro_class', 'label'])
y = df['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=11, stratify=y
)

#  BASELINE: predict the most frequent class
most_common_class = y_train.mode()[0]
y_pred = np.full(shape=y_test.shape, fill_value=most_common_class)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, zero_division=0)

print(" BASELINE CLASSIFIER (Most Frequent Class)")
print(f"Most Frequent Class: {most_common_class}")
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(report)

# Optional: save results
baseline_results = {
    "TeamID": 11,
    "Most_Frequent_Class": most_common_class,
    "Accuracy": accuracy
}
pd.DataFrame([baseline_results]).to_csv("t11_baseline_results.csv", index=False)

print(df['label'].value_counts())


df = pd.read_csv("product_data.csv")
print(df['f_FPro_class'].value_counts())
