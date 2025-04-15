# Import necessary libraries from scikit-learn and other packages
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
import itertools

# Load the cleaned dataset
data = pd.read_csv("t11_cleaned_data.csv")

# Separate features and target label
X = data.drop(columns=["label"])
y = data["label"]

# Split the data into training+validation and test sets (90% train_val, 10% test)
X_train_val, Xtest, y_train_val, ytest = train_test_split(X, y, test_size=0.1, random_state=11)

# Define hyperparameter grid for Random Forest
rf_params = {
    "n_estimators": [10, 50, 100, 200],  # number of trees
    "max_features": [1, 2, 4, 8]         # number of features to consider at each split
}

# Create all combinations of hyperparameters
keys, values = zip(*rf_params.items())
param_combos = list(itertools.product(*values))

# Set up cross-validation strategy
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=11)

results = []  # To store average metrics for each hyperparameter combination

# Iterate through each hyperparameter combination
for combo in param_combos:
    params = dict(zip(keys, combo))
    
    # Initialize lists to collect metrics across folds
    train_accuracies = []
    train_precisions = []
    train_recalls = []
    train_f1 = []
    validation_accuracies = []
    validation_precisions = []
    validation_recalls = []
    validation_f1 = []
    test_accuracies = []
    test_precisions = []
    test_recalls = []
    test_f1 = []
    
    best_current_model = None
    best_accuracy = 0

    # Cross-validation loop
    for train_index, val_index in cv.split(X_train_val, y_train_val):
        # Split training and validation data
        X_train, X_val = X_train_val.iloc[train_index], X_train_val.iloc[val_index]
        y_train, y_val = y_train_val.iloc[train_index], y_train_val.iloc[val_index]

        # Train Random Forest with current hyperparameters
        model = RandomForestClassifier(**params, class_weight="balanced", random_state=11)
        model.fit(X_train, y_train)

        # Evaluate on training set
        train_preds = model.predict(X_train)
        train_accuracies.append(accuracy_score(y_train, train_preds))
        train_precisions.append(precision_score(y_train, train_preds))
        train_recalls.append(recall_score(y_train, train_preds))
        train_f1.append(f1_score(y_train, train_preds))
        
        # Evaluate on validation set
        validation_preds = model.predict(X_val)
        validation_accuracy = accuracy_score(y_val, validation_preds)

        # Keep the best model (based on validation accuracy)
        if best_current_model is None or validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            best_current_model = model

        validation_accuracies.append(validation_accuracy)
        validation_precisions.append(precision_score(y_val, validation_preds))
        validation_recalls.append(recall_score(y_val, validation_preds))
        validation_f1.append(f1_score(y_val, validation_preds))
    
    # Evaluate the best model from cross-validation on the test set
    test_preds = best_current_model.predict(Xtest)
    test_accuracies.append(accuracy_score(ytest, test_preds))
    test_precisions.append(precision_score(ytest, test_preds))
    test_recalls.append(recall_score(ytest, test_preds))
    test_f1.append(f1_score(ytest, test_preds))

    # Store averaged metrics for this parameter combination
    results.append({
        **params,
        "train accuracy": np.mean(train_accuracies),
        "train precision": np.mean(train_precisions),
        "train recall": np.mean(train_recalls),
        "train f1": np.mean(train_f1),
        "validation accuracy": np.mean(validation_accuracies),
        "validation precision": np.mean(validation_precisions),
        "validation recall": np.mean(validation_recalls),
        "validation f1": np.mean(validation_f1),
        "test accuracy": np.mean(test_accuracies),
        "test precision": np.mean(test_precisions),
        "test recall": np.mean(test_recalls),
        "test f1": np.mean(test_f1),
    })

# Save the results from all hyperparameter combinations
selection_results = pd.DataFrame(results)
selection_results.to_csv("t11_RFall.csv", index=False)

# Select and save the best-performing model configuration (highest validation accuracy)
best_model = selection_results.loc[selection_results['validation accuracy'].idxmax()]
evaluation_results = best_model.to_frame().T
evaluation_results.drop(columns=["train accuracy", "train precision", "train recall", "train f1"], inplace=True)
evaluation_results.to_csv("t11_RFbest.csv", index=False)
