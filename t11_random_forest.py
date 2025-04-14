from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
import itertools

data = pd.read_csv("t11_cleaned_data.csv")
X = data.drop(columns=["label"])
y = data["label"]

X_train_val, Xtest, y_train_val, ytest = train_test_split(X, y, test_size=0.1, random_state=11)

rf_params = {
    "n_estimators": [10, 50, 100, 200],
    "criterion": ["gini", "entropy", "log_loss"],
    "max_depth": [None, 5, 10, 15],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 2, 4, 8],
    "min_weight_fraction_leaf": [0.0, 0.1, 0.2, 0.3],
    "max_features": [None, "sqrt", "log2", 1],
    "max_leaf_nodes": [None, 5, 10, 20],
}

keys, values = zip(*rf_params.items())
param_combos = list(itertools.product(*values))

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=11)
results = []
best_results = []
for combo in param_combos:
    params = dict(zip(keys, combo))
    print(params)
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
    # cross validation
    for train_index, val_index in cv.split(X_train_val, y_train_val):
        # Split the data into training and validation sets
        X_train, X_val = X_train_val.iloc[train_index], X_train_val.iloc[val_index]
        y_train, y_val = y_train_val.iloc[train_index], y_train_val.iloc[val_index]

        # Train the model
        model = RandomForestClassifier(**params, class_weight="balanced", random_state=11)
        model.fit(X_train, y_train)

        # Training metrics
        train_preds = model.predict(X_train)
        train_accuracies.append(accuracy_score(y_train, train_preds))
        train_precisions.append(precision_score(y_train, train_preds))
        train_recalls.append(recall_score(y_train, train_preds))
        train_f1.append(f1_score(y_train, train_preds))
        
        # Validation metrics        
        validation_preds = model.predict(X_val)
        validation_accuracy = accuracy_score(y_val, validation_preds)
        if best_current_model is None or validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            best_current_model = model
        validation_accuracies.append(validation_accuracy)
        validation_precisions.append(precision_score(y_val, validation_preds))
        validation_recalls.append(recall_score(y_val, validation_preds))
        validation_f1.append(f1_score(y_val, validation_preds))
    
    # Test metrics
    test_preds = best_current_model.predict(Xtest)
    test_accuracies.append(accuracy_score(ytest, test_preds))
    test_precisions.append(precision_score(ytest, test_preds))
    test_recalls.append(recall_score(ytest, test_preds))
    test_f1.append(f1_score(ytest, test_preds))

    # Average metrics across all folds
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

selection_results = pd.DataFrame(results)
selection_results.to_csv("t11_RFall.csv", index=False)

# Get the row with the highest validation accuracy
best_model = selection_results.loc[selection_results['validation accuracy'].idxmax()]
evaluation_results = best_model.to_frame().T
evaluation_results.drop(columns=["train accuracy", "train precision", "train recall", "train f1"], inplace=True)
evaluation_results.to_csv("t11_RFbest.csv", index=False)
