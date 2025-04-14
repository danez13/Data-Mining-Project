from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd

data = pd.read_csv("product_data_cleaned.csv")
X = data.drop(columns=["f_FPro_class"])
y = data["f_FPro_class"]

X_train_val, Xtest, y_train_val, ytest = train_test_split(X, y, test_size=0.1, random_state=11)

rf_params = {
    
}

rf = RandomForestClassifier(n_estimators=100, max_features='sqrt', random_state=42)
rf.fit(X, y)
