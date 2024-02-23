import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score, roc_curve, auc, f1_score

diabetes = pd.read_csv("diabetes.csv")
X = diabetes.drop('Outcome', axis=1)
Y = diabetes['Outcome']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)

# Define a range of values for n_estimators and max_features to try
n_estimators_range = [100, 500, 1000]  # Adjust these values as needed
max_features_range = [2,3,4,6]  # Adjust these values as needed

best_f1_score = 0.0
best_n_estimators = None
best_max_features = None

# Loop through different combinations of n_estimators and max_features
for n_estimators in n_estimators_range:
    for max_features in max_features_range:
        clf = RandomForestClassifier(bootstrap=True, warm_start=True, oob_score=True, n_estimators=n_estimators, max_features=max_features)
        
        # Fit the model
        clf.fit(X_train, Y_train)
        
        # Calculate OOB score
        oob_accuracy = clf.oob_score_
    
        avg_cv = cross_val_score(clf, X_train, Y_train, cv=3)
        
        Y_pred = clf.predict(X_test)
        
        f1 = f1_score(Y_test, Y_pred)
        
        print(f"n_estimators: {n_estimators}, max_features: {max_features}")
        print("OOB Score:", oob_accuracy)
        print("Avg. accuracy using 3-fold CV:", cv_scores.mean())
        print("Test Set F1 Score:", f1)
        print()
        
        # Check if this combination achieved a higher F1 score
        if f1 > best_f1_score:
            best_f1_score = f1
            best_n_estimators = n_estimators
            best_max_features = max_features

print("Best combination:")
print("n_estimators:", best_n_estimators)
print("max_features:", best_max_features)
print("Best F1 Score:", best_f1_score)






