import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, precision_score, recall_score
import numpy as np
import matplotlib.pyplot as plt

# Load your dataset
diabetes = pd.read_csv("diabetes.csv")

X = diabetes.drop('Outcome', axis=1)
Y = diabetes['Outcome']

# Create a RandomForestClassifier with your desired hyperparameters
clf = RandomForestClassifier(bootstrap=True, warm_start=True, oob_score=True, n_estimators=1000, max_features=3)
clf.fit(X, Y)

# Get 1 positive and 1 negative sample from the training data
positive_sample = X[Y == 1].sample(1, replace=False).values
negative_sample = X[Y == 0].sample(1, replace=False).values

# Predict the class and probability of the positive sample
positive_prediction = clf.predict(positive_sample)
positive_probabilities = clf.predict_proba(positive_sample)

# Predict the class and probability of the negative sample
negative_prediction = clf.predict(negative_sample)
negative_probabilities = clf.predict_proba(negative_sample)

print("Classification results:")
print("Predicted label for Positive sample:", positive_prediction[0])
print("Predicted label for Negative sample:", negative_prediction[0])

print("\nActual label for Positive sample: 1")
print("Actual label for Negative sample: 0\n")

# Print the probabilities for the positive and negative samples
print("Probability for Positive sample (Class 0, Class 1):", positive_probabilities[0])
print("Probability for Negative sample (Class 0, Class 1):", negative_probabilities[0])

# Compare the predictions to the actual labels
if positive_prediction == 1 and negative_prediction == 0:
    print("\nAccurate predictions.")
else:
    print("\nIncorrect predictions.")
