import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, precision_score, recall_score
import numpy as np
import matplotlib.pyplot as plt

# Load your dataset
diabetes = pd.read_csv("diabetes.csv")

# Extract features (X) and target variable (Y)
X = diabetes.drop('Outcome', axis=1)
Y = diabetes['Outcome']

# Create a RandomForestClassifier with your desired hyperparameters
clf = RandomForestClassifier(bootstrap=True, warm_start=True, oob_score=True, n_estimators=1000, max_features=3)

# Perform 3-fold cross-validation using StratifiedKFold
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

overall_cm = np.zeros((2, 2), dtype=int)

for fold, (train_index, test_index) in enumerate(skf.split(X, Y)):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

    # Fit the model on the training data
    clf.fit(X_train, Y_train)

    # Predict on the test data
    Y_pred = clf.predict(X_test)

    # Calculate confusion matrix for this fold
    cm = confusion_matrix(Y_test, Y_pred)

    overall_cm += cm

    # Calculate accuracy, precision, recall, and F1 score
    accuracy = accuracy_score(Y_test, Y_pred)
    precision = precision_score(Y_test, Y_pred)
    recall = recall_score(Y_test, Y_pred)
    f1 = f1_score(Y_test, Y_pred)

    # Print the results
    print(f"Fold {fold + 1} Results:")
    print("Confusion Matrix:")
    print(f"Actual (True) 0/1: {cm[0, 0]} / {cm[1, 1]}")
    print(f"Predicted 0/1: {cm[0, 0] + cm[0, 1]} / {cm[1, 0] + cm[1, 1]}")
    print(cm)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}\n")

# Print overall confusion matrix
print("Overall Confusion Matrix:")
print(overall_cm)

# Calculate overall accuracy, precision, recall, and F1 score
overall_accuracy = accuracy_score(Y, clf.predict(X))
overall_precision = precision_score(Y, clf.predict(X))
overall_recall = recall_score(Y, clf.predict(X))
overall_f1 = f1_score(Y, clf.predict(X))

# Print overall metrics
print(f"\nOverall Metrics:")
print(f"Overall Accuracy: {overall_accuracy:.4f}")
print(f"Overall Precision: {overall_precision:.4f}")
print(f"Overall Recall: {overall_recall:.4f}")
print(f"Overall F1 Score: {overall_f1:.4f}")

# Plot and save the overall confusion matrix as an image with numbers
plt.imshow(overall_cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Overall Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks(np.arange(2), ['0', '1'])
plt.yticks(np.arange(2), ['0', '1'])

# Add numbers to each block of the matrix
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(overall_cm[i, j]), ha='center', va='center', color='red', fontsize=12)

plt.savefig('confusion_matrix.png')
plt.show()





