import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_curve, f1_score, auc, roc_curve
from sklearn.inspection import permutation_importance

diabetes = pd.read_csv("diabetes.csv")
columns_to_process = ["Glucose", "BloodPressure", "BMI", "Insulin", "SkinThickness"]

# Replace 0 values with NaN
diabetes[columns_to_process] = diabetes[columns_to_process].replace(0, float('nan'))

median_values = diabetes[columns_to_process].median(skipna=True)

# Replace NaN values with the calculated median
diabetes[columns_to_process] = diabetes[columns_to_process].fillna(median_values)
print(diabetes.head())
print(diabetes.shape)
print(Counter(diabetes['Outcome']))

missing_values = diabetes.isnull().any()

# Display columns with missing values (if any)
missing_columns = missing_values[missing_values]
if missing_columns.empty:
    print("No missing values in the dataset.")
else:
    print("Columns with missing values:")
    print(missing_columns)

X = diabetes.drop('Outcome', axis=1)
Y = diabetes['Outcome']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)

clf = RandomForestClassifier(bootstrap=True, warm_start=True, oob_score=True, n_estimators=1000, max_features=3)

# Calculate OOB score
clf.fit(X_train, Y_train)

# Extract the feature ranking
feature_importances = clf.feature_importances_
feature_names = X_train.columns

# Sort the features by importance in descending order
sorted_features = sorted(zip(feature_importances, feature_names), reverse=True)

# Print the top 10 features
print("\nThe feature ranking is as follows:")
for i in range(8):
    importance, feature_name = sorted_features[i]
    print(f"{feature_name}: {importance}")

# Plot the importances of the top 10 features

top_feature_names = [feature_name for importance, feature_name in sorted_features[:10]]
top_feature_importances = [importance for importance, feature_name in sorted_features[:10]]

plt.figure(figsize=(10, 6))
plt.barh(top_feature_names, top_feature_importances)
plt.xlabel('Feature Importance')
plt.title('Top 10 Features by Importance')
plt.gca().invert_yaxis()
plt.show()