# =========================
# IMPROVED KNN MODEL
# =========================

from networkx import display
import pandas as pd
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =========================
# LOAD CLEANED DATASET
# =========================

df = pd.read_csv("cleaned_gas_monitoring.csv")

print("Cleaned dataset loaded successfully.")
# display(df.head())
print(df.shape)

# =========================
# PREPARE FEATURES AND TARGET
# =========================

y = df["ACTIVITY_LEVEL"]

X = df.drop(columns=["ACTIVITY_LEVEL", "SESSION_ID"])

# CO_GASSENSOR values are 1-4, so treat it as a category instead of continuous number
X["CO_GASSENSOR"] = X["CO_GASSENSOR"].astype(str)

print("Features used:")
print(X.columns)

print("\nTarget:")
print(y.name)

# =========================
# ENCODE CATEGORICAL COLUMNS
# =========================
# Use drop_first=False for KNN so each category is represented clearly

X = pd.get_dummies(X, drop_first=False)

print("Shape after encoding:")
print(X.shape)

display(X.head())

# =========================
# TRAIN-TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

# =========================
# TUNE KNN MODEL
# =========================
# This tests different K values, distance weighting, and distance metrics

knn_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier())
])

param_grid = {
    "knn__n_neighbors": list(range(3, 31)),
    "knn__weights": ["uniform", "distance"],
    "knn__p": [1, 2]
}

grid_search = GridSearchCV(
    knn_pipeline,
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("Best parameters:")
print(grid_search.best_params_)

print("\nBest cross-validation accuracy:")
print(round(grid_search.best_score_, 4))

# =========================
# TEST BEST KNN MODEL
# =========================

best_knn_model = grid_search.best_estimator_

y_pred_knn = best_knn_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred_knn)

print("Improved KNN Accuracy:")
print(round(accuracy, 4))

print("\nImproved KNN Accuracy Percentage:")
print(round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_knn))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_knn))

# =========================
# CONFUSION MATRIX HEATMAP
# =========================

cm = confusion_matrix(y_test, y_pred_knn)

plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=best_knn_model.classes_,
    yticklabels=best_knn_model.classes_
)

plt.title("Improved KNN Confusion Matrix")
plt.xlabel("Predicted Activity Level")
plt.ylabel("Actual Activity Level")
plt.show()

