import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# =========================
# LOAD CLEANED DATASET
# =========================

import pandas as pd

df = pd.read_csv("cleaned_gas_monitoring.csv")

print("Cleaned dataset loaded successfully.")
# display(df.head())
print(df.shape)


# =========================
# PREPARE FEATURES AND TARGET
# =========================

# Target column
y = df["ACTIVITY_LEVEL"]

# Drop target and SESSION_ID
# SESSION_ID is just an identifier, not useful for prediction
X = df.drop(columns=["ACTIVITY_LEVEL", "SESSION_ID"])

# Treat CO_GASSENSOR as categorical because values are 1-4
X["CO_GASSENSOR"] = X["CO_GASSENSOR"].astype(str)

# Convert categorical text columns into dummy variables
X = pd.get_dummies(X, drop_first=False)

print("Features shape:")
print(X.shape)

print("\nTarget distribution:")
print(y.value_counts())

# =========================
# TRAIN TEST SPLIT
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
# DECISION TREE HYPERPARAMETER TUNING
# =========================

dt = DecisionTreeClassifier(random_state=42)

param_grid = {
    "criterion": ["gini", "entropy"],
    "max_depth": [3, 5, 7, 10, 15, None],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 2, 5, 10],
    "class_weight": [None, "balanced"]
}

grid_search = GridSearchCV(
    dt,
    param_grid,
    cv=5,
    scoring="f1_weighted",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("Best parameters:")
print(grid_search.best_params_)

print("\nBest cross-validation score:")
print(round(grid_search.best_score_, 4))


# =========================
# USE BEST TUNED DECISION TREE MODEL
# =========================

best_dt_model = grid_search.best_estimator_

y_pred_best_dt = best_dt_model.predict(X_test)

print("Final Tuned Decision Tree Accuracy:")
print(round(accuracy_score(y_test, y_pred_best_dt), 4))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_best_dt))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_best_dt))

# =========================
# MAKE PREDICTIONS
# =========================

y_pred_dt = best_dt_model.predict(X_test)

print("Predictions completed.")

# =========================
# EVALUATE DECISION TREE MODEL
# =========================

dt_accuracy = accuracy_score(y_test, y_pred_dt)

print("Decision Tree Accuracy:")
print(round(dt_accuracy, 4))

print("\nDecision Tree Accuracy Percentage:")
print(round(dt_accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_dt))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_dt))


# =========================
# CONFUSION MATRIX FOR TUNED DECISION TREE
# =========================

cm = confusion_matrix(y_test, y_pred_best_dt)

plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=best_dt_model.classes_,
    yticklabels=best_dt_model.classes_
)

plt.title("Tuned Decision Tree Confusion Matrix")
plt.xlabel("Predicted Activity Level")
plt.ylabel("Actual Activity Level")
plt.show()

# =========================
# FEATURE IMPORTANCE FOR TUNED DECISION TREE
# =========================

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": best_dt_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

# display(feature_importance.head(10))

# =========================
# FEATURE IMPORTANCE BAR CHART
# =========================

top_features = feature_importance.head(10)

plt.figure(figsize=(10, 5))
plt.barh(top_features["Feature"], top_features["Importance"])
plt.gca().invert_yaxis()
plt.title("Top 10 Important Features - Decision Tree")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.show()

