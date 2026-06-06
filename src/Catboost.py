import pandas as pd
import numpy as np

from catboost import CatBoostClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import seaborn as sns
import matplotlib.pyplot as plt

# =========================
# LOAD DATA
# =========================

df = pd.read_csv(r"C:\Users\sweth\Desktop\Y3S1\AI Solutions Development\Project\SD_project\notebooks\cleaned_gas_monitoring.csv")

print(df.head())

# =========================
# REMOVE SESSION_ID
# =========================

if 'SESSION_ID' in df.columns:
    df = df.drop('SESSION_ID', axis=1)

# =========================
# FEATURES & TARGET
# =========================

X = df.drop('ACTIVITY_LEVEL', axis=1)
y = df['ACTIVITY_LEVEL']

# =========================
# ENCODE TARGET
# =========================

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("\nClass Mapping:")
for i, cls in enumerate(label_encoder.classes_):
    print(f"{cls} -> {i}")

# =========================
# IDENTIFY CATEGORICAL COLUMNS
# =========================

categorical_features = [
    'TIME_OF_DAY',
    'HVAC_OPERATION_MODE',
    'AMBIENT_LIGHT_LEVEL'
]

cat_feature_indices = [
    X.columns.get_loc(col)
    for col in categorical_features
]

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# =========================
# CATBOOST MODEL
# =========================

model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=8,
    loss_function='MultiClass',
    eval_metric='Accuracy',
    random_seed=42,
    verbose=100
)

# =========================
# TRAIN
# =========================

model.fit(
    X_train,
    y_train,
    cat_features=cat_feature_indices,
    eval_set=(X_test, y_test)
)

# =========================
# PREDICTIONS
# =========================

y_pred = model.predict(X_test)

# Convert from shape (n,1) to (n,)
y_pred = y_pred.flatten().astype(int)

# =========================
# ACCURACY
# =========================

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", round(accuracy, 4))

# =========================
# CLASSIFICATION REPORT
# =========================

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_
    )
)

# =========================
# CONFUSION MATRIX
# =========================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:\n")
print(cm)

# =========================
# HEATMAP
# =========================

plt.figure(figsize=(8,6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)

plt.title("CatBoost Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.show()

importance = model.get_feature_importance()

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importance
})

feature_importance = feature_importance.sort_values(
    by='Importance',
    ascending=False
)

plt.figure(figsize=(10,6))

sns.barplot(
    data=feature_importance,
    x='Importance',
    y='Feature'
)

plt.title("CatBoost Feature Importance")
plt.tight_layout()
plt.show()

print(feature_importance)

