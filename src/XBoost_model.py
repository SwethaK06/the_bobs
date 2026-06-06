import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV

# ======================
# LOAD DATA
# ======================
df = pd.read_csv(r"C:\Users\sweth\Desktop\Y3S1\AI Solutions Development\Project\SD_project\notebooks\cleaned_gas_monitoring.csv")

# ======================
# FEATURES & TARGET
# ======================
X = df.drop(["ACTIVITY_LEVEL", "SESSION_ID"], axis=1)
y = df["ACTIVITY_LEVEL"]

# ======================
# ENCODE CATEGORICAL FEATURES
# ======================
# for col in X.select_dtypes(include=['object']).columns:
#     le = LabelEncoder()
#     X[col] = le.fit_transform(X[col].astype(str))

X = pd.get_dummies(
    X,
    columns=[
        'TIME_OF_DAY',
        'HVAC_OPERATION_MODE',
        'AMBIENT_LIGHT_LEVEL'
    ],
    drop_first=True
)

# Encode target variable
target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)

# ======================
# TRAIN TEST SPLIT
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# ======================
# XGBOOST MODEL
# ======================
# xgb_model = XGBClassifier(
#     n_estimators=300,
#     learning_rate=0.05,
#     max_depth=6,
#     objective='multi:softprob',
#     num_class=len(np.unique(y_encoded)),
#     random_state=42,
#     eval_metric='mlogloss'
# )

xgb_model = XGBClassifier(
    n_estimators=800,
    learning_rate=0.02,
    max_depth=5,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    random_state=42,
    eval_metric='mlogloss'
)

# Train model
xgb_model.fit(X_train, y_train)

# ======================
# PREDICTIONS
# ======================
y_pred = xgb_model.predict(X_test)

# ======================
# EVALUATION
# ======================
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=target_encoder.classes_
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': xgb_model.feature_importances_
})

importance_df = importance_df.sort_values(
    by='Importance',
    ascending=False
)

print(importance_df)

import matplotlib.pyplot as plt

importance_df.head(10).plot(
    x='Feature',
    y='Importance',
    kind='bar'
)

plt.title('Top 10 Important Features')
plt.ylabel('Importance Score')
plt.tight_layout()
plt.show()

print(df["ACTIVITY_LEVEL"].value_counts())
print(df["ACTIVITY_LEVEL"].value_counts(normalize=True) * 100)

print(confusion_matrix(y_test, y_pred))

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot heatmap
plt.figure(figsize=(8,6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    cbar=True
)

plt.title('Confusion Matrix Heatmap')
plt.xlabel('Predicted Activity Level')
plt.ylabel('Actual Activity Level')
plt.show()