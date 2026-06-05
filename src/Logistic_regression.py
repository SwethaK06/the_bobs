# import pandas as pd
# import numpy as np

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.impute import SimpleImputer

# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import (
#     accuracy_score,
#     classification_report,
#     confusion_matrix
# )

# df = pd.read_csv(r"C:\Users\sweth\Desktop\Y3S1\AI Solutions Development\Project\SD_project\data\cleaned_gas_monitoring2.0.csv")

# print(df.head())
# print(df.info())

# X = df.drop("ACTIVITY_LEVEL", axis=1)
# y = df["ACTIVITY_LEVEL"]

# label_encoder = LabelEncoder()
# y_encoded = label_encoder.fit_transform(y)

# print("Classes:", label_encoder.classes_)

# categorical_features = X.select_dtypes(include=['object']).columns
# numerical_features = X.select_dtypes(exclude=['object']).columns

# print("Categorical Features:")
# print(categorical_features)

# print("\nNumerical Features:")
# print(numerical_features)

# numeric_transformer = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='median')),
#     ('scaler', StandardScaler())
# ])

# categorical_transformer = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='most_frequent'))
# ])

# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', numeric_transformer, numerical_features),
#         ('cat', categorical_transformer, categorical_features)
#     ]
# )

# X_encoded = X.copy()

# for col in categorical_features:
#     le = LabelEncoder()
#     X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
    
# X_train, X_test, y_train, y_test = train_test_split(
#     X_encoded,
#     y_encoded,
#     test_size=0.2,
#     random_state=42,
#     stratify=y_encoded
# )

# scaler = StandardScaler()

# X_train[numerical_features] = scaler.fit_transform(
#     X_train[numerical_features]
# )

# X_test[numerical_features] = scaler.transform(
#     X_test[numerical_features]
# )

# model = LogisticRegression(
#     max_iter=1000,
#     multi_class='multinomial'
# )

# model.fit(X_train, y_train)

# y_pred = model.predict(X_test)

# accuracy = accuracy_score(y_test, y_pred)

# print("Accuracy:", accuracy)

# print("\nClassification Report:")
# print(classification_report(y_test, y_pred))

# print("\nConfusion Matrix:")
# print(confusion_matrix(y_test, y_pred))

# feature_importance = pd.DataFrame(
#     model.coef_.T,
#     index=X_train.columns,
#     columns=[f"Class_{i}" for i in range(model.coef_.shape[0])]
# )

# print(feature_importance)

# import seaborn as sns
# import matplotlib.pyplot as plt

# cm = confusion_matrix(y_test, y_pred)

# plt.figure(figsize=(6,5))
# sns.heatmap(
#     cm,
#     annot=True,
#     fmt='d',
#     cmap='Blues',
#     xticklabels=label_encoder.classes_,
#     yticklabels=label_encoder.classes_
# )

# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.title("Confusion Matrix")
# plt.show()

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import seaborn as sns
import matplotlib.pyplot as plt

# =========================
# Load Dataset
# =========================

df = pd.read_csv(
    r"C:\Users\sweth\Desktop\Y3S1\AI Solutions Development\Project\SD_project\data\cleaned_gas_monitoring2.0.csv"
)

print(df.head())
print(df.info())



# =========================
# Features and Target
# =========================

X = df.drop("ACTIVITY_LEVEL", axis=1)
X = X.drop("SESSION_ID", axis=1)
y = df["ACTIVITY_LEVEL"]

# One-Hot Encode categorical features
X = pd.get_dummies(
    X,
    columns=[
        'TIME_OF_DAY',
        'HVAC_OPERATION_MODE',
        'AMBIENT_LIGHT_LEVEL'
    ],
    drop_first=True
)

y = df["ACTIVITY_LEVEL"]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# =========================
# Identify Feature Types
# =========================

# # Fixes Pandas 3.0 warning
# categorical_features = X.select_dtypes(
#     include=['object', 'string']
# ).columns

# numerical_features = X.select_dtypes(
#     exclude=['object', 'string']
# ).columns

# print("\nCategorical Features:")
# print(categorical_features)

# print("\nNumerical Features:")
# print(numerical_features)

# =========================
# Encode Categorical Features
# =========================

# X_encoded = X.copy()

# label_encoders = {}

# for col in categorical_features:
#     le = LabelEncoder()
#     X_encoded[col] = le.fit_transform(
#         X_encoded[col].astype(str)
#     )
#     label_encoders[col] = le

# =========================
# Train-Test Split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

feature_names = X.columns

# =========================
# Scale Numerical Features
# =========================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# scaler = StandardScaler()

# X_train = X_train.copy()
# X_test = X_test.copy()

# X_train.loc[:, numerical_features] = scaler.fit_transform(
#     X_train[numerical_features]
# )

# X_test.loc[:, numerical_features] = scaler.transform(
#     X_test[numerical_features]
# )

print(X.shape)
print(X.columns)

# =========================
# Train Logistic Regression
# =========================

model = LogisticRegression(
    max_iter=2000,
    solver='lbfgs',
    C=1.0
)

model.fit(X_train, y_train)

# =========================
# Predictions
# =========================

y_pred = model.predict(X_test)

# =========================
# Evaluation
# =========================

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_
    )
)

# =========================
# Confusion Matrix
# =========================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Logistic Regression Confusion Matrix")
plt.tight_layout()
plt.show()

# =========================
# Feature Importance
# =========================

feature_importance = pd.DataFrame(
    model.coef_.T,
    index=feature_names,
    columns=[
        f"Class_{label}"
        for label in label_encoder.classes_
    ]
)

print("\nFeature Importance:")
print(feature_importance)

# Optional: save feature importance
feature_importance.to_csv(
    "logistic_regression_feature_importance.csv"
)

# =========================
# Accuracy Percentage
# =========================

print(
    f"\nModel Accuracy: {accuracy * 100:.2f}%"
)