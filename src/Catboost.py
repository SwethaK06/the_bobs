# import pandas as pd
# import numpy as np

# from catboost import CatBoostClassifier

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder

# from sklearn.metrics import (
#     accuracy_score,
#     classification_report,
#     confusion_matrix
# )

# import seaborn as sns
# import matplotlib.pyplot as plt

# # =========================
# # LOAD DATA
# # =========================

# df = pd.read_csv(r"C:\Users\sweth\Desktop\Y3S1\AI Solutions Development\Project\SD_project\data\cleaned_gas_monitoring2.0.csv")

# print(df.head())

# # =========================
# # REMOVE SESSION_ID
# # =========================

# if 'SESSION_ID' in df.columns:
#     df = df.drop('SESSION_ID', axis=1)

# # =========================
# # FEATURES & TARGET
# # =========================

# X = df.drop('ACTIVITY_LEVEL', axis=1)
# y = df['ACTIVITY_LEVEL']

# # =========================
# # ENCODE TARGET
# # =========================

# label_encoder = LabelEncoder()
# y_encoded = label_encoder.fit_transform(y)

# print("\nClass Mapping:")
# for i, cls in enumerate(label_encoder.classes_):
#     print(f"{cls} -> {i}")

# # =========================
# # IDENTIFY CATEGORICAL COLUMNS
# # =========================

# categorical_features = [
#     'TIME_OF_DAY',
#     'HVAC_OPERATION_MODE',
#     'AMBIENT_LIGHT_LEVEL'
# ]

# cat_feature_indices = [
#     X.columns.get_loc(col)
#     for col in categorical_features
# ]

# # =========================
# # TRAIN TEST SPLIT
# # =========================

# X_train, X_test, y_train, y_test = train_test_split(
#     X,
#     y_encoded,
#     test_size=0.2,
#     random_state=42,
#     stratify=y_encoded
# )

# # =========================
# # CATBOOST MODEL
# # =========================

# model = CatBoostClassifier(
#     iterations=500,
#     learning_rate=0.05,
#     depth=8,
#     loss_function='MultiClass',
#     eval_metric='Accuracy',
#     random_seed=42,
#     verbose=100
# )

# # =========================
# # TRAIN
# # =========================

# model.fit(
#     X_train,
#     y_train,
#     cat_features=cat_feature_indices,
#     eval_set=(X_test, y_test)
# )

# # =========================
# # PREDICTIONS
# # =========================

# y_pred = model.predict(X_test)

# # Convert from shape (n,1) to (n,)
# y_pred = y_pred.flatten().astype(int)

# # =========================
# # ACCURACY
# # =========================

# accuracy = accuracy_score(y_test, y_pred)

# print("\nAccuracy:", round(accuracy, 4))

# # =========================
# # CLASSIFICATION REPORT
# # =========================

# print("\nClassification Report:\n")

# print(
#     classification_report(
#         y_test,
#         y_pred,
#         target_names=label_encoder.classes_
#     )
# )

# # =========================
# # CONFUSION MATRIX
# # =========================

# cm = confusion_matrix(y_test, y_pred)

# print("\nConfusion Matrix:\n")
# print(cm)

# # =========================
# # HEATMAP
# # =========================

# plt.figure(figsize=(8,6))

# sns.heatmap(
#     cm,
#     annot=True,
#     fmt='d',
#     cmap='Blues',
#     xticklabels=label_encoder.classes_,
#     yticklabels=label_encoder.classes_
# )

# plt.title("CatBoost Confusion Matrix")
# plt.xlabel("Predicted")
# plt.ylabel("Actual")

# plt.tight_layout()
# plt.show()

# importance = model.get_feature_importance()

# feature_importance = pd.DataFrame({
#     'Feature': X.columns,
#     'Importance': importance
# })

# feature_importance = feature_importance.sort_values(
#     by='Importance',
#     ascending=False
# )

# plt.figure(figsize=(10,6))

# sns.barplot(
#     data=feature_importance,
#     x='Importance',
#     y='Feature'
# )

# plt.title("CatBoost Feature Importance")
# plt.tight_layout()
# plt.show()

# print(feature_importance)

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from catboost import CatBoostClassifier

# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv(r"C:\Users\sweth\Desktop\Y3S1\AI Solutions Development\Project\SD_project\data\cleaned_gas_monitoring2.0.csv")

# ==========================================
# REMOVE SESSION ID
# ==========================================

if "SESSION_ID" in df.columns:
    df = df.drop(columns=["SESSION_ID"])

# ==========================================
# FEATURE ENGINEERING
# ==========================================

# Temperature/Humidity Features

df["TEMP_HUMIDITY"] = (
    df["TEMPERATURE"] *
    df["HUMIDITY"]
)

df["TEMP_HUM_RATIO"] = (
    df["TEMPERATURE"] /
    (df["HUMIDITY"] + 1)
)

df["TEMP_MINUS_HUM"] = (
    df["TEMPERATURE"] -
    df["HUMIDITY"]
)

df["TEMP_SQ"] = (
    df["TEMPERATURE"] ** 2
)

df["HUMIDITY_SQ"] = (
    df["HUMIDITY"] ** 2
)

# CO2 Features

df["CO2_RATIO"] = (
    df["CO2_INFRAREDSENSOR"] /
    (df["CO2_ELECTROCHEMICALSENSOR"] + 1)
)

df["CO2_DIFF"] = (
    df["CO2_INFRAREDSENSOR"] -
    df["CO2_ELECTROCHEMICALSENSOR"]
)

df["CO2_AVG"] = (
    df["CO2_INFRAREDSENSOR"] +
    df["CO2_ELECTROCHEMICALSENSOR"]
) / 2

df["TEMP_CO2"] = (
    df["TEMPERATURE"] *
    df["CO2_INFRAREDSENSOR"]
)

df["HUMIDITY_CO2"] = (
    df["HUMIDITY"] *
    df["CO2_INFRAREDSENSOR"]
)

# MOS Features

mos_cols = [
    "METALOXIDESENSOR_UNIT1",
    "METALOXIDESENSOR_UNIT2",
    "METALOXIDESENSOR_UNIT3",
    "METALOXIDESENSOR_UNIT4"
]

df["MOS_TOTAL"] = df[mos_cols].sum(axis=1)
df["MOS_AVG"] = df[mos_cols].mean(axis=1)
df["MOS_STD"] = df[mos_cols].std(axis=1)

df["MOS_RANGE"] = (
    df[mos_cols].max(axis=1)
    - df[mos_cols].min(axis=1)
)

df["MOS_CO_RATIO"] = (
    df["MOS_TOTAL"] /
    (df["CO_GASSENSOR"] + 1)
)

# ==========================================
# TARGET
# ==========================================

TARGET = "TIME_OF_DAY"

X = df.drop(columns=[TARGET])
y = df[TARGET]

# ==========================================
# DETECT CATEGORICAL FEATURES
# ==========================================

cat_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

# Convert categoricals to string
for col in cat_features:
    X[col] = X[col].astype(str)

print("\nCategorical Features:")
print(cat_features)

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ==========================================
# CATBOOST MODEL
# ==========================================

model = CatBoostClassifier(
    iterations=2000,
    learning_rate=0.03,
    depth=8,

    loss_function="MultiClass",
    eval_metric="Accuracy",

    random_seed=42,

    early_stopping_rounds=100,

    verbose=100
)

# ==========================================
# TRAIN
# ==========================================

print("\nTraining CatBoost...")

model.fit(
    X_train,
    y_train,
    cat_features=cat_features,
    eval_set=(X_test, y_test)
)

# ==========================================
# PREDICT
# ==========================================

predictions = model.predict(X_test)

predictions = predictions.flatten()

# ==========================================
# EVALUATION
# ==========================================

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\nAccuracy:")
print(f"{accuracy:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        predictions
    )
)

# ==========================================
# FEATURE IMPORTANCE
# ==========================================

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.get_feature_importance()
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 30 Most Important Features")
print(
    importance_df.head(30)
)

importance_df.to_csv(
    "feature_importance_catboost.csv",
    index=False
)

print(
    "\nFeature importance saved to "
    "feature_importance_catboost.csv"
)