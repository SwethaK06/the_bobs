import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

#load data
df = pd.read_csv("cleaned_gas_monitoring (1).csv")

#drop session id
if "SESSION_ID" in df.columns:
    df = df.drop(columns=["SESSION_ID"])

#set activity level as target
TARGET = "ACTIVITY_LEVEL"

X = df.drop(columns=[TARGET])
y = df[TARGET]

#one hot encode categorical features
categorical_features = X.select_dtypes(
    include=["object"]
).columns

X = pd.get_dummies(
    X,
    columns=categorical_features,
    drop_first=True
)

#train test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

#random forest model
model = RandomForestClassifier(
    n_estimators=2000,
    max_features=0.8,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    bootstrap=True,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

#train model
model.fit(X_train, y_train)

#make predictions
predictions = model.predict(X_test)

#evaluate
accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy:")
print(f"{accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

#feature importance
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 20 Most Important Features")
print(importance_df.head(20))

