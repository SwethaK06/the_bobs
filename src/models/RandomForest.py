import pandas as pd
import os
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)

def train_random_forest(df):
    print("\nTraining Random Forest...")
    
    os.makedirs(
        "outputs/RandomForest",
        exist_ok=True
    )
    
    # Preprocessing
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
    precision = precision_score(y_test, predictions, average='weighted')
    recall = recall_score(y_test, predictions, average='weighted')
    f1 = f1_score(y_test, predictions, average='weighted')
    
    print("\nAccuracy:")
    print(f"{accuracy:.4f}")

    print("\nPrecision:")
    print(f"{precision:.4f}")

    print("\nRecall:")
    print(f"{recall:.4f}")

    print("\nF1 Score:")
    print(f"{f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    cm = confusion_matrix(
        y_test,
        predictions
    )

    plt.figure(figsize=(8,6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )   

    plt.title("Random Forest Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.tight_layout()

    plt.savefig(
        "outputs/RandomForest/confusion_matrix.png",
        bbox_inches="tight"
    )

    plt.close()

    #feature importance
    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    })
    importance_df.to_csv(
        "outputs/RandomForest/feature_importance.csv",
        index=False
    )

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )
    
    plt.figure(figsize=(10,6))

    sns.barplot(
        data=importance_df.head(10),
        x="Importance",
        y="Feature"
    )

    plt.title("Top 10 Important Features")

    plt.tight_layout()

    plt.savefig(
        "outputs/RandomForest/feature_importance.png",
        bbox_inches="tight"
    )

    plt.close()

    #save the model
    joblib.dump(
        model,
        "outputs/RandomForest/random_forest_model.pkl"
    )
    
    return {
        "Model": "Random Forest",
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Estimator": model,
        "Encoder": None
}
