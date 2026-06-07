import pandas as pd
import os
import joblib

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,  # noqa: F401
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)


def train_knn(df):
    print("\nTraining KNN...")
    
    os.makedirs(
        "outputs/KNN",
        exist_ok=True
    )
    
    # Preprocessing
    y = df["ACTIVITY_LEVEL"]

    X = df.drop(
        columns=["ACTIVITY_LEVEL", "SESSION_ID"],
        errors="ignore"
    )

    # CO_GASSENSOR values are 1-4, so treat it as a category instead of continuous number
    X["CO_GASSENSOR"] = X["CO_GASSENSOR"].astype(str)
    
    # Use drop_first=False for KNN so each category is represented clearly
    X = pd.get_dummies(X, drop_first=False)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    # Tune knn 
    
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
    
    # Prediction
    
    best_knn_model = grid_search.best_estimator_

    y_pred_knn = best_knn_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred_knn)
    precision = precision_score(y_test, y_pred_knn, average='weighted')
    recall = recall_score(y_test, y_pred_knn, average='weighted')
    f1 = f1_score(y_test, y_pred_knn, average='weighted')
    
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

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
    plt.savefig(
        "outputs/KNN/confusion_matrix.png",
        bbox_inches="tight"
    )

    plt.close()
    
    # Calculate permutation importance
    perm_importance = permutation_importance(
        best_knn_model,
        X_test,
        y_test,
        n_repeats=10,
        random_state=42,
        scoring="accuracy",
        n_jobs=-1
    )

    # Put results into a DataFrame
    feature_importance = pd.DataFrame({
        "Feature": X_test.columns,
        "Importance": perm_importance.importances_mean
    })

    # Sort from most important to least important
    feature_importance = feature_importance.sort_values(
        by="Importance",
        ascending=False
    )
    
    feature_importance.to_csv(
        "outputs/KNN/feature_importance.csv",
        index=False
    )

    print("\nKNN Feature Importance:")
    print(feature_importance.head(15))

    # FEATURE IMPORTANCE BAR CHART


    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=feature_importance.head(15),
        x="Importance",
        y="Feature"
    )

    plt.title("Top 15 Feature Importance for KNN")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(
        "outputs/KNN/feature_importance.png",
        bbox_inches="tight"
    )
    plt.close()
    
    joblib.dump(
        best_knn_model,
        "outputs/KNN/knn_model.pkl"
    )

    return {
        "Model": "KNN",
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Estimator": best_knn_model,
        "Encoder": None
    }