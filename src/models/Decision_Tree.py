import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

from sklearn.model_selection import GridSearchCV



def train_decision_tree(df):
    print("\nTraining Decision Tree...")
    os.makedirs(
        "outputs/DecisionTree",
        exist_ok=True
    )
    
    # PREPROCESSING
    # Drop target and SESSION_ID
    # SESSION_ID is just an identifier, not useful for prediction
    X = df.drop(
        columns=["ACTIVITY_LEVEL", "SESSION_ID"],
        errors="ignore"
    )
    
    # Treat CO_GASSENSOR as categorical because values are 1-4
    X["CO_GASSENSOR"] = X["CO_GASSENSOR"].astype(str)
    
    # Convert categorical text columns into dummy variables
    X = pd.get_dummies(X, drop_first=False)
    
    # Target column
    y = df["ACTIVITY_LEVEL"]
    
    # TRAIN TEST SPLIT
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    # Decision Tree Hyperparameter Tuning
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
    
    # TRAIN
    best_dt_model = grid_search.best_estimator_
    
    # PREDICTION
    y_pred_dt = best_dt_model.predict(X_test)
    print("Predictions completed.")
    
    # EVALUATION
    accuracy = accuracy_score(y_test, y_pred_dt)
    precision = precision_score(y_test, y_pred_dt, average='weighted')
    recall = recall_score(y_test, y_pred_dt, average='weighted')
    f1 = f1_score(y_test, y_pred_dt , average='weighted')
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred_dt))
    
    # CONFUSION MATRIX
    cm = confusion_matrix(y_test, y_pred_dt)

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
    plt.savefig(
        "outputs/DecisionTree/confusion_matrix.png",
        bbox_inches="tight"
    )

    plt.close()
    
    # Feature importance
    feature_importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": best_dt_model.feature_importances_
    })
    
    feature_importance.to_csv(
        "outputs/DecisionTree/feature_importance.csv",
        index=False
    )

    feature_importance = feature_importance.sort_values(
        by="Importance",
        ascending=False
    )
    
    top_features = feature_importance.head(10)

    plt.figure(figsize=(10, 5))
    plt.barh(top_features["Feature"], top_features["Importance"])
    plt.gca().invert_yaxis()
    plt.title("Top 10 Important Features - Decision Tree")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.savefig(
        "outputs/DecisionTree/feature_importance.png",
        bbox_inches="tight"
    )
    plt.close()
    
    joblib.dump(
        best_dt_model,
        "outputs/DecisionTree/decision_tree_model.pkl"
    )

    return {
        "Model": "Decision Tree",
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Estimator": best_dt_model,
        "Encoder": None
    }
    