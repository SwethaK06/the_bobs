import pandas as pd
import os
import joblib

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    LabelEncoder,
    OrdinalEncoder
)

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)



def train_gradient_boosting(df):
    print("\nTraining Gradient Boosting...")
    
    os.makedirs(
        "outputs/GradientBoosting",
        exist_ok=True
    )
    
    # Preprocessing
    #remove session id
    if "SESSION_ID" in df.columns:
        df = df.drop(columns=["SESSION_ID"])
        
    #target
    TARGET = "ACTIVITY_LEVEL"

    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    
    #encoding target
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(y)

    #encode catagorical features
    categorical_cols = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    encoder = None

    if len(categorical_cols) > 0:
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )

        X[categorical_cols] = encoder.fit_transform(
            X[categorical_cols].astype(str)
        )
        
    #train test split  
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
    
    #gradient boosting model
    model = GradientBoostingClassifier(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=6,
        min_samples_split=5,
        min_samples_leaf=2,
        subsample=0.8,
        random_state=42
    )
    
    #training gradient boosting model
    model.fit(
        X_train,
        y_train
    )
    
    #predictions
    predictions = model.predict(X_test)
    
    #evaluation
    accuracy = accuracy_score(
        y_test,
        predictions
    )
    f1 = f1_score(
        y_test, 
        predictions,
        average="weighted"
    )
    
    precision = precision_score(y_test, predictions, average='weighted')
    recall = recall_score(y_test, predictions, average='weighted')
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    print("\nAccuracy:")
    print(f"{accuracy:.4f}")

    print("\nF1 Score:")
    print(f"{f1:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            target_encoder.inverse_transform(y_test),
            target_encoder.inverse_transform(predictions)
        )
    )

    print("\nConfusion Matrix:")
    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )
    
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

    plt.title("Gradient Boosting Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.tight_layout()

    plt.savefig(
        "outputs/GradientBoosting/confusion_matrix.png",
        bbox_inches="tight"
    )

    plt.close()
    
    #to see feature importance
    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )
    
    importance_df.to_csv(
        "outputs/GradientBoosting/feature_importance.csv",
        index=False
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
        "outputs/GradientBoosting/feature_importance.png",
        bbox_inches="tight"
    )

    plt.close()
    
    joblib.dump(
        model,
        "outputs/GradientBoosting/gradient_boosting_model.pkl"
    )
    
    joblib.dump(
        target_encoder,
        "outputs/GradientBoosting/target_encoder.pkl"
    )
    
    return {
        "Model": "Gradient Boosting",
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Estimator": model,
        "Encoder": target_encoder
    }
    
