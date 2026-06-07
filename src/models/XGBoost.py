import pandas as pd

import os
import joblib

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)

from xgboost import XGBClassifier

def train_xgboost(df):
    print("\nTraining XGBoost...")
    
    os.makedirs(
        "outputs/XGBoost",
        exist_ok=True
    )
    
    # PREPROCESSING
    # Remove SESSION_ID if it exists
    if 'SESSION_ID' in df.columns:
        df = df.drop('SESSION_ID', axis=1)
    
    # Features and target
    X = df.drop('ACTIVITY_LEVEL', axis=1)
    y = df['ACTIVITY_LEVEL']
    
    # Encode categorical features
    X = pd.get_dummies(
        X,
        columns=[
            'TIME_OF_DAY',
            'HVAC_OPERATION_MODE',
            'AMBIENT_LIGHT_LEVEL'
        ],
        drop_first=True
    )
    
    # Encode target
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)
    
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )
    
    # XGBoost model
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
    
    # Train the model
    xgb_model.fit(X_train, y_train)
    
    # Predictions
    y_pred = xgb_model.predict(X_test)
    
    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
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
    plt.savefig(
        "outputs/XGBoost/confusion_matrix.png",
        bbox_inches="tight"
    )

    plt.close()
    
    # Feature importance
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': xgb_model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by='Importance',
        ascending=False
    )

    print(importance_df)
    
    importance_df.to_csv(
        "outputs/XGBoost/feature_importance.csv",
        index=False
    )
    
    plt.figure(figsize=(10,6))

    sns.barplot(
        data=importance_df.head(10),
        x="Importance",
        y="Feature"
    )

    plt.title("Top 10 Important Features")

    plt.ylabel('Importance Score')
    plt.tight_layout()
    plt.savefig(
        "outputs/XGBoost/feature_importance.png",
        bbox_inches="tight"
    )  

    plt.close()
    
    joblib.dump(
        target_encoder,
        "outputs/XGBoost/label_encoder.pkl"
    )
    
    joblib.dump(
        xgb_model,
        "outputs/XGBoost/xgboost_model.pkl"
    )
    
    return {
        "Model": "XGBoost",
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Estimator": xgb_model,
        "Encoder": target_encoder
    }