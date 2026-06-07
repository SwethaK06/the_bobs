import pandas as pd
import os
import joblib


from catboost import CatBoostClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

import seaborn as sns
import matplotlib.pyplot as plt

def train_catboost(df):
    print("\nTraining CatBoost...")
    
    os.makedirs(
        "outputs/CatBoost",
        exist_ok=True
    )
    
    # PREPROCESSING
    # Remove SESSION_ID if it exists
    if 'SESSION_ID' in df.columns:
        df = df.drop('SESSION_ID', axis=1)
    
    # Features and target
    X = df.drop('ACTIVITY_LEVEL', axis=1)
    y = df['ACTIVITY_LEVEL']
    
    # Encode target
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    print("\nClass Mapping:")
    for i, cls in enumerate(label_encoder.classes_):
        print(f"{cls} -> {i}")
        
    # Identify categorical features
    categorical_features = [
        col for col in [
            'TIME_OF_DAY',
            'HVAC_OPERATION_MODE',
            'AMBIENT_LIGHT_LEVEL'
        ]
        if col in X.columns
    ]

    cat_feature_indices = [
        X.columns.get_loc(col)
        for col in categorical_features
    ]
    
    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )
    
    # Train CatBoost
    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=8,
        loss_function='MultiClass',
        eval_metric='Accuracy',
        random_seed=42,
        verbose=100
    )
    
    model.fit(
        X_train,
        y_train,
        cat_features=cat_feature_indices,
        eval_set=(X_test, y_test)
    )
    
    # Predictions
    y_pred = model.predict(X_test)

    # Convert from shape (n,1) to (n,)
    y_pred = y_pred.flatten().astype(int)
    
    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("\nClassification Report:\n")

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=label_encoder.classes_
        )
    )
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    print("\nConfusion Matrix:\n")
    print(cm)
    
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

    plt.savefig(
        "outputs/CatBoost/confusion_matrix.png",
        bbox_inches="tight"
    )

    plt.close()
    
    # Feature Importance
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
    plt.savefig(
        "outputs/CatBoost/feature_importance.png",
        bbox_inches="tight"
    )
    plt.close()

    print(feature_importance)
    feature_importance.to_csv(
        "outputs/CatBoost/feature_importance.csv",
        index=False
    )
    
    joblib.dump(
        label_encoder,
        "outputs/CatBoost/label_encoder.pkl"
    )
    
    return {
        "Model": "CatBoost",
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Estimator": model,
        "Encoder": label_encoder
    }



