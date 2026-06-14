import pandas as pd # to read and manipulate data

import os # to create directories for outputs
import joblib # to save trained models and encoders

import matplotlib.pyplot as plt # for plotting confusion matrix and feature importance
import seaborn as sns # for visualization of confusion matrix and feature importance

from sklearn.model_selection import train_test_split # to split data into training and testing sets
from sklearn.preprocessing import LabelEncoder # to encode target variable
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
) # to evaluate model performance

from xgboost import XGBClassifier # xgboost implementation for training

def train_xgboost(df):
    print("\nTraining XGBoost...")
    
    # Makes a folder called CatBoost in outputs if it doesn't already exist to save results
    os.makedirs(
        "outputs/XGBoost",
        exist_ok=True
    )
    
    # ================
    #  PREPROCESSING
    # ================
    
    # Remove SESSION_ID if it exists
    # I removed session ID because I felt that it is not useful for training a model as it only acts as an identifier, so it would learn wrong patterns if we train the model using SESSION_ID
    if 'SESSION_ID' in df.columns:
        df = df.drop('SESSION_ID', axis=1)
    
    # Features and target
    X = df.drop('ACTIVITY_LEVEL', axis=1) # Dropped the activity level for Features as activity level is a target
    y = df['ACTIVITY_LEVEL'] # Added activity level column to the target value as it is the value that the model is going to predict
    
    # XGBoost cannot naturally understand text categories
    # XGBoost works well with one-hot encoding as:
    # 1. Converts categorical text into numbers
    # 2. Creates clean binary features for tree splits
    # 3. Improves interpretability
    # 4. Works well with decision-tree-based boosting algorithms
    # 5. Standard pre-processing techniques for XGBoost when using categorical features
    
    # Encode categorical features
    # This performs One-hot Encoding on categorical features
    # get_dummies() converts categories into binary (0/1) columns
    X = pd.get_dummies(
        X,
        # Specifies which columns should be one-hot encoded
        columns=[
            'TIME_OF_DAY',
            'HVAC_OPERATION_MODE',
            'AMBIENT_LIGHT_LEVEL'
        ],
        drop_first=True # drops the first category from each encoded feature
    )
    
    # XGBoost cannot predict text label directly, it needs numerical target variables
    # Encode target
    target_encoder = LabelEncoder() # Creates a LabelEncoder object
    y_encoded = target_encoder.fit_transform(y) # fit(y) learns all unique classes in the target variable and transform(y) converts the original target values into encoded numerical values
    
    # ===================
    #  TEST-TRAIN SPLIT
    # ===================
    
    # Define the varaibles, X_train for training featurs, X_tests is for Testing features, y_train is for training labels, y_test is for Testing labels
    # Machine Learning Models should be trained on one set of data and evaluated on unseen data
    X_train, X_test, y_train, y_test = train_test_split(
        X, # Passes the feature matrix into the function, x contains all predictor variables. The model needs these inputs to make predictions
        y_encoded, # Passes the target variable. These are the values the model is trying to predict. Without y, the model would not know the corract answers
        test_size=0.2,  # Uses 20% of the data for testing, uses 80% of the data for testing. This is good for Catboost as it performs very well with moderate amounts of data. This usually provides Good learning capaiblity and reliable performance measurement
        random_state=42, # Sets the random seed, without this every run might create a different split. Once this is applied, every run gets exactly the same split
        stratify=y_encoded # Maintains the same class distribution in both training and testing sets. This is extremely important for classification problems.
    )
    
    # =================
    #  TRAIN XGBoost
    # =================
    
    # XGBoost Model (Extreme Gradient Boosting) is a powerful learning algorithms that:
    # 1. Builds many decision trees
    # 2. Combines their predictions 
    # 3. Corrects previous mistakes every step
    
    # XGBoost model
    # Configurations for the XGBoost Model
    xgb_model = XGBClassifier(
        n_estimators=800, # Specifies the number of trees to build, each tree learns from mistakes made by previous trees
        learning_rate=0.02, # Controls how much each tree contributes. 0.02 is a small and conservative learning rate
        max_depth=5, # Limits how deep each tree can grow. Controls model complexitiy
        min_child_weight=3, # Controls the minimum amount of data required before a split is allowed. Prevents trees from creating tiny branches that memorize training data
        subsample=0.8, # Uses only  80% of the training rows for each tree. Adds randomness, makes the ensemble more diverse. Less overfitting, better generalization & more robust model
        colsample_bytree=0.8, # Uses only 80% of features when building each tree. Prevents the model from relying too heavily on the same features. Creates more diverse trees
        gamma=0.1, # Controls how beneficial a split must be before XGBoost allows it
        random_state=42, # Fixes the randomness used during training. machine learning algorithms contain randomness by sampling, tree construction and feature selection
        eval_metric='mlogloss' # Multi-class Log loss. XGBoost needs a metric designed for multiclass classification
    )
    
    # This is where the actual training of the XGBoost model begins. 
    # The XGBoost starts training and learns relationships between X_train and y_train
    xgb_model.fit(X_train, y_train)
    
    # ==============
    #  PREDICTIONS
    # ==============
    # Uses the trained XGBoost model to predict the classes of the test dataset
    y_pred = xgb_model.predict(X_test)
    
    # =============
    #  EVALUATION
    # =============
    
    # Uses the y_test as actual labels and y_pred as predicted labels
    # These calculate the performance meterics of the XGBoost model
    accuracy = accuracy_score(y_test, y_pred) # Calculates model's accuracy. Accuracy = Correct predictions / Total Predictions
    # Weighted averaging combines all the precisions, recall, f1 score values while considering class frequencies
    precision = precision_score(y_test, y_pred, average='weighted') # Calculates model precision. Precision = TP / TP + FP. 
    recall = recall_score(y_test, y_pred, average='weighted') # Calculates model recall. Recall = TP / TP + FN
    f1 = f1_score(y_test, y_pred, average='weighted')  # Calculates model f1 score. Combines precision and recall. F1 = 2 x [(precision x recall) / (precision + recall)]
    # Prints out all the performance metric values
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Prints detailed classification report
    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=target_encoder.classes_ # Replaces numeric class labels with the original class names
        )
    )
    
    # Confusion Matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # ================
    #  VISUALIZATIONS
    # ================
    
    # Heatmap for the confusion Matrix
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
    # saves the confusion matrix picture
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
    
    # Save feature importance
    # Saves teh Dataframe to CSV file
    # index=False prevents pandas from saving row numbers
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
    # Save Feature Importance graph 
    plt.savefig(
        "outputs/XGBoost/feature_importance.png",
        bbox_inches="tight"
    )  

    plt.close()
    
    # Saves LabelEncoder object to disk. Store this so that we are able to reuse it later.
    joblib.dump(
        target_encoder,
        "outputs/XGBoost/label_encoder.pkl"
    )
    
    # Saves model to disk. Store this so that we are able to reuse it later.
    joblib.dump(
        xgb_model,
        "outputs/XGBoost/xgboost_model.pkl"
    )
    
    # Return all these values to the pipeline.py file
    return {
        "Model": "XGBoost",
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Estimator": xgb_model,
        "Encoder": target_encoder
    }