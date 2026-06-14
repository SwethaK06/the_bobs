import pandas as pd # to read and manipulate data
import os # to create directories for outputs
import joblib # to save trained models and encoders


from catboost import CatBoostClassifier # CatBoost implementation for training

from sklearn.model_selection import train_test_split # to split data into training and testing sets
from sklearn.preprocessing import LabelEncoder # to encode target variable

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
) # to evaluate model performance

import seaborn as sns # for visualization of confusion matrix and feature importance
import matplotlib.pyplot as plt # for plotting confusion matrix and feature importance

def train_catboost(df):
    print("\nTraining CatBoost...")
    
    # Makes a folder called CatBoost in outputs if it doesn't already exist to save results
    os.makedirs(
        "outputs/CatBoost",
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
    
    # Encode target
    label_encoder = LabelEncoder() # LabelEncoder is used to convert categorical target variable into numerical format that the model can understand
    y_encoded = label_encoder.fit_transform(y) # fit(y) learns all unique classes in the target variable and transform(y) converts the original target values into encoded numerical values
    print("\nClass Mapping:")
    for i, cls in enumerate(label_encoder.classes_): # Print the mapping of original class labels to encoded numerical values using loop to iterate through the classes and their corresponding indices
        print(f"{cls} -> {i}")
        
    # This identifies which features are categorical and need to be handled differently by CatBoost
    # CatBoost can automatically handle categorical features, but we need to specify which ones they are by providing their column indices
    # We check if the expected categorical features are present in the dataset and get their indices for CatBoost training
    # We only include the categorical features that are actually present in the dataset to avoid errors during training
    # Starts to create a list called categorical_features which contains the names of the expected categorical features and checks if they are in the dataset columns, then creates a list of their corresponding indices in the feature set X
    categorical_features = [
        col for col in [
            'TIME_OF_DAY',
            'HVAC_OPERATION_MODE',
            'AMBIENT_LIGHT_LEVEL'
        ]
        if col in X.columns
    ]
    
    # Get the indices of the categorical features for CatBoost
    # CatBoost requires the indices of the categorical features rather than their names, so we convert the column names to their corresponding indices in the feature set X
    # We use X.columns.get_loc(col) to get the index of each categorical feature column and create a list of these indices called cat_feature_indices
    # This allows us to tell CatBoost which features are categorical so it can handle them appropriately during training
    # Some machine learning libraries requrie the column index rather than the column name
    cat_feature_indices = [
        X.columns.get_loc(col)
        for col in categorical_features
    ]
    
    # ===================
    #  TRAIN-TEST SPLIT
    # ===================
    
    # Define the varaibles, X_train for training featurs, X_tests is for Testing features, y_train is for training labels, y_test is for Testing labels
    # Machine Learning Models should be trained on one set of data and evaluated on unseen data 
    X_train, X_test, y_train, y_test = train_test_split(
        X, # Passes the feature matrix into the function, x contains all predictor variables. The model needs these inputs to make predictions
        y_encoded, # Passes the target variable. These are the values the model is trying to predict. Without y, the model would not know the corract answers
        test_size=0.2, # Uses 20% of the data for testing, uses 80% of the data for testing. This is good for Catboost as it performs very well with moderate amounts of data. This usually provides Good learning capaiblity and reliable performance measurement
        random_state=42, # Sets the random seed, without this every run might create a different split. Once this is applied, every run gets exactly the same split
        stratify=y_encoded # Maintains the same class distribution in both training and testing sets. This is extremely important for classification problems.
    )
    
    # =================
    #  TRAIN CATBOOST
    # =================
    # Catboost is a gradient boosting algorithm developed by Yandex
    # 1. handles categorical features naturally 
    # 2. Requires less preprocessing
    # 3. Usually performas very well on tabular datasets
    # 4. Reduces overfitting compared to a single decision tree
    
    # This code creates and configures a CatBoost model before training it
    model = CatBoostClassifier(
        iterations=500, # Tells Catboost model to build 500 trees, 1 iteration = 1 tree. 500 iterations is a goood compromise between accuracy, training speed and overfitting risk
        learning_rate=0.05, # Controls how much each tree contributes. 0.05 is a good learning rate as it learns more cargull, reduce overfitting and usually generalize better
        depth=8, # Controls how deep each decison tree can grow. Deeper trees can learn complex relationships, feature interactions and non-linear patterns
        loss_function='MultiClass', # Tells CatBoost what type of problem you are solving. The target has multiple classes. Catboost must know how to calculate erros, for multiclass problems it uses multiclass loss calcualtions internally. without this, the model may use wrong objective
        eval_metric='Accuracy', # Determines which metric CatBoost reports during training. Accuracy is used for balanced multiclass datasets. Accuracy is also easy to understand
        random_seed=42, # Fixes the randomness used during training. machine learning algorithms contain randomness by sampling, tree construction and feature selection
        verbose=100 # Prints training process every 100 iterations, this allows us to monitor training progress, detect overfitting and ensure trainign has not stalled.
    )
    
    # This is where the actual training of th Catboost model begins. 
    # The Catboost starts training and learns relationships between X_train and y_train
    model.fit(
        X_train, # input variables
        y_train, # output variables, catboost tries to predict
        cat_features=cat_feature_indices, # Catboost can handle categories directly, one of catboost biggest advantages
        eval_set=(X_test, y_test) # Provides a validation dataset
    )
    
    # ==============
    #  PREDICTIONS
    # ==============
    
    # Uses the trained CatBoost model to predict the classes of the test dataset
    y_pred = model.predict(X_test)

    # Convert from shape (n,1) to (n,)
    # This peforms 2 operations. 
    # 1. flatten() -> converts a 2D array into a 1D array, this is done because many evaluation function requrie predictions in a flat array
    # 2. astype(int) -> Converts every value into an integer. Using integers avoids compatibility issues with evaluation metrics and reporting functions.
    y_pred = y_pred.flatten().astype(int)
    
    # =============
    #  EVALUATION
    # =============
    
    # Uses the y_test as actual labels and y_pred as predicted labels
    # These calculate the performance meterics of the Catboost model
    accuracy = accuracy_score(y_test, y_pred) # Calculates model's accuracy. Accuracy = Correct predictions / Total Predictions
    # Weighted averaging combines all the precisions, recall, f1 score values while considering class frequencies
    precision = precision_score(y_test, y_pred, average='weighted') # Calculates model precision. Precision = TP / TP + FP. 
    recall = recall_score(y_test, y_pred, average='weighted') # Calculates model recall. Recall = TP / TP + FN
    f1 = f1_score(y_test, y_pred, average='weighted') # Calculates model f1 score. Combines precision and recall. F1 = 2 x [(precision x recall) / (precision + recall)]
    # Prints out all the performance metric values
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("\nClassification Report:\n")
    
    # Prints detailed classification report
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=label_encoder.classes_ # Replaces numeric class labels with the original class names
        )
    )
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    print("\nConfusion Matrix:\n")
    print(cm)
    
    # ================
    #  VISUALIZATIONS
    # ================
    
    # Heatmap for the confusion Matrix
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
    
    # saves the confusion matrix picture
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
    # Save Feature Importance graph 
    plt.savefig(
        "outputs/CatBoost/feature_importance.png",
        bbox_inches="tight"
    )
    plt.close()

    print(feature_importance)
    
    # Save feature importance
    # Saves teh Dataframe to CSV file
    # index=False prevents pandas from saving row numbers
    feature_importance.to_csv(
        "outputs/CatBoost/feature_importance.csv",
        index=False
    )
    
    # Saves LabelEncoder object to disk. Store this so that we are able to reuse it later.
    joblib.dump(
        label_encoder,
        "outputs/CatBoost/label_encoder.pkl"
    )
    
    # Return all these values to the pipeline.py file
    return {
        "Model": "CatBoost",
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Estimator": model,
        "Encoder": label_encoder
    }

