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
    
    # =========================
    # START KNN TRAINING
    # =========================
    # This function trains a K-Nearest Neighbours model to predict ACTIVITY_LEVEL.
    # KNN was chosen because it is a simple supervised learning algorithm that classifies
    # new data points based on the activity levels of the most similar nearby data points.
    
    print("\nTraining KNN...")
    
    # Create a folder to save all KNN outputs such as graphs, feature importance,
    # and the trained model file. This keeps the project organised.
    os.makedirs(
        "outputs/KNN",
        exist_ok=True
    )
    
    # =========================
    # DATA PREPARATION
    # =========================
    
    # ACTIVITY_LEVEL is the target variable because this is what the model is trying to predict.
    # The model will learn patterns from the sensor readings and environmental conditions
    # to classify whether the activity level is low, moderate, or high.
    y = df["ACTIVITY_LEVEL"]

    # Remove ACTIVITY_LEVEL from the input features because it is the answer/output.
    # SESSION_ID is also removed because it is only an identifier for each session.
    # Keeping SESSION_ID may cause the model to memorise session numbers instead of learning
    # real patterns from the sensor data.
    X = df.drop(
        columns=["ACTIVITY_LEVEL", "SESSION_ID"],
        errors="ignore"
    )

    # CO_GASSENSOR contains values from 1 to 4.
    # Although these values are numbers, they represent categories/levels rather than
    # a true continuous measurement. Therefore, it is converted to string so that
    # one-hot encoding treats it as a categorical feature.
    X["CO_GASSENSOR"] = X["CO_GASSENSOR"].astype(str)
    
    # Convert categorical variables into numerical dummy variables.
    # Machine learning models cannot directly process text categories, so one-hot encoding
    # creates separate columns for each category.
    #
    # drop_first=False is used because KNN is distance-based, and keeping all categories
    # makes the category representation clearer for the model.
    X = pd.get_dummies(X, drop_first=False)
    
    # =========================
    # TRAIN-TEST SPLIT
    # =========================
    
    # Split the data into training and testing sets.
    # The training set is used to teach the model patterns.
    # The testing set is used to evaluate how well the model performs on unseen data.
    #
    # test_size=0.2 means 80% of the data is used for training and 20% is used for testing.
    # random_state=42 ensures that the split is reproducible, meaning the same result
    # can be obtained when the code is run again.
    #
    # stratify=y is important because the activity level classes are imbalanced.
    # It keeps the percentage of low, moderate, and high activity similar in both
    # the training and testing datasets.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    # =========================
    # KNN HYPERPARAMETER TUNING
    # =========================
    
    # KNN calculates distances between data points.
    # Therefore, scaling is required because features with larger values can dominate
    # the distance calculation if the data is not standardised.
    #
    # A Pipeline is used to make sure scaling happens before KNN training.
    # This also prevents data leakage because the scaler is fitted only on the training data
    # during cross-validation.
    knn_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier())
    ])

    # This grid contains the KNN settings that will be tested.
    #
    # n_neighbors controls how many nearby points are used to make a prediction.
    # A smaller value may overfit, while a larger value may smooth the predictions too much.
    #
    # weights controls how neighbours contribute:
    # - "uniform" means all neighbours have equal influence.
    # - "distance" means closer neighbours have more influence.
    #
    # p controls the distance formula:
    # - p=1 uses Manhattan distance.
    # - p=2 uses Euclidean distance.
    param_grid = {
        "knn__n_neighbors": list(range(3, 31)),
        "knn__weights": ["uniform", "distance"],
        "knn__p": [1, 2]
    }

    # GridSearchCV tests every combination of hyperparameters in param_grid.
    # cv=5 means 5-fold cross-validation is used, where the training data is split
    # into 5 parts and the model is trained/tested multiple times.
    #
    # This gives a more reliable result compared to testing only one set of parameters.
    # scoring="accuracy" means the best model is selected based on the highest accuracy.
    # n_jobs=-1 allows the computer to use all available processors to speed up the search.
    grid_search = GridSearchCV(
        knn_pipeline,
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    # Train and test all parameter combinations using the training data only.
    # The test data is kept separate until the final evaluation.
    grid_search.fit(X_train, y_train)
    
    # =========================
    # MODEL EVALUATION
    # =========================
    
    # Get the best KNN model found by GridSearchCV.
    # This model contains the best number of neighbours, best weighting method,
    # and best distance metric based on cross-validation.
    best_knn_model = grid_search.best_estimator_

    # Use the best model to predict activity levels on the unseen test set.
    y_pred_knn = best_knn_model.predict(X_test)

    # Calculate evaluation metrics to measure model performance.
    #
    # Accuracy shows the overall percentage of correct predictions.
    # Precision shows how many predicted classes were actually correct.
    # Recall shows how many actual classes were correctly identified.
    # F1 score balances precision and recall.
    #
    # average="weighted" is used because the dataset is imbalanced.
    # This means the metric considers the number of samples in each activity class.
    accuracy = accuracy_score(y_test, y_pred_knn)
    precision = precision_score(y_test, y_pred_knn, average="weighted")
    recall = recall_score(y_test, y_pred_knn, average="weighted")
    f1 = f1_score(y_test, y_pred_knn, average="weighted")
    
    # Display the evaluation results.
    # These values can be used to compare KNN with other models such as Decision Tree,
    # Gradient Boosting, or XGBoost.
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    # =========================
    # CONFUSION MATRIX
    # =========================
    
    # The confusion matrix shows which activity levels were predicted correctly
    # and which activity levels were confused with other classes.
    #
    # Rows represent the actual activity level.
    # Columns represent the predicted activity level.
    cm = confusion_matrix(y_test, y_pred_knn)

    plt.figure(figsize=(6, 4))
    
    # Plot the confusion matrix as a heatmap so it is easier to interpret visually.
    # This helps identify whether the model is better at predicting certain classes
    # and weaker at predicting others.
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
    
    # Save the confusion matrix image into the KNN output folder.
    # This can be used later in the report or presentation.
    plt.savefig(
        "outputs/KNN/confusion_matrix.png",
        bbox_inches="tight"
    )

    # Close the plot to avoid displaying or overlapping with other graphs.
    plt.close()
    
    # =========================
    # FEATURE IMPORTANCE
    # =========================
    
    # KNN does not naturally provide feature importance like a Decision Tree.
    # Therefore, permutation importance is used.
    #
    # Permutation importance works by randomly shuffling one feature at a time
    # and checking how much the model accuracy decreases.
    # If accuracy drops a lot, it means that feature was important for prediction.
    perm_importance = permutation_importance(
        best_knn_model,
        X_test,
        y_test,
        n_repeats=10,
        random_state=42,
        scoring="accuracy",
        n_jobs=-1
    )

    # Store each feature name together with its importance score in a DataFrame.
    # This makes it easier to sort, view, save, and plot the results.
    feature_importance = pd.DataFrame({
        "Feature": X_test.columns,
        "Importance": perm_importance.importances_mean
    })

    # Sort the features from most important to least important.
    # The highest values show which features affected the KNN model the most.
    feature_importance = feature_importance.sort_values(
        by="Importance",
        ascending=False
    )
    
    # Save the feature importance results as a CSV file.
    # This allows the results to be reused in the report or model comparison table.
    feature_importance.to_csv(
        "outputs/KNN/feature_importance.csv",
        index=False
    )

    print("\nKNN Feature Importance:")
    print(feature_importance.head(15))

    # =========================
    # FEATURE IMPORTANCE VISUALIZATION
    # =========================

    plt.figure(figsize=(10, 6))

    # Plot the top 15 most important features.
    # This visual helps explain which sensor or environmental readings influenced
    # the KNN model predictions the most.
    sns.barplot(
        data=feature_importance.head(15),
        x="Importance",
        y="Feature"
    )

    plt.title("Top 15 Feature Importance for KNN")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    
    # Save the feature importance chart into the KNN output folder.
    plt.savefig(
        "outputs/KNN/feature_importance.png",
        bbox_inches="tight"
    )
    
    # Close the plot after saving.
    plt.close()
    
    # =========================
    # SAVE TRAINED MODEL
    # =========================
    
    # Save the trained KNN model as a .pkl file.
    # This allows the model to be loaded again later without retraining.
    # It is useful for future predictions or deployment.
    joblib.dump(
        best_knn_model,
        "outputs/KNN/knn_model.pkl"
    )

    # Return the model results in dictionary format.
    # This makes it easier to combine KNN results with other machine learning models
    # in a comparison table.
    return {
        "Model": "KNN",
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Estimator": best_knn_model,
        "Encoder": None
    }