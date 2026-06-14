import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import os 
import joblib 

from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)
from sklearn.model_selection import GridSearchCV


def train_decision_tree(df):
    
    # =========================
    # START DECISION TREE TRAINING
    # =========================
    # This function trains a Decision Tree model to predict ACTIVITY_LEVEL.
    # A Decision Tree was chosen because it is easy to understand and explain.
    # It makes predictions by splitting the dataset based on feature values,
    # similar to a flowchart of questions.
    
    print("\nTraining Decision Tree...")
    
    # Create a folder to store all Decision Tree outputs.
    # This keeps the confusion matrix, feature importance chart, CSV file,
    # and trained model organised in one location.
    os.makedirs(
        "outputs/DecisionTree",
        exist_ok=True
    )
    
    # =========================
    # PREPROCESSING
    # =========================
    
    # Remove ACTIVITY_LEVEL because it is the target/output that the model must predict.
    # SESSION_ID is removed because it is only an identifier for each session.
    # Keeping SESSION_ID may cause the model to memorise session numbers instead of
    # learning useful patterns from the sensor data.
    X = df.drop(
        columns=["ACTIVITY_LEVEL", "SESSION_ID"],
        errors="ignore"
    )
    
    # CO_GASSENSOR contains values from 1 to 4.
    # Even though these values are numbers, they represent categories/levels rather than
    # a continuous measurement. Therefore, it is converted to string so that it will be
    # treated as a categorical feature during one-hot encoding.
    X["CO_GASSENSOR"] = X["CO_GASSENSOR"].astype(str)
    
    # Convert categorical columns into numerical dummy variables.
    # Machine learning models require numerical input, so one-hot encoding creates
    # separate columns for each category.
    #
    # drop_first=False is used so that all categories are kept and represented clearly.
    X = pd.get_dummies(X, drop_first=False)
    
    # ACTIVITY_LEVEL is the target variable because this is what the model is trying to predict.
    # The model uses the sensor readings and environmental features to classify the activity
    # level as low, moderate, or high.
    y = df["ACTIVITY_LEVEL"]
    
    # =========================
    # TRAIN-TEST SPLIT
    # =========================
    
    # Split the dataset into training and testing sets.
    # The training set is used to train the model.
    # The testing set is used to evaluate how well the model performs on unseen data.
    #
    # test_size=0.2 means 80% of the data is used for training and 20% is used for testing.
    # random_state=42 makes the split reproducible, so the same result can be obtained
    # every time the code is run.
    #
    # stratify=y is used because the activity level classes are imbalanced.
    # It keeps the class proportions similar in both training and testing sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    # =========================
    # DECISION TREE HYPERPARAMETER TUNING
    # =========================
    
    # Create the base Decision Tree model.
    # random_state=42 ensures the tree can be reproduced consistently.
    dt = DecisionTreeClassifier(random_state=42)

    # This grid contains different Decision Tree settings to test.
    #
    # criterion controls how the tree decides the best split:
    # - "gini" measures impurity using Gini index.
    # - "entropy" measures impurity using information gain.
    #
    # max_depth limits how deep the tree can grow.
    # A very deep tree may overfit, while a shallow tree may underfit.
    #
    # min_samples_split controls the minimum number of samples needed to split a node.
    # Increasing this value can reduce overfitting.
    #
    # min_samples_leaf controls the minimum number of samples required in each leaf node.
    # This also helps prevent the tree from becoming too specific to the training data.
    #
    # class_weight is included because the dataset is imbalanced.
    # "balanced" gives more importance to minority classes such as high activity.
    param_grid = {
        "criterion": ["gini", "entropy"],
        "max_depth": [3, 5, 7, 10, 15, None],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 5, 10],
        "class_weight": [None, "balanced"]
    }

    # GridSearchCV tests all combinations of the parameters above.
    #
    # cv=5 means 5-fold cross-validation is used.
    # The training data is split into 5 parts, and the model is trained and tested
    # multiple times to get a more reliable performance estimate.
    #
    # scoring="f1_weighted" is used instead of only accuracy because the dataset is imbalanced.
    # Weighted F1 considers precision and recall while also accounting for class size.
    #
    # n_jobs=-1 uses all available processors to speed up the search.
    grid_search = GridSearchCV(
        dt,
        param_grid,
        cv=5,
        scoring="f1_weighted",
        n_jobs=-1
    )

    # Train and evaluate all hyperparameter combinations using only the training data.
    # The test data is kept separate for final evaluation.
    grid_search.fit(X_train, y_train)
    
    # =========================
    # MODEL TRAINING
    # =========================
    
    # Retrieve the best-performing Decision Tree model found by GridSearchCV.
    # This model has the best combination of depth, split rules, leaf size,
    # criterion, and class weight based on cross-validation.
    best_dt_model = grid_search.best_estimator_
    
    # =========================
    # PREDICTION
    # =========================
    
    # Use the best model to predict activity levels on the unseen test data.
    # This checks how well the model generalises to new data.
    y_pred_dt = best_dt_model.predict(X_test)
    print("Predictions completed.")
    
    # =========================
    # MODEL EVALUATION
    # =========================
    
    # Calculate evaluation metrics.
    #
    # Accuracy shows the overall percentage of correct predictions.
    # Precision shows how many predicted classes were actually correct.
    # Recall shows how many actual classes were correctly identified.
    # F1-score balances precision and recall.
    #
    # average="weighted" is used because the activity level classes are imbalanced.
    # This means the metric gives more weight to classes with more samples while still
    # including performance across all classes.
    accuracy = accuracy_score(y_test, y_pred_dt)
    precision = precision_score(y_test, y_pred_dt, average="weighted")
    recall = recall_score(y_test, y_pred_dt, average="weighted")
    f1 = f1_score(y_test, y_pred_dt, average="weighted")
    
    # Display the overall model performance.
    # These values can be compared with other models such as KNN, XGBoost,
    # Gradient Boosting, or CatBoost.
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Display a detailed classification report.
    # This shows precision, recall, and F1-score for each activity level separately.
    # It is useful because accuracy alone may hide poor performance on minority classes.
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred_dt))
    
    # =========================
    # CONFUSION MATRIX
    # =========================
    
    # Create a confusion matrix to compare actual and predicted activity levels.
    #
    # Rows represent the actual activity level.
    # Columns represent the predicted activity level.
    #
    # This helps identify which classes the model predicts correctly and which classes
    # are commonly confused with each other.
    cm = confusion_matrix(y_test, y_pred_dt)

    plt.figure(figsize=(6, 4))
    
    # Plot the confusion matrix as a heatmap for easier visual interpretation.
    # Darker or larger values show where more predictions were made.
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
    
    # Save the confusion matrix image so it can be used in the report or presentation.
    plt.savefig(
        "outputs/DecisionTree/confusion_matrix.png",
        bbox_inches="tight"
    )

    # Close the plot to prevent it from overlapping with other graphs.
    plt.close()
    
    # =========================
    # FEATURE IMPORTANCE
    # =========================
    
    # Decision Tree models naturally provide feature importance scores.
    # These scores show how much each feature contributed to the tree's decisions.
    #
    # Higher importance means the feature was more useful for splitting the data
    # and predicting the activity level.
    feature_importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": best_dt_model.feature_importances_
    })
    
    # Save the raw feature importance scores into a CSV file.
    # This allows the results to be reused later in reports or model comparison.
    feature_importance.to_csv(
        "outputs/DecisionTree/feature_importance.csv",
        index=False
    )

    # Sort the features from most important to least important.
    feature_importance = feature_importance.sort_values(
        by="Importance",
        ascending=False
    )
    
    # Select the top 10 most important features for the graph.
    # Showing only the top 10 keeps the chart clear and easy to explain.
    top_features = feature_importance.head(10)

    plt.figure(figsize=(10, 5))
    
    # Create a horizontal bar chart to display the top feature importance scores.
    plt.barh(top_features["Feature"], top_features["Importance"])
    
    # Invert the y-axis so the most important feature appears at the top.
    plt.gca().invert_yaxis()
    
    plt.title("Top 10 Important Features - Decision Tree")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    
    # Save the feature importance chart as an image.
    plt.savefig(
        "outputs/DecisionTree/feature_importance.png",
        bbox_inches="tight"
    )
    
    # Close the plot after saving.
    plt.close()
    
    # =========================
    # SAVE TRAINED MODEL
    # =========================
    
    # Save the trained Decision Tree model as a .pkl file.
    # This allows the model to be loaded again later without retraining.
    # It is useful for future predictions, deployment, or comparison.
    joblib.dump(
        best_dt_model,
        "outputs/DecisionTree/decision_tree_model.pkl"
    )

    # Return the model performance metrics and trained estimator in dictionary format.
    # This makes it easier to combine Decision Tree results with other model results
    # in one comparison table.
    return {
        "Model": "Decision Tree",
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Estimator": best_dt_model,
        "Encoder": None
    }