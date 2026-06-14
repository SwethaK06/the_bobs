from models.Decision_Tree import train_decision_tree
from models.Gradient_Boosting import train_gradient_boosting
from models.KNN import train_knn
from models.RandomForest import train_random_forest
from preprocessing.Cleaning import clean_data
from models.Catboost import train_catboost
from models.XGBoost import train_xgboost

import pandas as pd
import os


def main():

    df = clean_data()

    results = []

    # Train and evaluate each model, storing results for comparison
    results.append(train_catboost(df)) # Trains the catboost model and saves the outputs in results
    results.append(train_xgboost(df)) # Trains XGBoost model and saves the outputs in results
    results.append(train_decision_tree(df)) # Trains the decision tree model and saves the outputs in results
    results.append(train_gradient_boosting(df)) # Trains the gradient boost model and saves the outputs in results
    results.append(train_random_forest(df)) # Trains the random forest model and saves the outputs in results
    results.append(train_knn(df)) # Trains the KNN model and saves the outputs in results
 
    
    # Create comparison table
    comparison_df = pd.DataFrame([
        {
            "Model": r["Model"],
            "Accuracy": r["Accuracy"],
            "Precision": r["Precision"],
            "Recall": r["Recall"],
            "F1": r["F1"]
        }
        for r in results
    ])

    # Sort by accuracy
    comparison_df = comparison_df.sort_values(
        by="Accuracy",
        ascending=False
    )
    
    # Create outputs directory if it doesn't exist
    os.makedirs(
        "outputs",
        exist_ok=True
    )
    
    # Save model comparison results to CSV
    comparison_df.to_csv(
        "outputs/model_comparison.csv",
        index=False
    )

    print("\nMODEL COMPARISON")
    print("=" * 50)
    print(comparison_df)

    # Best model
    best_model = comparison_df.iloc[0]
    
    # Display best model information
    print("\nBEST MODEL")
    print("=" * 50)
    print(f"Model    : {best_model['Model']}")
    print(f"Accuracy : {best_model['Accuracy']:.4f}")
    print(f"Precision: {best_model['Precision']:.4f}")
    print(f"Recall   : {best_model['Recall']:.4f}")
    print(f"F1 Score : {best_model['F1']:.4f}")
    
    print("\nModel rankings:")
    
    # Print model rankings based on accuracy
    for i, row in comparison_df.iterrows():
        print(
            f"{row['Model']} -> "
            f"Accuracy: {row['Accuracy']:.4f}"
        )

if __name__ == "__main__":
    main()