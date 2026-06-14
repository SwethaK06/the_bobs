# Project Title
Activity levels in Elderly Homes

# Group name: The Bobs

# Who wrote what .py fles
Swetha -> Catboost.py, XGBoost.py, pipeline.py
Prerana -> Decision_Tree.py, KNN.py
Qianhe -> Gradient_boosting.py, Random_Forest.py

Swetha, Prerana, Qianhe -> Cleaning.py

# Contributions of team members
Swetha
- Assisted with the cleaning of the dataset
- Did the catboost.py model
- Did teh XGBoost.py model
- Combined everyone's model and cleaning into 1 working pipeline
- Containerized the application using docker

Prerana
- Assisted with the cleaning of dataset
- Did the Decision_Tree.py model
- Did the KNN.py model
- Did EDA

Qianhe
- Assisted with the cleaning of the dataset
- Did the Gradient Boosing model
- Did the Random forest model
- Did EDA

# Instructions on how to run the pipeline
1) Install the relevant libraries for the code to run
    - pandas
    - numpy
    - matplotlib
    - seaborn
    - scikit-learn
    - xgboost
    - catboost
    - joblib

2) Make sure you are in the project folder

3) Run the command on the terminal to run the pipeline file 
    python pipeline.py

This would run the cleaning file plus all the models.

# Instructions on how to start your docker environment 
1) Open your ubuntu terminal and make sure you are in the project folder directory

2) Run this command to build the image 
    docker build -t gas-monitoring .

3) Run the container once the image is built
    docker run --name gas-container gas-monitoring

4) To check your docker images run 
    docker images

5) To check your containers run 
    docker ps -a

# Summary of key finding of EDA
The exploratory data analysis identified several key patterns within the dataset. Correlation analysis showed that the **CO2_ElectroChemicalSensor** and the four **MetalOxideSensor** units were moderately to strongly correlated, indicating that these sensors respond similarly to changes in indoor environmental conditions. In contrast, the two CO₂ sensors showed little correlation with each other, suggesting differences in sensor technology, measurement units, or placement. Analysis of the target variable revealed a class imbalance, with **low activity** being the most common class and **high activity** the least common, which may affect the model's ability to accurately classify high activity instances.

Temperature analysis showed that low activity periods had the highest average temperatures, possibly due to reduced HVAC usage, while high activity periods also recorded relatively high temperatures due to increased occupancy and movement. Ambient light levels were predominantly **very bright** throughout the day, indicating consistently well-lit environments. However, when comparing light levels across activity categories, similar distributions were observed, suggesting that ambient light is not a strong indicator of activity level. Additionally, CO gas readings remained relatively consistent across different HVAC operation modes, indicating that HVAC settings may have limited influence on CO concentrations. Overall, the findings suggest that the gas sensor measurements, particularly the electrochemical CO₂ and metal oxide sensors, are likely to be the most useful features for activity-level classification.

# Explain and justify features that are engineered
- Catboost
Several feature engineering and preprocessing steps were performed to prepare the dataset for model training. First, the **SESSION_ID** column was removed because it functions only as a unique identifier and does not contain meaningful information related to activity levels. Retaining such an identifier could cause the model to learn irrelevant patterns and reduce its ability to generalize to unseen data. Next, the dataset was separated into **features (X)** and the **target variable (y)**, where **ACTIVITY_LEVEL** was designated as the target because it represents the outcome that the model is intended to predict. Since machine learning algorithms cannot directly process textual class labels, the target variable was transformed into numerical values using **Label Encoding**, ensuring that each activity category could be interpreted by the model while preserving the distinct class labels. Finally, the categorical features **TIME_OF_DAY**, **HVAC_OPERATION_MODE**, and **AMBIENT_LIGHT_LEVEL** were identified and their corresponding column indices were extracted. This step was necessary because CatBoost requires categorical features to be explicitly specified so that it can apply its specialised handling of categorical data. By identifying these features, the model is able to utilise both numerical sensor readings and categorical contextual information effectively, improving its ability to classify activity levels accurately.


- XGBoost
Several feature engineering and preprocessing steps were applied to prepare the dataset for XGBoost training. First, the **SESSION_ID** column was removed because it serves only as a unique identifier and does not contain meaningful information related to activity levels. Including such an identifier could cause the model to learn irrelevant patterns and negatively affect its ability to generalize to new data. The dataset was then separated into **features (X)** and the **target variable (y)**, where **ACTIVITY_LEVEL** was designated as the target because it represents the outcome that the model is intended to predict. Since XGBoost cannot directly process categorical text values, the categorical features **TIME_OF_DAY**, **HVAC_OPERATION_MODE**, and **AMBIENT_LIGHT_LEVEL** were transformed using **One-Hot Encoding**. This technique converts each category into a binary (0 or 1) feature, allowing the model to utilise categorical information without introducing artificial numerical relationships between categories. The parameter `drop_first=True` was used to remove one category from each encoded feature, reducing redundancy and avoiding unnecessary duplication of information. Finally, the target variable was converted into numerical labels using **Label Encoding**, enabling XGBoost to process the activity-level classes during training. These preprocessing steps ensure that all input features are represented in a format suitable for XGBoost while preserving the information necessary for accurate activity-level classification.

- Decision Tree
The same preprocessing steps were applied to the Decision Tree model. The *SESSION_ID* column was removed, the dataset was separated into features and the target variable (*ACTIVITY_LEVEL), and **CO_GASSENSOR* was treated as a categorical feature using one-hot encoding. A stratified train-test split was also used to maintain the class distribution of activity levels in both the training and testing datasets, ensuring a fair evaluation of model performance.

- KNN
Before training the KNN model, the *SESSION_ID* column was removed because it is only an identifier and does not contribute to predicting activity levels. The dataset was then split into features (*X) and the target variable (ACTIVITY_LEVEL). The **CO_GASSENSOR* feature was converted into a categorical variable and one-hot encoded together with other categorical features so that they could be processed by the model. A stratified train-test split was used to preserve the class distribution of activity levels, and *StandardScaler* was applied to standardise numerical features since KNN relies on distance calculations and is sensitive to differences in feature scales.


- Gradient Boosting
For feature engineering for the gradient boosting model, I dropped the session ID column since it has no relevance to the target variable and will only confuse the model. I also used an ordinal encoder to convert all categorical features to numbers so that the model can read the dataset. For the target variable, I used a label encoder to convert the activity level from categorical to numbers so the model can read. The reason I used a different encoder for the target variable is so it would be easier to convert back into categorical values (the original value) if we were to implement asking the model to predict the activity level after the user inputs their values.

- Random Forest
For feature engineering for the Random Forest model, I dropped the session ID column as it is irrelevant to the target variable we are trying to predict and will only confuse the model. I used one hot encoding to convert all categorical features into numerical ones that the model can understand. 


# Explanation of choice of models and justify any tuning methods used
- Catboost
*Why we chose the model*
CatBoost (Categorical Boosting) was selected because it is specifically designed to handle datasets containing both numerical and categorical features. In this project, variables such as TIME_OF_DAY, HVAC_OPERATION_MODE, and AMBIENT_LIGHT_LEVEL are categorical in nature, making CatBoost particularly well-suited for the task. Unlike many machine learning algorithms that require extensive preprocessing and one-hot encoding of categorical variables, CatBoost can process categorical features directly while preserving their relationships within the data. This reduces the risk of information loss and simplifies the preprocessing pipeline. Furthermore, CatBoost employs an ordered boosting technique that helps reduce overfitting and improve prediction accuracy. Its ability to effectively combine numerical sensor readings with categorical contextual information makes it a strong candidate for classifying activity levels into low, moderate, and high activity categories.

*Tuning methods used*
The CatBoost model was configured with 500 iterations, a learning rate of 0.05, and a maximum tree depth of 8 to balance predictive performance and generalization. The MultiClass loss function was selected because the target variable consists of three activity-level categories. Accuracy was used as the evaluation metric to measure classification performance, while a fixed random seed of 42 ensured reproducibility of results. CatBoost's ability to directly process categorical variables was leveraged by specifying the categorical feature indices, reducing the need for additional preprocessing. An evaluation dataset was also provided during training to monitor model performance on unseen data and help detect overfitting.

- XGBoost
*Why we chose the model*
XGBoost (Extreme Gradient Boosting) was selected because it is a powerful ensemble machine learning algorithm that builds multiple decision trees sequentially, with each new tree correcting the errors of the previous ones. This boosting approach enables XGBoost to capture complex and non-linear relationships between environmental sensor readings and activity levels. The model is particularly effective for structured tabular datasets, such as the gas monitoring dataset used in this project, and is capable of handling interactions between multiple features. Additionally, XGBoost includes built-in regularization techniques that help reduce overfitting, improving the model's ability to generalize to unseen data. Its high predictive performance and ability to provide feature importance rankings make it a suitable choice for multiclass activity-level classification.

*Tuning methods used*
The XGBoost model was configured with 800 estimators and a learning rate of 0.02 to enable gradual learning and improve predictive accuracy. A maximum tree depth of 5 was selected to balance model complexity and generalization, while a minimum child weight of 3 helped prevent splits based on small and potentially noisy subsets of data. To reduce overfitting, subsampling (subsample=0.8) and feature sampling (colsample_bytree=0.8) were applied, introducing randomness into the training process and increasing tree diversity. Additionally, a gamma value of 0.1 was used to ensure that only meaningful splits were created. A fixed random seed (random_state=42) ensured reproducibility, and multiclass logarithmic loss (mlogloss) was chosen as the evaluation metric because the task involves classifying observations into three activity-level categories.

- Decision Tree
*Why we chose this model*
Decision Tree was chosen because it is easy to interpret and can model complex, non-linear relationships between environmental variables and activity levels. The model creates a tree-like structure of decision rules, making it useful for understanding how different sensor readings influence predictions. It also provides feature importance scores, allowing the identification of the most influential variables in the dataset.

*Tuning methods used*
GridSearchCV was used to tune the Decision Tree by testing different values for *criterion* (Gini and entropy), *max_depth, **min_samples_split, **min_samples_leaf, and **class_weight. These parameters were selected because they affect the complexity of the tree and help prevent overfitting. The **f1_weighted* score was used during optimisation because it accounts for class imbalance while considering both precision and recall.

- KNN
*Why we chose this model*
KNN was chosen because it classifies data based on similarity between observations, making it suitable for predicting activity levels using environmental sensor readings such as temperature, humidity, CO₂ levels, HVAC mode, and ambient light levels. Since similar environmental conditions are likely to correspond to similar activity levels, KNN can effectively identify patterns in the sensor data without making assumptions about the underlying relationships between variables.

*Tuning methods used*
GridSearchCV was used to optimise the KNN model by testing different values of *n_neighbors, weighting methods (uniform* and *distance), and distance metrics (Manhattan* and *Euclidean). These parameters were tuned because they directly influence how neighbouring observations are selected and weighted during classification. A pipeline containing **StandardScaler* and KNN was used together with five-fold cross-validation to ensure reliable model evaluation and selection of the best-performing hyperparameters.

- Random Forest
*Why we chose this model*
I selected a random forest model for this since it is effective for datasets with a combination of numerical sensor readings and categorical variables. Random forest is able to capture these complex relationships by building a large number of decision trees and combining their predictions to produce a final result. It reduces the risk of overfitting compared to a single decision tree and can handle large numbers of features. 

*Tuning methods used*
I used hyperparameter tuning to tune my random forest model to ensure that it generalises effectively on unseen data. Adjusting key hyperparameters can improve accuracy and reduce prediction errors. To do this, I used gridsearch and cross validation to run multiple iterations of the model using different hyperparameters each time to find the most optimal one which is the one that outputs highest accuracy. I then used that hyperparameter and inputted into my model

- Gradient Boosting
*Why we chose this model*
I chose gradient boosting since the dataset contains a mixture of readings which are likely to have complex and non-linear relationships with the target variable which is activity level. Gradient boosting can effectively capture these complex patterns by building multiple trees sequentially where each one learns from the errors made by the previous one. This learning also allows the model to gradually improve its performance. The model can also automatically identify interactions between features, simplifying the process

*Tuning methods used*
I used hyperparameter tuning to tune this model so as to ensure it generalises well to unseen data. The default parameters of the gradient boosting model may not be suitable for this specific dataset so I had to change the hyperparameters around to get the optimal ones. To do this, I used gridsearch and cross validation to run multiple iterations of the model with varying hyperparameters to find the one that gives the highest accuracy.

# Explain any specific choice of metrics that are important to the problem statement
Accuracy is a suitable metric for this project because the primary objective stated in the problem statement is to correctly classify the activity level of elderly residents into the categories of low, moderate, and high activity. Since the goal is to determine how accurately the model can identify the correct activity category based on environmental sensor readings and indoor air quality data, accuracy provides a direct measure of the proportion of correctly classified instances across all activity levels. A higher accuracy indicates that the model is more effective at identifying residents' activity levels, which supports the development of a reliable early warning system capable of detecting situations that may require attention from caregivers or healthcare providers.

Furthermore, the problem statement specifically requires the development and evaluation of multiple prediction models to determine which model performs best at categorising activity levels. Accuracy is particularly useful for this comparison because it provides a simple and easily interpretable measure of overall model performance. By calculating the percentage of correct predictions made by each model, it becomes straightforward to compare models such as XGBoost and CatBoost and identify the one that most effectively classifies activity levels. Since all models are evaluated on the same dataset, accuracy offers a consistent benchmark for comparison.

Although the dataset exhibits some class imbalance, the objective of the project is not to prioritise one activity category over another but rather to maximise the overall correctness of activity-level predictions. Therefore, accuracy remains an appropriate primary evaluation metric because it reflects the model's overall ability to classify residents into the correct activity categories. In addition, other metrics such as precision, recall, and F1-score can be used as supporting measures to provide further insight into performance on individual classes, particularly the minority high-activity class. However, accuracy is the most suitable metric for selecting the best overall model because it aligns directly with the project's objective of achieving the highest proportion of correct activity-level predictions.