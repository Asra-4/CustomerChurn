Customer Churn Analysis Project

Overview

This project focuses on analyzing customer churn data and building machine learning models to predict customer churn. The process includes data preprocessing, feature engineering, exploratory data analysis, and implementing various machine learning algorithms for classification.

Dataset

The dataset contains customer information, including demographics, tenure, monthly charges, and contract types. The target variable is Churn, which indicates whether a customer has left the service (Yes or No).

Project Workflow

1. Data Loading and Exploration

Code:

import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('customer_churn_data.csv')
df.head()
df.info()
df.isna().sum().sum()

Explanation:
The dataset is loaded using Pandas, and basic exploration is conducted to check the structure and presence of missing or duplicate values.
2. Handling Missing Values

Code:

df["InternetService"] = df["InternetService"].fillna("")

Explanation:
Missing values in the InternetService column are filled with an empty string.

3. Exploratory Data Analysis (EDA)

Code Examples:

Correlation analysis of numeric columns:

numeric_columns_data = df.select_dtypes(include=["number"])
numeric_columns_data.corr()

Churn distribution:

df["Churn"].value_counts().plot(kind="pie")
plt.title("Churn (Y/N)")
plt.ylabel("")
plt.show()

Monthly charges grouped by contract type:

df.groupby("ContractType")["MonthlyCharges"].mean().plot(kind="bar")
plt.title("Contract Type by Average Pricing")
plt.show()

Explanation:
EDA helps uncover patterns, relationships, and insights from the data. Various visualizations and grouping operations are used to explore the data.

4. Feature Engineering

Code:

y = df[['Churn']]
x = df[['Age', 'Gender', 'Tenure', 'MonthlyCharges']]
x['Gender'] = x['Gender'].apply(lambda x: 1 if x == "Female" else 0)
y['Churn'] = y['Churn'].apply(lambda x: 1 if x == "Yes" else 0)

Explanation:
The categorical Gender column is converted to binary, and the target variable Churn is encoded as 0 or 1.

5. Data Splitting and Scaling

Code:

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
joblib.dump(scaler, "scaler.pkl")
scaled_x_test = scaler.transform(x_test)

Explanation:
The dataset is split into training and testing sets. Features are scaled for better performance of machine learning models.

6. Model Training and Evaluation

Logistic Regression

Code:

from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression()
log_model.fit(x_train, y_train)
y_pred = log_model.predict(x_test)

Explanation:
Logistic Regression is trained and used to predict churn.

K-Nearest Neighbors (KNN)

Code:

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

param_grid = {"n_neighbors": [3, 5, 7, 9], "weights": ["uniform", "distance"]}
gridkn = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
gridkn.fit(x_train, y_train)

Explanation:
GridSearchCV is used to tune hyperparameters of the KNN classifier.

Support Vector Machines (SVM)

Code:

from sklearn.svm import SVC

param_grid = {"C": [0.01, 0.1, 0.5, 1], "kernel": ["linear", "rbf", "poly"]}
gridsvc = GridSearchCV(SVC(), param_grid, cv=5)
gridsvc.fit(x_train, y_train)

Explanation:
SVM with different kernels is trained, and the best hyperparameters are selected.

Decision Tree Classifier

Code:

from sklearn.tree import DecisionTreeClassifier

param_grid = {
    "criterion": ["gini", "entropy"],
    "splitter": ["best", "random"],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}
grid_tree = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)
grid_tree.fit(x_train, y_train)

Explanation:
Decision Tree Classifier is optimized using GridSearchCV.

Random Forest Classifier

Code:

from sklearn.ensemble import RandomForestClassifier

param_grid = {"n_estimators": [32, 64, 128, 256]}
grid_rfc = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid_rfc.fit(x_train, y_train)

Explanation:
Random Forest Classifier is tuned for the best performance.

7. Model Evaluation

Code:

from sklearn.metrics import accuracy_score

def modelperformance(predictions):
    print("Accuracy score on model is {}".format(accuracy_score(y_test, predictions)))

Explanation:
The function modelperformance evaluates the accuracy of the predictions.

8. Saving the Best Model

Code:

best_model = gridsvc.best_estimator_
joblib.dump(best_model, "model.pkl")

Explanation:
The best-performing model is saved as a .pkl file for future use.

Key Insights

Customers with longer contract durations tend to have lower monthly charges.

Tenure has a significant impact on churn likelihood.

Gender distribution does not significantly affect churn.

Future Improvements

Incorporate additional features for better predictive power.

Experiment with advanced models like Gradient Boosting or Neural Networks.

Perform feature selection to reduce dimensionality.

Requirements

Python 3.x

Libraries: pandas, matplotlib, sklearn, joblib

Usage

Place the customer_churn_data.csv in the project directory.

Run the notebook or script to train and evaluate models.

Use the saved model.pkl for predictions on new data.

