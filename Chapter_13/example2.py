# Example 13.2 - RBM[LogisticRegression]
import numpy as np
from sklearn.datasets import load_digits
from sklearn import datasets
from sklearn.neural_network import BernoulliRBM
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report

import matplotlib.pyplot as plt

# Load a dataset (for this example, we'll use the digits dataset)
digits = datasets.load_digits()
X = digits.data
y = digits.target

# Preprocess the data (you may need different preprocessing for your specific dataset)
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(X, y,test_size=0.2,random_state=42)

# Initialize the RBM model
rbm2 = BernoulliRBM(n_components=256, learning_rate=0.01, n_iter=5, verbose=1)
# Initialize the logistic regression model
logistic = LogisticRegression(max_iter=100)
# Create a pipeline that first extracts features using the RBM and then classifies with logistic regression
rbm_pipeline = Pipeline(steps=[('rbm', rbm2), ('logistic', logistic)])
# Train the DBN
rbm_pipeline.fit(X_train, Y_train)

y_pred = rbm_pipeline.predict(X_test)
print(
    "LR using RBM features:\n",
    classification_report(Y_test, y_pred)
)
# Evaluate the model on the test set
rbm_score = rbm_pipeline.score(X_test, Y_test)
print(f"RBM Classification score: {rbm_score}")
