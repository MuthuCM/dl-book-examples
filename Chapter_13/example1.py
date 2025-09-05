# Example 13.1 - RBM [ KNeighborsClassifier ]
import numpy as np
from sklearn.datasets import load_digits
from sklearn import datasets
from sklearn.neural_network import BernoulliRBM
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
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

knn = KNeighborsClassifier(n_neighbors=7, algorithm='kd_tree')

rbm = BernoulliRBM(n_components=625, learning_rate=0.00001, n_iter=10, verbose=True, random_state=42)

rbm_features_classifier = Pipeline(steps=[("rbm", rbm), ("KNN", knn)])
# Training RBM-Logistic Pipeline
rbm_features_classifier.fit(X_train, Y_train)


Y_pred = rbm_features_classifier.predict(X_test)
print(
    "KNN using RBM features:\n",
    classification_report(Y_test, Y_pred)
)

rbm_score = rbm_features_classifier.score(X_test, Y_test)
print(f"RBM Classification score: {rbm_score}")