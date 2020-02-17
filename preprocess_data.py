## Data and Visual Analytics - Homework 4
## Georgia Institute of Technology
## Applying ML algorithms on the seizure dataset.

import numpy as np
import pandas as pd
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.svm import SVC

######################################### Reading and Splitting the Data ###############################################

# Read in all the data.
data = pd.read_csv('seizure_dataset.csv')

# Separate out the x_data and y_data.
x_data = data.loc[:, data.columns != "y"]
y_data = data.loc[:, "y"]

# The random state to use while splitting the data.
random_state = 100 # DO NOT CHANGE

# XXX
# TODO: Split 70% of the data into training and 30% into test sets. Call them x_train, x_test, y_train and y_test.
# Use the train_test_split method in sklearn with the parameter 'shuffle' set to true and the 'random_state' set to 100.
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.30, random_state=100)
# XXX


# ###################################### Without Pre-Processing Data ##################################################
# XXX
# TODO: Fit the SVM Classifier (with the tuned parameters)  on the x_train and y_train data.
start = time.time()
mysvc = SVC(C = 100, kernel = 'rbf')
mysvc.fit(x_train,y_train)
# XXX


# XXX
# TODO: Predict the y values for x_test and report the test accuracy using the accuracy_score method.
y_predict = mysvc.predict(x_test)
end = time.time()
print(end-start)
print(accuracy_score(y_test, y_predict.round()))
# XXX


# ###################################### With Data Pre-Processing ##################################################
# XXX
# TODO: Standardize or normalize x_train and x_test using either StandardScaler or normalize().
# Call the processed data x_train_p and x_test_p.
Scaler = StandardScaler()
Scaler.fit(x_train)
x_train_p = Scaler.transform(x_train)
x_test_p = Scaler.transform(x_test)
# XXX


# XXX
# TODO: Fit the SVM Classifier (with the tuned parameters) on the x_train_p and y_train data.
mysvc_p = SVC(C = 100, kernel = 'rbf')
mysvc_p.fit(x_train_p,y_train)
# XXX


# XXX
# TODO: Predict the y values for x_test_p and report the test accuracy using the accuracy_score method.
y_predict_p = mysvc_p.predict(x_test_p)
print(accuracy_score(y_test, y_predict_p.round()))
# XXX





