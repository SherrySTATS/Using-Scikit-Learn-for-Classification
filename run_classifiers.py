## Data and Visual Analytics - Homework 4
## Georgia Institute of Technology
## Applying ML algorithms to recognize seizure from EEG brain wave signals

import numpy as np
import pandas as pd
import time 

from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

######################################### Reading and Splitting the Data ###############################################

# Read in all the data.
data = pd.read_csv('seizure_dataset.csv')

# Separate out the x_data and y_data.
x_data = data.loc[:, data.columns != "y"]
y_data = data.loc[:, "y"]

# The random state to use while splitting the data. DO NOT CHANGE.
random_state = 100 # DO NOT CHANGE

# XXX
# TODO: Split each of the features and labels arrays into 70% training set and
#       30% testing set (create 4 new arrays). Call them x_train, x_test, y_train and y_test.
#       Use the train_test_split method in sklearn with the parameter 'shuffle' set to true 
#       and the 'random_state' set to 100.
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.30, random_state=100)
# XXX



# ############################################### Linear Regression ###################################################
# XXX
# TODO: Create a LinearRegression classifier and train it.
start = time.time()
regr = LinearRegression()
regr.fit(x_train, y_train)
# XXX

# XXX
# TODO: Test its accuracy (on the testing set) using the accuracy_score method.
# Note: Use y_predict.round() to get 1 or 0 as the output.
y_predict = regr.predict(x_test)
y_train_predict = regr.predict(x_train)
end = time.time()
print(end-start)
print('Linear Regression Running time')
print('regression test accuracy')
print(accuracy_score(y_test, y_predict.round()) )#testing accuracy
print('regression training accuracy')
print(accuracy_score(y_train, y_train_predict.round())) #training accuracy
# XXX


# ############################################### Multi Layer Perceptron #################################################
# XXX
# TODO: Create an MLPClassifier and train it.
start = time.time()
clf = MLPClassifier(hidden_layer_sizes=(100,100,100), max_iter=500, alpha=0.0001,
                     solver='sgd', verbose=10,  random_state=100,tol=0.000000001)
clf.fit(x_train, y_train)
# XXX


# XXX
# TODO: Test its accuracy on the test set using the accuracy_score method.
y_predict = clf.predict(x_test)
end = time.time()
print(end-start)
print('MLP Running Time')
y_train_predict = clf.predict(x_train)
print('MLP test accuracy')
print(accuracy_score(y_test, y_predict.round()) )#testing accuracy
print('MLP train accuracy')
print(accuracy_score(y_train, y_train_predict.round())) #training accuracy
# XXX




# ############################################### Random Forest Classifier ##############################################
# XXX
# TODO: Create a RandomForestClassifier and train it.
start = time.time()
rdf = RandomForestClassifier()
#rdf = RandomForestClassifier(n_estimators=100, max_depth=55,
#    min_samples_split=2, random_state=100)
rdf.fit(x_train, y_train)
# XXX


# XXX
# TODO: Test its accuracy on the test set using the accuracy_score method.
y_predict = rdf.predict(x_test)
y_train_predict = rdf.predict(x_train)
end = time.time()
print(end-start)
print('RDF Running time')
print('RDF testing accuracy')
print(accuracy_score(y_test, y_predict.round())) #testing accuracy
print('RDF training accuracy')
print(accuracy_score(y_train, y_train_predict.round())) #training accuracy
# XXX

# XXX
# TODO: Tune the hyper-parameters 'n_estimators' and 'max_depth'.
#       Print the best params, using .best_params_, and print the best score, using .best_score_.

n_estimators = [int(x) for x in np.linspace(start = 10, stop = 100, num = 3)]
max_depth = [int(x) for x in np.linspace(start = 10, stop = 100, num = 3)]
random_grid = {'n_estimators':n_estimators, 'max_depth':max_depth}
rdf_random = GridSearchCV(rdf, random_grid, cv = 10)
Scaler = MinMaxScaler()
Scaler.fit(x_train)
x_train_norm = Scaler.transform(x_train)
x_test_norm = Scaler.transform(x_test)
rdf_random.fit(x_train_norm,y_train)
print('RDF best params')
print(rdf_random.best_params_)
rdf_random.best_score_

#rdf = RandomForestClassifier(n_estimators=100, max_depth=55,
#    min_samples_split=2, random_state=100)
rdf_random.fit(x_train, y_train)
y_predict = rdf_random.predict(x_test)
print('RDF tuned test accuracy')
print(accuracy_score(y_test, y_predict.round())) #testing accuracy
# XXX


# ############################################ Support Vector Machine ###################################################
# XXX
# TODO: Create a SVC classifier and train it.
mysvc = SVC()
mysvc.fit(x_train,y_train)
# XXX

# XXX
# TODO: Test its accuracy on the test set using the accuracy_score method.
y_predict = mysvc.predict(x_test)
y_train_predict = mysvc.predict(x_train)
print('SVC test accuracy')
print(accuracy_score(y_test, y_predict.round())) #testing accuracy
print('SVC train accuracy')
print(accuracy_score(y_train, y_train_predict.round())) #training accuracy
# XXX

# XXX
# TODO: Tune the hyper-parameters 'C' and 'kernel' (use rbf and linear).
#       Print the best params, using .best_params_, and print the best score, using .best_score_.
para_grid = {'C':[0.0001,0.1,100], 'kernel':['linear','rbf']}
svc_random = GridSearchCV(mysvc, para_grid, cv = 10)
svc_random.fit(x_train_norm,y_train)
print('SVC tune params')
print(svc_random.best_params_)
svc_random.best_score_
y_predict = svc_random.predict(x_test_norm)
print('SVC tuned accuracy')
print(accuracy_score(y_test, y_predict.round()))
# XXX


# XXX 
# ########## PART C ######### 
# TODO: Print your CV's highest mean testing accuracy and its corresponding mean training accuracy and mean fit time.
# 		State them in report.txt.
svc_random.cv_results_['mean_train_score'][4]
svc_random.cv_results_['mean_fit_time'][4]
svc_random.cv_results_['mean_test_score'][4]

#rdf_random.cv_results_
# XXX


