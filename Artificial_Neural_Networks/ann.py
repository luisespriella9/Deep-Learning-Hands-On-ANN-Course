# Artificial Neural Network

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

#Transform categorical variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder_geography = LabelEncoder()
label_encoder_gender = LabelEncoder()
X[:, 1] = label_encoder_geography.fit_transform(X[:, 1])
X[:, 2] = label_encoder_gender.fit_transform(X[:, 2])
one_hot_encoder = OneHotEncoder(categorical_features = [1])
one_hot_encoder.fit_transform(X).to_array()
X = X[: , 1:]
'''
only 2 dummy variables for the country: for example spain, france while not labeling germany 
since it can correspond to value zero from both spain and france
'''
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting classifier to the Training set
# Create your classifier here

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)