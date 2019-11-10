# Artificial Neural Network

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

## Creating the Artificial Neural Networks (ANN)

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
print("Predicting the Test set results\n", y_test)

# Predicting a single new observation

"""
Predict if the customer with the following information will leave the bank:

Geography: France
Credit Score: 600
Gender: Male
Age: 40
Tenure: 3
Balance: 60000
Number of Products: 2
Has Credit Card: Yes
Is Active Member: Yes
Estimated Salary: 50000
"""

new_prediction = classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = (new_prediction > 0.5)

print("Predicting [0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]\n", new_prediction)

# Creating the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print("\n")

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix\n", cm)

# Calculating metrics using the confusion matrix

print("\n")

TP = cm[0][0]
FN = cm[0][1]
TN = cm[1][0]
FP = cm[1][1]
print("True Positive (TP):", TP)
print("False Negative (FN):", FN)
print("True Negative (TN):", TN)
print("False Positive (FP):", FP)

print("\n")

accuracy = (TP + TN) / (TP + TN + FP + FN)
print("Accuracy = (TP + TN) / (TP + TN + FP + FN): %.2f %%" %(accuracy*100))

recall = TP / (TP + FN)
print("Recall = TP / (TP + FN): %.2f %%" %(recall*100) )

precision = TP / (TP + FP)
print("Precision = TP / (TP + FP): %.2f %%" %(precision*100) )

Fmeasure = (2 * recall * precision) / (recall + precision)
print("Fmeasure = (2 * recall * precision) / (recall + precision): %.2f %%" %(Fmeasure*100) )