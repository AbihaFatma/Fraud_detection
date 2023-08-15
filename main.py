import pandas as pd
# dataset as csv file from this file data can be analyzed,pandas is used for making structured table of csv data
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# loading dataset to a pandas DataFrame
credit_card_data = pd.read_csv(r'C:\Users\abiha\Downloads\creditcard.csv.zip')
# first 5 rows of the dataset
print(credit_card_data.head())
print(credit_card_data.tail())
# Dataset Information
print(credit_card_data.info())
# checking the number of missing values in each column
print(credit_card_data.isnull().sum())
# distribution of legit transaction and fraudulent transaction
print(credit_card_data['Class'].value_counts())
# This dataset is highly unbalanced
# There are two labels zero represent normal transaction one represent fraud transaction
# separating the data for analysis
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]
print(legit.shape)
print(fraud.shape)
# statistical measures of the data
print(legit.Amount.describe())
print(fraud.Amount.describe())
# Compare the values for both transaction
print(credit_card_data.groupby('Class').mean())
# under sampling
# Build a sample dataset containing similar distribution of normal transaction and fraud transaction
# number of fraudulent transaction is 492
legit_sample = legit.sample(n=492)
# concatenating two dataframes
new_dataset = pd.concat([legit_sample, fraud], axis=0)
# axis=0 means dataframe should be added one by one, row-wise
# axis=1 means column-wise
print(new_dataset.head())
print(new_dataset.tail())
print(new_dataset['Class'].value_counts())
print(new_dataset.groupby('Class').mean())
# split dataset into features and targets
X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']
print(X)
print(Y)
# split data into training data and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)
# Use Logistic Regression Model for binary classification problem
model = LogisticRegression()
# training the logistic regression model with training data
# here fit function is used to train our model
model.fit(X_train, Y_train)
# X contains all features of the training data
# Y contains the corresponding labels, 0 and 1 labels here
# Model Evaluation based on Accuracy score
# accuracy on training data first of all
X_train_prediction = model.predict(X_train)
# here predict function is used to predict our model
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
# Here labels for X_train is predicting and store it into the X_train_prediction
# Then values predicted by our model that is X_train_prediction will be compared to the original label Y_train
# then it will give accuracy score and that will be stored in the training_data_accuracy
print('Accuracy on Training data : ', training_data_accuracy)
# here out of 100 predictions our model predict 94 predictions correctly
# Accuracy on testing data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy on Testing data :', test_data_accuracy)
# if accuracy score on training data is very different from accuracy score on test data
# then it means our model is over fitted or under fitted with the training data

