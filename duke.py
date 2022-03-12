import numpy as np
import matplotlib.pyplot as plt  # To visualize
import pandas as pd  # To read data
from sklearn.linear_model import LinearRegression
# importing train_test_split from sklearn
from sklearn.model_selection import train_test_split
# splitting the data
from sklearn.model_selection import cross_validate

df = pd.read_csv('CCPP_data.csv')  # load data set
x = df[['AT','V','AP','RH']]
y = df['PE']

print(x)
print(y)

#split data step for training and testing
'''
# importing train_test_split from sklearn
from sklearn.model_selection import train_test_split
# splitting the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
'''
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

linear_regression = LinearRegression()  # create object for the class
linear_regression.fit(x_train, y_train)  # perform linear regression
y_pred = linear_regression.predict(x_test)  # make predictions

print(y_pred)


print("x test set below")
print(x_test)

result = cross_validate(linear_regression, x, y)
print("cross validation --> test scores array")
print(result['test_score'])