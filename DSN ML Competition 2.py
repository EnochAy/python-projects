#import libraries
import numpy as np
from sklearn import preprocessing, neighbors
import pandas as pd
import quandl
import math
import numpy as np
from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
import matplotlib as plt
import numpy as np
import matplotlib.pyplot as plt  # To visualize
import pandas as pd  # To read data
from sklearn.linear_model import LinearRegression
# manually impute missing values with numpy
from pandas import read_csv
from numpy import nan


#Loading train and test datasets
df1 = pd.read_csv('train.csv')
df2 = pd.read_csv('test.csv')

#Converting to dataframe
df1 = pd.DataFrame(df1)
df2 = pd.DataFrame(df2)
#Checking df1
print(df1)
print(df1.head())
print(df1.tail())
df1.keys()
print(type(df1))
print(df1.shape)
print(df1.columns)
print(df1.describe())
print(df1.duplicated())
print(df1.drop_duplicates())
print(df1.isnull())
print(df1.info())
print(df1.isnull().sum()) #Missing values: There are: 802(Item_Weight, float64) and 1450(Store_Size, object)                      
print(df1['Item_Store_Returns'].value_counts())

#Checking df2
df2 = pd.DataFrame(df2)
print(df2)
print(df2.head())
print(df2.tail())
df2.keys()
print(type(df2))
print(df2.shape)
print(df2.columns)
print(df2.describe())
print(df2.duplicated())
print(df2.drop_duplicates())
print(df2.isnull())
print(df2.info())
print(df2.isnull().sum()) #Missing values: There are: 802(Item_Weight, float64) and 1450(Store_Size, object)                     

#Treating Missing values: Item_Weight and Store_Size
#For the float64 variable: Item_Weight
df1['Item_Weight'].fillna(np.mean(df1['Item_Weight']), inplace=True)
df2['Item_Weight'].fillna(np.mean(df2['Item_Weight']), inplace=True)
#For the object variable: Store_Size
#Getting the modal Store_Size
df1['Store_Size'].hist() #Medium is the modal Store_Size
#Fill null Store_Size with the mode (Medium)
df1['Store_Size'].fillna('Medium', inplace=True)
df2['Store_Size'].fillna('Medium', inplace=True)
print(df1.isnull().sum())
print(df1.isnull().sum())
print(df1)
print(df2)

df1.drop(['Item_Type'], 1, inplace=True) 
df2.drop(['Item_Type'], 1, inplace=True) 
##converting some variables(columns) to data type the algorithm can take and work with: 
#create numeric factors out of the two levels in Item_Sugar_Content, Store_Location_Type, Store_Type, and Store_Size
df1['Item_Sugar_Content'].replace(['Ultra Low Sugar', 'Low Sugar', 'Normal Sugar'], [0, 1, 2], inplace= True)
df1['Store_Location_Type'].replace(['Cluster 1','Cluster 2', 'Cluster 3'], [0, 1, 2], inplace= True)
df1['Store_Type'].replace(['Grocery Store','Supermarket Type1', 'Supermarket Type2', 'Supermarket Type3'], [0, 1, 2, 3], inplace= True)
df1['Store_Size'].replace(['Small', 'Medium', 'High'], [0, 1, 2], inplace= True)

df2['Item_Sugar_Content'].replace(['Ultra Low Sugar', 'Low Sugar', 'Normal Sugar'], [0, 1, 2], inplace= True)
df2['Store_Location_Type'].replace(['Cluster 1', 'Cluster 2', 'Cluster 3'], [0, 1, 2], inplace= True)
df2['Store_Type'].replace(['Grocery Store','Supermarket Type1', 'Supermarket Type2', 'Supermarket Type3'], [0, 1, 2, 3], inplace= True)
df2['Store_Size'].replace(['Small','Medium', 'High'], [0, 1, 2], inplace= True)
print(df1)
print(df2)


print(df1['Item_Weight'].isnull().sum().sum())
print(df1['Store_Size'].isnull().sum().sum())
print(df2['Item_Weight'].isnull().sum().sum())
print(df2['Store_Size'].isnull().sum().sum()) #It is confirmed that there is no missing values in the train and test datasets
#Building Model (Regression Model) is the next task
print(df1.columns)
print(df2.columns)

#Selecting all rows, containing the listed columns
X_train = df1.iloc[:, 3:10] #NOTE: use .iloc for pandas, if it is numpy, there will be no need to add .iloc
y_train = df1['Item_Store_Returns']
X_test = df2.iloc[:, 3:10]
print(X_train)
print(y_train)
print(X_test)
#y_test = df2['Item_Store_Returns'] #Variable to be predicted
#Combining train and test data sets
all_data = pd.concat([df1, df2], axis = 0, ignore_index = True, sort = True)
print(all_data)

#Exporting the cleaned dataset to csv(comma separeted value)
all_data.to_csv('all_data.csv')
from sklearn.linear_model import LinearRegression
#X, y = all_data(n_samples=60)
#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
#lr = LinearRegression().fit(X_train, y_train)


#print("lr.coef_: {}".format(lr.coef_))
#print("lr.intercept_: {}".format(lr.intercept_))
#print("Training set score: {:.2f}".format(lr.score(X_train, y_train)))
#print("Test set score: {:.2f}".format(lr.score(X_test, y_test)))
linear_regressor = LinearRegression()  # create object for the class
model = linear_regressor.fit(X_train, y_train)  # perform linear regression
print(model)
r_sq = model.score(X_train, y_train)
print('coefficient of determination:', r_sq)
print('intercept:', model.intercept_)
print('slope:', model.coef_)

y_pred = model.predict(X_test)
print('predicted response:', y_pred, sep='\n')
print(y_pred)
print(y_train)

#Exporting the predicted values to csv(comma separeted value)
df = pd.DataFrame(data=y_pred)
df.to_csv('df.csv')
#accuracy = clf.score(X_test, y_test)
#print(accuracy)
#Y_pred = linear_regressor.predict(X_test)  # make predictions
#plt.scatter(X_train, Y_pred)
#plt.plot(X_test, Y_pred, color='green')
#plt.show()
#print(Y_pred)
