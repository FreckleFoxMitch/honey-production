import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv('honeyproduction.csv')

# check out the data
print(df.head())
print(df.info())


"""Make a Linear Regression model based on total production of honey per year"""

#get the mean of totalprod column per year
prod_per_year = df.groupby('year').totalprod.mean().reset_index()

# create X and y for scatterplot
# we will need to reshape X to get it into the right format
X = prod_per_year.year
X = X.values.reshape(-1, 1)

y = prod_per_year.totalprod

# create a scatterplot
plt.scatter(X, y, alpha=0.4)

# create and fit a Linear Regression Model
regr = linear_model.LinearRegression()
regr.fit(X, y)

# remove # to see the slope and the intercept
#print(regr.coef_[0])
#print(regr.intercept_)

y_predict = regr.predict(X)

# visualization
plt.plot(X, y_predict)
plt.title("Total Production of Honey per Year")
plt.xlabel("Year")
plt.ylabel("Total production (lbs)")
plt.show()
plt.show()


"""Predict the Honey Decline"""

# create a NumPy array called X_future that is the range from 2013 to 2050 (incl.) and reshape it for scikit-learn
# we need .reshape() for rotating this array from the row of numbers to the column of numbers
X_future = np.array(range(2013, 2051))
X_future = X_future.reshape(-1, 1)

# create a prediction model
future_predict = regr.predict(X_future)

# visualization
plt.scatter(X, y, alpha=0.4)
plt.plot(X, y_predict)
plt.plot(X_future, future_predict)
plt.title("Honey Decline Prediction")
plt.xlabel("Year")
plt.ylabel("Total production (lbs)")
plt.show()
