import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression as lr
import matplotlib.pyplot as plt

data = pd.read_csv("linear.csv")

X = data["squaremeters"]
Y = data["price"]

X = np.array(X).reshape(len(X),1)
Y = np.array(Y).reshape(len(Y),1)


linearregression = lr() # We call linear regression function.
linearregression.fit(X,Y)  # We show X and Y values on the graph

linearregression.predict(X) # We predict the price of the houses respect to the x which is size of the house.

m = linearregression.coef_ # We find the coefficient which is m (y=mx+b)
b = linearregression.intercept_ #We find intercept which is b. (when x = 0)

a = np.arange(150) # We define an array size of 150.

plt.scatter(X,Y) #We draw the points of X and Y values on the graph.
plt.scatter(a,m*a+b, c="red",marker=">")
plt.show()



