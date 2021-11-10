''' Using the Diabetes dataset that is in scikit-learn, answer the questions below and create a scatterplot
graph with a regression line '''

import matplotlib.pylab as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import datasets

from sklearn.datasets import load_diabetes
diabetes = load_diabetes()

#how many samples and How many features?
print(diabetes.data.shape)
# returns "(442, 10)" meaning 442 samples and 10 features


# What does feature s6 represent?
print(diabetes.DESCR)


#print out the coefficient
from sklearn.model_selection import train_test_split

data_train, data_test, target_train, target_test = train_test_split(
    diabetes.data, diabetes.target, random_state=11
)

mymodel = LinearRegression()

mymodel.fit(X=data_train, y=target_train)

coef = mymodel.coef_
print(coef)


#print out the intercept
intercept = mymodel.intercept_
print(intercept)

# use predict to test your model
predicted = mymodel.predict(data_test)
expected = target_test

# create a scatterplot with regression line
plt.plot(expected, predicted, ".")


x = np.linspace(0,330,100)
print(x)
y = x
plt.plot(x,y)
plt.show()

# not an accurate model because the dots are so far away from the line
