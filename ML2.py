import pandas as pd
from sklearn.model_selection import train_test_split

nyc = pd.read_csv("ave_hi_nyc_jan_1895-2018.csv")

print(nyc.head(3))

print(nyc.Date.values)

print(nyc.Date.values.reshape(-1,1))
# this -1 is telling it to make as many rows as you need as long as there is one column

X_train, X_test, y_train, y_test = train_test_split(
    nyc.Date.values.reshape(-1,1), nyc.Temperature.values, random_state=11
)

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

#training
lr.fit(X=X_train, y=y_train)

coef = lr.coef_
intercept = lr.intercept_

#only needs X_test because we want it to predict it on its own
predicted = lr.predict(X_test)
expected = y_test

print(predicted[:20])
print(expected[:20])

predict = lambda x: coef * x + intercept

print(predict(2025))

import seaborn as sns

axes = sns.scatterplot(
    data=nyc,
    x = 'Date',
    y = "Temperature",
    hue = "Temperature",
    palette = "winter",
    legend = False
)

axes.set_ylim(10,70)

import numpy as np

x = np.array([min(nyc.Date.values),max(nyc.Date.values)])
print(x)
y = predict(x)
print(y)

import matplotlib.pyplot as plt
line = plt.plot(x,y)
plt.show()
