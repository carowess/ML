import pandas as pd
import numpy as np
import csv

training_file = pd.read_csv("gameratings.csv")
data_train = training_file.T.loc['console':'violence']
target_train = training_file['Target']

testing_file = pd.read_csv("test_esrb.csv")
data_test = testing_file.T.loc['console':'violence']
target_test = testing_file['Target']

data_train = data_train.T
data_test = data_test.T

##### #1 #####
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()

knn.fit(X=data_train, y=target_train)

predicted = knn.predict(X=data_test)

expected = target_test.astype('int').to_numpy()

##### #2 #####
wrong = [(p,e) for (p,e) in zip(predicted, expected) if p != e]
print(wrong)

##### #3 #####
target_key = pd.read_csv("target_names.csv")
target_key = target_key.drop('target_abbreviation',axis=1).to_numpy() 

titles = testing_file['title']

with open('ratings.csv','w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['title','prediction'])
    for x,y in zip(titles, predicted):
        for r in target_key:
            if r[0] == y:
                y = r[1]
                writer.writerow([x,y])
    print('Done')

