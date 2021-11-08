from sklearn.datasets import load_digits

digits = load_digits()

# digits = bunch object
# 3 useful attributes:
    # digits.data - contains all the samples
    # digits.target - tells us what each of the samples represents 
    # digits.images

#print(digits.DESCR)

#print(digits.data[150])
# represents pixel intensity
# 64 column array is returned

#print(digits.target[150])
# this prints the TARGET of the numbers from the array, a singular number is returned (in this case a "0" is returned)

#print(digits.data.shape) #returns rows and columns because it's your data - you have 1797 rows and 64 columns (or 1797 samples with 64 features)
#print(digits.target.shape) #only shows rows (1797 rows and 1 column) because it is a single number, not a set of data


import matplotlib.pyplot as plt

figure, axes = plt.subplots(nrows=4, ncols=6, figsize=(6,4))

#plt.show()

for item in zip(axes.ravel(), digits.images, digits.target):
    axes, image, target = item
    axes.imshow(image, cmap=plt.cm.gray_r)
    axes.set_xticks([])
    axes.set_yticks([])
    axes.set_title(target)

plt.tight_layout()
#plt.show()

from sklearn.model_selection import train_test_split
# splits the data to where some of it is used to train the model and some is used to test it
# usually this method RANDOMLY picks the rows it's going to train and test, but to make it not random you use 'random_state=#'

data_train, data_test, target_train, target_test = train_test_split(
    digits.data, digits.target, random_state=11
)

print(data_train.shape)
print(data_test.shape)
print(target_train.shape)
print(target_test.shape)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

knn.fit(X=data_train, y=target_train)
# we give it the target so it can LEARN

predicted = knn.predict(X=data_test)
# we only have an x value for the predicted because it's supposed to spit out an answer 
# we do not give it the target so we can TEST it
expected = target_test

print(predicted[:20])
print(expected[:20])
# the second to last element of the lists are different *** but for the most part it was accurate 
# this difference seems like a small problem, but in the real world where machine learning is used (ex: self-driving cars) you can't afford those mistakes


# we can see the score of how it predicted it
print(format(knn.score(data_test,target_test), ".2%"))

# we can see the exact ones it got wrong by iterating through both lists at the same time using zipping 
wrong = [(p,e) for (p,e) in zip(predicted, expected) if p != e]
print(wrong)

###########################################################################################################################

## CONFUSION MATRIX allows you to see everything visually what it got wrong and how far it got wrong
from sklearn.metrics import confusion_matrix

confusion = confusion_matrix(y_true=expected, y_pred=predicted)

print(confusion)
# there are 10 rows and 10 columns 
# first row is the first class - meaning the number '0' - it guessed all of them correctly 
# fourth row is the fourth class - meaning the number '3' - it guessed all but 2 correctly (there are 42 in the '3' column and 1 in each of the '5' and '7' columns)
# the number '8' was the worst accuracy for guessing 

###########################################################################################################################
## CONFUSION MATRIXES CAN BE HARD TO READ, so let's create a HEAT MAP

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt2

confusion_df = pd.DataFrame(confusion, index=range(10), columns=range(10))

figure = plt2.figure(figsize=(7,6))
axes = sns.heatmap(confusion_df, annot=True, cmap=plt2.cm.nipy_spectral_r)

plt2.show()

### solution for errors is to add more samples!!!