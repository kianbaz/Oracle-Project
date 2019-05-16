import pandas as pd
import numpy as np


col = ['No.', 'Source', 'Destination', 'Protocol', 'Length', 'Label']
# load data

traffic = pd.read_csv("internetchicago2.csv", header=None, names=col, low_memory=False)
print(traffic.head())
# shows the list of csv with given column values

features = ['Destination', 'Length']
label = ['Label']
# check to see that your columns are properly labeled without whitespaces


# choosing dependent variables vs Independent variables
X = traffic[features]
y = traffic.Label



# training data, we must split x sets and y sets into training data
import sklearn
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
'''
# It's important to train your data to give you accurate results, however overtraining will lead to false positives
# Here we are using a 0.3 ratio, 70% of the data will be used to train the model, 30% to test the model.

# Sometimes, your data set wont be complete or may contain categorical variables, here we can create dummy variables
# These dummy variables can be taxing for a computer to process so, it's important to be careful
'''
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(X)
X_train_dumb = enc.transform(X_train)
X_test_dumb = enc.transform(X_test)



# Now we import and test the logistic regression, thankfully, sklearn library provides most of the hard work
from sklearn.linear_model import LogisticRegression

logRegression = LogisticRegression()
# introduce the logistic regression function

logRegression.fit(X_train_dumb, y_train)
# fit the model with training data

y_pred=logRegression.predict(X_test_dumb)
# prediction model using test data
print(y_test)
print(y_pred)
y_test = np.nan_to_num(y_test)
y_pred = np.nan_to_num(y_pred)
# Sometimes with these logregressions you can obtain Nan or inf values, you need to get rid of them for the model to wor
'''
# time for some metrics and data mapping
# First we start with confusion matrix, giving false positive, false negatives, etc.
'''
from sklearn import metrics
confusionMatrix = metrics.confusion_matrix(y_test,y_pred)
print((confusionMatrix))

# Lets vizualize the heatmap

import matplotlib.pyplot as plt
import seaborn as sns

class_id=[0,1]
fig, ax= plt.subplots()
tick = np.arange(len(class_id))
plt.xticks(tick, class_id)
plt.yticks(tick, class_id)

sns.heatmap(pd.DataFrame(confusionMatrix), annot=True, cmap="YlGnBu",fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion Matrix', y=1.1)
plt.ylabel("Actual")
plt.xlabel("Predicted Label")
plt.show()
'''
# This plot will show us which IP addresses that we will accurately predict as potential clients
# The model shows very strong statistics, 24 and 90 represent those that were accurately predicted
# But we all know that stats can lie, so lets check other metrics
'''
print("Model's precision is approx.:", metrics.precision_score(y_test,y_pred))
print("Model's accuracy is approx.:", metrics.accuracy_score(y_test,y_pred))
print("Model's recall is approx.:", metrics.recall_score(y_test,y_pred))
'''
# This shows that this model is 75.63% precise, which means when clients are identified, 75.63% are true clients
# A strong accuracy bodes well for this model
# Recall score is very impressive at 92.78%, of course these stats are taken in a vacuum, this means if there are
# potential clients on a dataset, this model will identify 92.78% of them.
# and client retention is hard to predict, nevertheless this simple model works simply to identify potential clients
'''



