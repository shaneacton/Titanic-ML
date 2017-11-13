import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
import math

###import data
train_data = pd.read_csv('input/train.csv', index_col='PassengerId')
task_data = pd.read_csv('input/test.csv', index_col='PassengerId')

###set target data
train_data['target_name'] = train_data['Survived'].map({0: 'Not Survived', 1: 'Survived'})

#interperet training data
train_data['Sex'][:] = [(1 if "fe" in str(a) else 0) for a in train_data['Sex']]
train_data['Age'][:] = [(a if math.isfinite(a) else 35) for a in train_data['Age']]
train_data['Pclass'][:] = [(a if math.isfinite(a) else 2) for a in train_data['Pclass']]
train_data['SibSp'][:] = [(a if math.isfinite(a) else 1) for a in train_data['SibSp']]
train_data['Parch'][:] = [(a if math.isfinite(a) else 1) for a in train_data['Parch']]
train_data['Fare'][:] = [(a if math.isfinite(a) else 25) for a in train_data['Fare']]
train_data['Cabin'][:] = [(0 if 'A' in str(a) else (
    1 if 'B' in str(a) else(2 if 'C' in str(a) else(3 if 'D' in str(a) else(4 if 'E' in str(a) else 3))))) for a in
                          train_data['Cabin']]

#interperet testing data
task_data['Sex'][:] = [(1 if "fe" in str(a) else 0) for a in task_data['Sex']]
task_data['Age'][:] = [(a if math.isfinite(a) else 35) for a in task_data['Age']]
task_data['Pclass'][:] = [(a if math.isfinite(a) else 2) for a in task_data['Pclass']]
task_data['SibSp'][:] = [(a if math.isfinite(a) else 1) for a in task_data['SibSp']]
task_data['Parch'][:] = [(a if math.isfinite(a) else 1) for a in task_data['Parch']]
task_data['Fare'][:] = [(a if math.isfinite(a) else 25) for a in task_data['Fare']]
task_data['Cabin'][:] = [(0 if 'A' in str(a) else (
    1 if 'B' in str(a) else(2 if 'C' in str(a) else(3 if 'D' in str(a) else(4 if 'E' in str(a) else 3))))) for a in
                         task_data['Cabin']]

x_train_full = train_data[['Sex','Age','Pclass','SibSp','Parch','Fare','Cabin']]
y_full = train_data['Survived']


x_train, x_test, y_train, y_test = train_test_split(x_train_full,y_full, test_size=0.4)



knn1 = KNeighborsClassifier(n_neighbors=1)
knn1.fit(x_train, y_train)
y_predicted = knn1.predict(x_test)
print('knn:1 accuracy: ' + str(accuracy_score(y_test, y_predicted)))

knn3 = KNeighborsClassifier(n_neighbors=3)
knn3.fit(x_train, y_train)
y_predicted = knn3.predict(x_test)
print('knn:3 accuracy: ' + str(accuracy_score(y_test, y_predicted)))

knn5 = KNeighborsClassifier(n_neighbors=5)
knn5.fit(x_train, y_train)
y_predicted = knn5.predict(x_test)
print('knn:5 accuracy: ' + str(accuracy_score(y_test, y_predicted)))

log = LogisticRegression()
log.fit(x_train, y_train)
y_predicted = log.predict(x_test)
print('logistic accuracy: ' + str(accuracy_score(y_test, y_predicted)))


#create testing input vectors
x_task = task_data[['Sex','Age','Pclass','SibSp','Parch','Fare','Cabin']]

#fit model to full data
log = LogisticRegression()
log.fit(x_train_full,y_full)
y_predicted = log.predict(x_task)

task_data['Survived'] = y_predicted
# print(task_data['Survived'])

task_data['Survived'].to_csv("out.csv")




