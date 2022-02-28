'''training logistics regression classifier to predict it is whether vergeniaca or not '''
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

iris = datasets.load_iris()

x = iris['data'][:,3:]
# print(x)

y = (iris['target'] == 2).astype(np.int_) # show featues in 0,1(true / false) on basis or vergenica

clf = LogisticRegression()
clf.fit(x, y)
print(clf.predict([[2.6]]))

#plotting
x_new = np.linspace(0,3,1000).reshape(-1,1)
y_prob = clf.predict_proba(x_new)
print(y_prob)
plt.plot(x_new,y_prob[:,1],"-g",label = "Vergenica")
plt.show()

# print(x_new)

# print(x)
# print(y)
# print(list(iris.keys()))
# print(list(iris['data']))
# print(list(iris['target']))
# print(list(iris['target']))

#training model