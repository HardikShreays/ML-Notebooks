from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

# loading data
iris = datasets.load_iris()

features = iris.data
label = iris.target

# print(features[0], label[0])

# training classifier
clf = KNeighborsClassifier()

clf.fit(features,label)

pred = clf.predict([[13,1,1,1]])

print(pred)