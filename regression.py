import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

#loading data sets
diabetes = datasets.load_diabetes()


diabetes_x = diabetes.data
# print(diabetes_x)
#features.....
diabetes_x_train = diabetes_x[:-30]
diabetes_x_test = diabetes_x[-20:]
#labels.....
diabetes_y_train = diabetes.target[:-30]
diabetes_y_test = diabetes.target[-20:]
# print(diabetes_y_test)
#model creation
model = linear_model.LinearRegression()
#training data
model.fit(diabetes_x_train,diabetes_y_train)
#prediction
diabetes_y_predict = model.predict(diabetes_x_test)

print("Mean sqaured erro is: ",mean_squared_error(diabetes_y_test, diabetes_y_predict))
print('weights:', model.coef_)#weights
print('intercepts:', model.intercept_)
#
# plt.scatter(diabetes_x_test,diabetes_y_test)
# plt.plot(diabetes_x_test,diabetes_y_predict)
# plt.show()
'''Mean sqaured erro is:  2561.3204277283867
weights: [941.43097333]
intercepts: 153.39713623331698'''