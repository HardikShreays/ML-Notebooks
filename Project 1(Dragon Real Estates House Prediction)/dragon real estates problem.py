import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

housing = pd.read_csv("housing.csv")
# print(housing)

train_set, test_set = train_test_split(housing, test_size = 0.2, random_state = 42)
print(f'Rows in train set: {len(train_set)}\n Rows in test set: {len(test_set)}')

split = StratifiedShuffleSplit(n_splits = 1, random_state = 42, test_size = 0.2)
for train_index, test_index in split.split(housing, housing['CHAS']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

#
# print(strat_test_set)
#
# print('/......//')
#
# print(strat_train_set)

housing = strat_train_set.copy()

housing = strat_train_set.drop('MEDV',axis = 1)
housing_labels = strat_train_set['MEDV'].copy()

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = 'median')
imputer.fit(housing)

X = imputer.transform(housing)
housing_tr = pd.DataFrame(X,columns =housing.columns)

my_pipeline = Pipeline([
    ('imputer',SimpleImputer(strategy = 'median')),
    ('std_scaler', StandardScaler())
])
my_pipeline.fit_transform(housing)

housing_num_tr = my_pipeline.fit_transform(housing_tr)

from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
# model = LinearRegression()
model = RandomForestRegressor()
# model = DecisionTreeRegressor()
model.fit(housing_num_tr,housing_labels)

some_label = housing_labels[:5]
some_data = housing.iloc[:5]
prepared_data = my_pipeline.transform(some_data)

print(model.predict(prepared_data))

print('..........')
print(some_label)