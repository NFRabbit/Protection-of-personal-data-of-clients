#!/usr/bin/env python
# coding: utf-8

#  - Пол - Gender
# 
#  - Возраст- Age
# 
#  - Зарплата - Salary
# 
#  - Члены семьи - Family members
# 
#  - Страховые выплаты - Insurance payments



import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


import warnings
warnings.filterwarnings('ignore')



data = pd.read_csv("/datasets/insurance.csv")
data.head(15)

data.info()

data.duplicated().sum()


data.drop_duplicates(inplace=True)
data.duplicated().sum()


features = data.drop('Страховые выплаты', axis=1)
target = data['Страховые выплаты']
features_train, features_test, target_train, target_test = train_test_split(features, target,
                                                                            test_size=0.4, random_state=12345)
model = LinearRegression().fit(features_train, target_train)
predictions = model.predict(features_test)


model = LinearRegression().fit(features_train, target_train)
predictions = model.predict(features_test)
rscore = r2_score(target_test, predictions)
print("The coefficient of determination is equal to ", rscore)

matrix = features.values @ features.values.T

matrix_train, matrix_test, target_m_train, target_m_test = train_test_split(matrix, target, 
                                                                            test_size=0.4, random_state=12345)


model = LinearRegression().fit(matrix_train, target_m_train)
predictions = model.predict(matrix_test)
rscore_2 = r2_score(target_m_test, predictions)
print("The coefficient of determination is equal to ", rscore_2)


# Conclusion:
# In the case of multiplying the feature matrix by a random invertible matrix, it is possible to protect the data while not losing much in the quality of the model. (The slight difference is caused by the peculiarity of the matrix transformations and floating-point arithmetic, which is normal.)
