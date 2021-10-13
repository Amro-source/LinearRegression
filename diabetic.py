# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 17:16:26 2021

@author: Zikantika
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

#cereal_df = pd.read_csv("/tmp/tmp07wuam09/data/cereal.csv")
diabetic_df2 = pd.read_csv("diabetes.csv")


print(diabetic_df2.head(5))

print(diabetic_df2.head(10))

print(diabetic_df2['Glucose'][0])


print(diabetic_df2['Glucose'])

X=diabetic_df2['Glucose']
Y=diabetic_df2['BloodPressure']

x, y = np.array(X), np.array(Y)

x=np.array(x).reshape(1,-1)

y=np.array(y).reshape(1,-1)

# Step 2b: Transform input data
x_ = PolynomialFeatures(degree=1, include_bias=False).fit_transform(x)

# Step 3: Create a model and fit it
model = LinearRegression().fit(x_, y)

# Step 4: Get results
r_sq = model.score(x_, y)
intercept, coefficients = model.intercept_, model.coef_

# Step 5: Predict
y_pred = model.predict(x_)


print(y_pred)

