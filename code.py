#Charlie wants to purchase office-space. He does a detailed survey of the offices and corporate complexes in the area,
#and tries to quantify a lot of factors, such as the distance of the offices from residential and other commercial areas,
#schools and workplaces; the reputation of the construction companies and builders involved in constructing the apartments; 
#the distance of the offices from highways, freeways and important roads; the facilities around the office space and so on.
#Each of these factors are quantified, normalized and mapped to values on a scale of 0 to 1. Charlie then makes a table. 
#Each row in the table corresponds to Charlie's observations for a particular house. If Charlie has observed and noted F features,
#the row contains F values separated by a single space, followed by the office-space price in dollars/square-foot. 
#If Charlie makes observations for H houses, his observation table has (F+1) columns and H rows, and a total of (F+1) * H entries.

#Charlie does several such surveys and provides you with the tabulated data. 
#At the end of these tables are some rows which have just F columns (the price per square foot is missing). 
#Your task is to predict these prices. F can be any integer number between 1 and 5, both inclusive.

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# extract first line of input
A = input().split()
F = int(A[0])
N = int(A[1])

X = []
Y = []

#format input data to dataframe
for i in range(0, N):
    data = list(map(float, input().split()))
    X.append(data[:F])
    Y.append(data[F])
    
test = []
T = int(input())

#format input test data to dataframe
for i in range(0, T):
    data = list(map(float, input().split()))
    test.append(data)
    
poly = PolynomialFeatures(degree=3)
X_train = poly.fit_transform(X)
X_test = poly.fit_transform(test)

model = LinearRegression()
model.fit(X_train, Y)

pred = model.predict(X_test)

print('\n'.join(map('{0:.2f}'.format, pred)))
