#descenso de grad

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import numba 
from numba import jit

data = pd.read_csv('datos2.csv')
X = data.iloc[:,0]
Y = data.iloc[:,1]

print(X)
print(Y)

N = len(X)
print(N)
sumx = sum(X)
sumy = sum(Y)

sumxy = sum(X*Y)
sumx2 = sum(X*X)
w1 = (N*sumxy - sumx*sumy)/(N*sumx2 - sumx*sumx)
w0 = (sumy - w1*sumx)/N
Ybar = w0 + w1*X

w0 = 0.0
w1 = 0.0
alpha =.0025
epocs = 100

@jit(nopython=True)
def descensoG(epocs,sumx,sumy,sumxy,sumx2,N,alpha):
    w0 = 0.0
    w1 = 0.0
    for i in range(epocs):
        Gradw0 = -2*(sumy-w0*N-w1*sumx)
        Gradw1 = -2*(sumxy-w0*sumx-w1*sumx2)
        w0 -= alpha*Gradw0
        w1 -= alpha*Gradw1
    return w0,w1
w0,w1 = descensoG(epocs,sumx,sumy,sumxy,sumx2,N,alpha)
Ybay2 = w0 + w1*X

plt.scatter(X,Y)
plt.rcParams['figure.figsize'] = (12.0, 9.0)
plt.plot([min(X), max(X)], [min(Ybar), max(Ybar)], color='red')
plt.plot([min(X), max(X)], [min(Ybay2), max(Ybay2)], color = 'yellow')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
