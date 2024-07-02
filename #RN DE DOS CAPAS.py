#RN DE DOS CAPAS

import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt
from numba import jit

#Leer imagenes

data = pd.read_csv('train.csv')

# pasarlas a arreglos y revolverlas

data = np.array(data)
m, n = data.shape
print("Numero de imagenes = ", m)
print("Numero de pixeles = ", n)
np.random.shuffle(data)

#Separar imagenes  en prueba y entrenamiento

data_dev = data[0:1000].T #.T es transpuesto
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255
_, m_train = X_train.shape

#valores iniciales al azar

@jit(nopython=True)
def init_params():
    W = []
    b = []
    m = np.array([784,10,10])
    n = np.array([10,10,10])
    capas = len(m)
    for i in range(capas):
        W.append(np.random.rand(n[i], m[i]) - 0.5)
        b.append(np.random.rand(n[i], 1) - 0.5)
    return W, b

#Funcion reLU

def ReLU(Z):
    return np.maximum(Z, 0)

#funcon softmax

def softmax(Z):
    A = np.exp(Z)/sum(np.exp(Z))
    return A

#evaluar red

def forward_prop(W, b, X):
    A = []
    Z = []
    AA = X
    for i in range(len(W)):
        ZZ = W[i].dot(AA) + b[i]
        if i < len(W)-1:
            AA = ReLU(ZZ)
        if i == len(W)-1:
            AA = softmax(ZZ)
        A.append(AA)
        Z.append(ZZ)
    return Z, A

#deriv relu

def ReLU_deriv(Z):
    return Z > 0

# Codificacion de la clasificacion

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size),Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

# Calculo numerico dl grad

def backward_prop(Z, A, W, X, Y):
    dW = []
    db = []
    n = len(W)-1
    one_hot_Y = one_hot(Y)
    dZ = A[n] - one_hot(Y)
    dWW = 1 / m * dZ.dot(A[n].T)
    dbb = 1 / m * np.sum(dZ)
    db.append(dbb)
    dW.append(dWW)
    if n>1:
        for i in range(n-1,0,-1):
            dZ = W[i+1].T.dot(dZ)
            dWW = 1 / m * dZ.dot(A[i].T)
            dbb = 1 / m * np.sum(dZ)
            db.append(dbb)
            dW.append(dWW)
    dZ = W[1].T.dot(dZ) * ReLU_deriv(Z[0])
    dWW = 1 / m * dZ.dot(X.T)
    dbb = 1 / m * np.sum(dZ)
    db.append(dbb)
    dW.append(dWW)
    db.reverse()
    dW.reverse()    

#mejora de parametros

def update_params(W,b, dW, db, alpha):
    for i in range(len(W)):
        W[i] = W[i] - alpha * dW[i]
        b[i] = b[i] - alpha * db[i]
    return W, b

# Precision

@jit(nopython=True)
def get_prediction(A2):
    return np.argmax(A2, 0)

# Precicoion

@jit(nopython=True)
def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size 

#descenso de gradiente

def gradient_descent(X, Y, alpha, iterations):
    W, b = init_params()
    for i in range(iterations):
        Z, A = forward_prop(W, b, X)
        dW, db = backward_prop(Z, A, W, X, Y)
        W, b = update_params(W, b, dW, db, alpha)
        if i % 10 == 0:
            print("Iteracion: ", i)
            predictions = get_prediction(A[len(A)-1])
            print(get_accuracy(predictions, Y))
    return W, b

#entrenar la red

W, b = gradient_descent(X_train, Y_train, 0.10, 1000)

# hacer predicciones

def make_predictions(X, W, b):
    Z, A = forward_prop(W, b, X)
    predictions = get_prediction(A[len(A)-1])
    return predictions




