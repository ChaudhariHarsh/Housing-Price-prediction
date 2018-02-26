"""      """


import numpy as np
import math

def initialize_parameters(n_x,method):
    ##This function is for initialize All parameter
    
    parameters = {}
    if method == "Linear":
        W = np.zeros((1,n_x))
        b = 0
        parameters = {"W": W,
                      "b": b}
    elif method == "Logistic":
        W = np.zeros((1,n_x))
        b = 0
        parameters = {"W": W,
                      "b": b}
    else :
        print "Error : Define method in initialize parameter"
        
    
    return W, b
#parameters = initialize_parameters(5,method="Logistic")
#print parameters

def feature_normalization(X):
    (row, col) = X.shape
    for f in range(row):
        X[f,:] = (X[f,:]- min(X[f,:]))/(max(X[f,:])- min(X[f,:]))
    print X
    return X

def cost_function(X, Y, W, b, method):
    ## where X shape is (input_size, no_examples)
    (n,m) = X.shape
    if method == "Linear":
        hyponthsis_function = np.dot(W,X) + b
        cost = (hyponthsis_function - Y)**2
        error = np.sum(cost /(2*m))
    elif method == "Logistic":
        hyponthsis_function = np.dot(W,X) + b
        cost = 1/(1+exp(-hyponthsis_function))
        error = None
    else:
        print "Error In Cost Function : No method Found"

    return hyponthsis_function, cost, error

def gradient_descent(X, Y, W, b, itertions, learning_rate, method):
    (n,m) = X.shape
    for iteration in range(itertions):
        if method == "Linear":
            dJ = np.dot(W,X) + b
            db = np.sum(dJ, axis=1)/(2*m)
            dW = np.sum(np.dot(dJ,X), axis=1)/(2*m)
        elif method == "Logistic":
            dJ = None
            db = None
            dW = None
        else:
            print "Error in gradient descent: No method Found"

        W = W - learning_rate * dW
        b = b - learning_rate * db

        

    return W,b

def linear_regression(X, Y, itertions, learning_rate, method):
    
    (row, col) = X.shape

    (W, b) = initialize_parameters(row,method)

    X = feature_normalization(X)

    (W,b) = gradient_descent(X, Y, W, b, itertions, learning_rate, method)

    hyponthsis_function, cost, error = cost_function(X, Y, W, b, method)
    
    return hyponthsis_function, error
          

def logistic_regression(X, Y, itertions, learning_rate, method):
    
    (row, col) = X.shape

    (W, b) = initialize_parameters(row,method)

    X = feature_normalization(X)

    (W,b) = gradient_descent(X, Y, W, b, itertions, learning_rate, method)

    hyponthsis_function, cost, error = cost_function(X, Y, W, b, method)
    
    return hyponthsis_function, error



Z = []
file = open("data.txt", "r") 
for line in file:
    line = line.split(',')
    Z.append(line)
    
Z = np.matrix(Z, dtype=float)
(j, k) = Z.shape
print Z[5,:]
X = np.zeros((j,k+2),dtype=float)
Y = np.zeros((j,1),dtype=float)
for i in range(j):
    X[i,0]= 1
    X[i,1]=Z[i,0]
    X[i,2]=Z[i,0]**(1/2.0)
    X[i,3]=Z[i,1]
    X[i,4]=Z[i,1]**(1/2.0)
    Y[i,0]=Z[i,2]
X = np.matrix(X)
X = X.T

hponthsis_function, error = linear_regression(X, Y, itertions=100, learning_rate=0.000000001, method="Linear")

