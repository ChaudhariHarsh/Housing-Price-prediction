import numpy as np

def housingPrice():
    Z = []
    file = open("data.txt", "r") 
    for line in file:
        line = line.split(',')
        Z.append(line)
    
    Z = np.matrix(Z, dtype=float)
    Zv = Z
    Zv = np.sort(Zv, axis=0)
    (j, k) = Z.shape
    X = np.zeros((j,k+2),dtype=float)
    Y = np.zeros((j,1),dtype=float)
    Xv = np.zeros((j,k+2),dtype=float)
    Yv = np.zeros((j,1),dtype=float)
    for i in range(j):
        X[i,0]= 1
        X[i,1]=Z[i,0]
        X[i,2]=Z[i,0]**(1/2.0)
        X[i,3]=Z[i,1]
        X[i,4]=Z[i,1]**(1/2.0)
        Y[i,0]=Z[i,2]
        Xv[i,0]= 1
        Xv[i,1]=Zv[i,0]
        Xv[i,2]=Zv[i,0]**(1/2.0)
        Xv[i,3]=Zv[i,1]
        Xv[i,4]=Zv[i,1]**(1/2.0)
        Yv[i,0]=Zv[i,2]
    X = np.matrix(X)
    Xv = np.matrix(Xv)

    X = X.T ## Make this X with shape of (5,47)
    Y = Y.T ## Make this Y with shape of (1,47)
    Xv = np.matrix(Xv)
    Xv = Xv.T ## Make this X with shape of (5,47)
    Yv = Yv.T ## Make this Y with shape of (1,47)
    return X, Y, Xv, Yv
