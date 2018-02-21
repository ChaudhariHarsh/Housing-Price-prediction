import numpy as np


##---------------- Load Data------------------------
Z = []
file = open("data.txt", "r") 
for line in file:
    line = line.split(',')
    Z.append(line)
    
Z = np.matrix(Z, dtype=int)
(j, k) = Z.shape
X = np.zeros((j,k),dtype=int)
x1 = np.zeros(j)
x2 = np.zeros(j)
Y = np.zeros(j,dtype=int)
for i in range(j):
    X[i,0]= 1
    X[i,1]=Z[i,0]
    x1[i] = Z[i,0]
    X[i][2]=Z[i,1]
    x2[i] = Z[i,1]
    Y[i]=Z[i,2]
X = np.matrix(X)


##---------------- LinearRegression Function--------
def LinearRegression(Response, Feature, **options):
    if options.get("alpha") != None:
        alpha = options.get("alpha")
    else:
        alpha = 0.0000000001
        
    Feature = np.matrix(Feature)
    Response = np.matrix(Response)
    (row, column) = Feature.shape
    Thetas = np.zeros(column)
    Thetas = np.matrix(Thetas)
    Thetas = np.transpose(Thetas)
    J = np.zeros(( row + 1, column + 1))
    ##------------ Training Phase------------------   
    for iteration in range(1000000):

        
    ##------------ Hyponthsis function-------------
        hx = np.matmul(Feature,Thetas)
        hx = np.transpose(hx)
        
    ##------------ Cost Function-------------------
        J0 = (hx - Response)
        J0 = np.matrix(J0)
        J1 = J0.dot(x1)
        J1 = np.matrix(J1)
        J2 = J0.dot(x2)
        J2 = np.matrix(J2)
        Jsum0 = int(np.sum(J0)) / (2*column)
        Jsum1 = int(np.sum(J1)) / (2*column)
        Jsum2 = int(np.sum(J2)) / (2*column)
    ##------------ Theta Update--------------------
        Thetas[0] = Thetas[0] - ( alpha * Jsum0 )        
        Thetas[1] = Thetas[1] - ( alpha * Jsum1 )
        Thetas[2] = Thetas[2] - ( alpha * Jsum2 )
        
    return Thetas

##---------------- Function Call ----------------
Thetas = LinearRegression(Y,X)
print Thetas[0] + (Thetas[1]*X[0,1]) + (Thetas[2]*X[0,2]) , Y[1]
