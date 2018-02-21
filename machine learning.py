import numpy as np



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
##print X, X.shape
X = np.matrix(X)
def LinearRegression(Response, Feature, **options):
    alpha = 0.001
    Feature = np.matrix(Feature)
    Response = np.matrix(Response)
    
    (r, c) = Feature.shape
    print r
##    Feature = np.concatenate([np.ones(c),Feature])
#    Feature = np.transpose(Feature)
#    Feature = Feature.resize(r,c)
    Thetas = np.zeros(c)
    Thetas = np.matrix(Thetas)
    Thetas = np.transpose(Thetas)
    J = np.zeros((r+1,c+1))
##    print J
    for i in range(10000):
        hx = np.matmul(Feature,Thetas)
        hx = np.transpose(hx)
        #print hx.shape, Response.shape
        J0 = (hx - Response)
        J0 = np.matrix(J0)
        J1 = J0.dot(x1)
        J1 = np.matrix(J1)
        Jsum0 = int(np.sum(J0))
        Jsum1 = int(np.sum(J1))
        Thetas[0] = Thetas[0] - (Jsum0 * (alpha /96) )        
        Thetas[1] = Thetas[1] - (Jsum1 * (al.pha /96) )
        
##A = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
##B = [3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,48,51,54,57,60]
##Thetas = LinearRegression(A,B)
##print Thetas[0] + Thetas[1]*60

Thetas = LinearRegression(Y,X)

