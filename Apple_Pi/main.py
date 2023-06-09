import numpy as np

def matrix_factorization(R, X, Y, K, iterations = 5000, alpha = 0.0002, beta = 0.02, error_treshhold = 0.001):

    Y = Y.T

    for _ in range(iterations):
        for i in range(len(R)):
            for j in range(len(R[i])):

                #if there exists a value
                if R[i][j] > 0:
                    #we calculate the difference (error) 
                    eij = R[i][j] - np.dot(X[i,:], Y[:,j])

                    for k in range(K):
                        X[i][k] = X[i][k] + alpha * (2 * eij * Y[k][j] - beta * X[i][k])
                        Y[k][j] = Y[k][j] + alpha * (2 * eij * X[i][k] - beta * Y[k][j])

        
        total_error = 0
        for i in range(len(R)):
            for j in range (len(R[i])):

                #Adding up all the errors after one iteration
                if R[i][j] > 0:
                    total_error = total_error + pow(R[i][j] - np.dot(X[i, :], Y[:, j]), 2)
                    
                    #As per the regularization
                    for k in range(K):
                        total_error = total_error + (beta/2) * ( pow(X[i][k],2) + pow(Y[k][j],2) )


        if total_error < error_treshhold:
            break
    
    return X, Y.T

