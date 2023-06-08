import numpy as np

def matrix_factorization(R, X, Y, K, iterations = 5000, alpha = 0.0002, beta = 0.02, error_treshhold = 0.001):

    Y = Y.T

    for _ in range(iterations):
        for i in range(len(R)):
            for j in range(len(R[i])):

                #if there exists a value
                if R[i][j] > 0:
                    #we calclualte the error 
                    eij = R[i][j] - np.dot(X[i,:], Y[:,j])

                    for k in range(K):
                        X[i][k] = X[i][k] + alpha * (2 * eij * Y[k][j])
                        Y[k][j] = Y[k][j] + alpha * (2 * eij * X[i][k])

        
        total_error = 0
        for i in range(len(R)):
            for j in range (len(R[i])):

                if R[i][j] > 0:
                    total_error = total_error + pow(R[i][j] - np.dot(X[i, :], Y[:, j]), 2)


        if total_error < error_treshhold:
            break
    
    return X, Y.T

