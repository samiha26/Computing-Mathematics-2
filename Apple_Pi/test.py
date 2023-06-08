import numpy as np
import sys
from main import matrix_factorization
import csv

with open("C:\\Users\\hp\\OneDrive\\Desktop\\Apple_Pi\\reviews_updated.csv") as fp:
    reader = csv.reader(fp, delimiter=",", quotechar='"')
    next(reader, None)  # skip the headers
    data_read = [row for row in reader]

print("before cutting")
print(data_read[0:10])

data_read = np.array(data_read)
data_read = data_read[:,1:]

data_read = [list(map(int, x)) for x in data_read]

R = data_read

print("After cutting")
print(R[0:10])

R = np.array(R)
# N: num of User
N = len(R)
# M: num of Movie
M = len(R[0])
# Num of Features
K = min(M, N)

 
X = np.random.rand(N,K)
Y = np.random.rand(M,K)

 

nX, nY = matrix_factorization(R, X, Y, K, alpha=0.002)

print("after mf")

nR = np.dot(nX, nY.T)

print (nR)