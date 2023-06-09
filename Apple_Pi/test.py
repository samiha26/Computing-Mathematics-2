import numpy as np
import sys
from main import matrix_factorization
import csv
import pandas as pd

with open("C:\\Users\\hp\\OneDrive\\Desktop\\Apple_Pi\\reviews_updated.csv") as fp:
    reader = csv.reader(fp, delimiter=",", quotechar='"')
    next(reader, None)  # skip the headers
    data_read = [row for row in reader]


data_read = np.array(data_read)
data_read = data_read[:,1:]

data_read = [list(map(int, x)) for x in data_read]

R = data_read[0:10]

print("Before Matrix Factorization")
for i in range(len(R)):
    print(R[i])

R = np.array(R)
# N: num of Users
N = len(R)
# M: num of Apps
M = len(R[0])
# Num of Latent Features
K = min(M, N)

 
X = np.random.rand(N,K)
Y = np.random.rand(M,K)

 

nX, nY = matrix_factorization(R, X, Y, K)

print("\nAfter Matrix Factorization")

R_hat = np.dot(nX, nY.T)

R_hat = np.round(R_hat, 1)

for i in range(len(R_hat)):
    print(repr(R_hat[i]))

