from __future__ import division
import numpy as np
from math import log10, floor
import copy

# LR Zerlegung mit Zeilentausch MIT RUNDEN AUF MANTISSE 2! -> fm2: float mantissa 2 base 10
def PLR_fm2(A, printIterations = True):
    n = A.shape[0]
    L = np.eye(n)
    P = np.eye(n)
    R = A.astype('float')
    for k in range(n - 1):
        p = np.argmax(abs(R[k:n,k])) # finde Pivotelement
        # Tausche Zeilen von R, L, P
        R[[k,p+k]] = R[[p+k, k]]
        P[[k,p+k]] = P[[p+k,k]]
        L[[k,p+k]] = L[[p+k,k]]
        # Tausche Spalten von L
        L.T[[k,p+k]] = L.T[[p+k,k]]

        for j in range(k+1,n):
            L[j,k] = round_sig(R[j,k]/R[k,k])
            for i in range(k, n):
                R[j, i] = round_sig(R[j, i] - round_sig(L[j, k] * R[k, i]))

        if printIterations:
            print("----------########### ITERATION " + str(k+1) + " ###########-------------")
            print('L:')
            print(L)
            print('R:')
            print(R)
            print('P:')
            print(P)
            print('A:')
            print(calc_A_fromLR(L,R))
            print("\n")
    return P,L,R

def round_sig(x, sig=2):
    epsilon = 0.000000001
    if (x != 0.0) or (abs(x - 0.0) > epsilon):
        return round(x, sig-int(floor(log10(abs(x))))-1)
    else:
        return 0.0

def round_Matrix(M):
    m,n = M.shape
    for i in range(m):
        for j in range(n):
            M[i,j] = round_sig(M[i,j])
    return M;

def round_Vector(V):
    for i in range(len(V)):
        V[i] = round_sig(V[i])
    return V

def round_MatrixMul(A, B):
    # A, B gleiche Dimension
    X = round_Matrix(A)
    Y = round_Matrix(B)
    n = A.shape[0]
    if A.shape != B.shape:
        return
    else:
        C = np.zeros(shape=A.shape)
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    C[i,j] = round_sig(C[i,j] + round_sig(X[i, k] * Y[k, j]))
        #print(C)
        return C

def calc_A_fromLR(L,R):
    # only works for n x n matrices!
    # Calc A from L+R-I (I = unit matrix diag(1))
    n = L.shape[0]
    I = np.eye(n)
    return np.subtract(np.add(L,R),I)

A = np.array([[-1/3, -34/5, -131/30, -32/15],
              [3/4, 59/30, -7/8, 1],
              [1/4, -58/45, 257/120, -67/120],
              [-1/2, -1/5, 5/4, -1/2]])
b = np.array([[-2/3], [5/6], [-11/72], [-773/426]])

# auf mantisse 2 runden
A = round_Matrix(A)
b = round_Matrix(b)

# LR Zerlegung
P,L,R = PLR_fm2(A)


# berechne x (aber leider nicht mit mantisse 2)
A_ = round_MatrixMul(L,R)
x = np.linalg.solve(A_,b)
print("---------------###########  Loesung x fuer Ax=b: ###########------------------")
print("!! Lösungsverfahren von numpy verwendet allerdings keine Rundungen auf Mantisse 2 !!")
print("x:")
print(x)





