from sympy import *
from sympy import init_printing
import copy
import numpy as np
from fractions import Fraction
init_printing()

def zeromat(p, q):
    return [[0] * q for i in range(p)]

def matmul(a, b):
    n, p = len(a), len(a[0])
    p1, q = len(b), len(b[0])
    if p != p1:
        raise ValueError("Incompatible dimensions")
    c = zeromat(n, q)
    for i in range(n):
        for j in range(q):
            c[i][j] = sum(a[i][k] * b[k][j] for k in range(p))
    return c

def MatrixMul(A, B):
    # A, B gleiche Dimension
    X = copy.deepcopy(A)
    Y = copy.deepcopy(B)
    n = A.shape[0]
    if A.shape != B.shape:
        return
    else:
        C = np.zeros(shape=A.shape)
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    C[i,j] = C[i,j] + X[i, k] * Y[k, j]
        return C

def mapmat(f, a):
    return [list(map(f, v)) for v in a]

def ratmat(a):
    return mapmat(Fraction, a)

def PLR(A, printIterations = True):
    n = A.shape[0]
    L = np.array(ratmat(np.eye(n)))
    P = np.array(ratmat(np.eye(n)))
    R = copy.deepcopy(A)
    for k in range(n - 1):
        p = np.argmax(abs(R[k:n,k])) # finde Pivotelement
        # Tausche Zeilen von R, L, P
        R[[k,p+k]] = R[[p+k, k]]
        P[[k,p+k]] = P[[p+k,k]]
        L[[k,p+k]] = L[[p+k,k]]
        # Tausche Spalten von L
        L.T[[k,p+k]] = L.T[[p+k,k]]

        for j in range(k+1,n):
            L[j,k] = R[j,k]/R[k,k]
            #R[j,k:] = R[j,k:]-L[j,k]*R[k,k:]
            for i in range(k, n):
                R[j, i] = R[j, i] - (L[j, k] * R[k, i])

        if printIterations:
            print("----------################# ITERATION " + str(k) + " ####################-------------")
            print('L:')
            print(L)
            print('R:')
            print(R)
            print('P:')
            print(P)
            print("\n")
            #print('A:')
            #print(np.array(matmul(R,matmul(P,L))))
        #round_MatrixMul(L,R)
    return P,L,R

def forward_elimination(A, b, n):
    """
    Calculates the forward part of Gaussian elimination.
    """
    for row in range(0, n-1):
        for i in range(row+1, n):
            factor = A[i,row] / A[row,row]
            for j in range(row, n):
                A[i,j] = A[i,j] - factor * A[row,j]

            b[i] = b[i] - factor * b[row]
    return A, b

def back_substitution(a, b, n):
    """"
    Does back substitution, returns the Gauss result.
    """
    x = np.array(ratmat(np.zeros((n,1))))
    x[n-1] = b[n-1] / a[n-1, n-1]
    for row in range(n-2, -1, -1):
        sums = b[row]
        for j in range(row+1, n):
            sums = sums - a[row,j] * x[j]
        x[row] = sums / a[row,row]
    return x

def gauss(A, b):
    """
    This function performs Gauss elimination without pivoting.
    """
    n = A.shape[0]

    # Check for zero diagonal elements
    if any(np.diag(A)==0):
        raise ZeroDivisionError(('Division by zero will occur; '
                                  'pivoting currently not supported'))

    A, b = forward_elimination(A, b, n)
    x = back_substitution(A, b, n)
    print("----------------------------###################-----------------------------")
    print("The solution x for the equation Ax=b considering exact fractions is:")
    print("x = ")
    print(x)
    print("----------------------------###################-----------------------------")
    return x

A = np.array([
    [Fraction(-1,3), Fraction(-34,5), Fraction(-131,30), Fraction(-32,15)],
    [Fraction(3,4), Fraction(59,30), Fraction(-7,8), Fraction(1,1)],
    [Fraction(1,4), Fraction(-58,45), Fraction(257,120), Fraction(-67,120)],
    [Fraction(-1,2), Fraction(-1,5), Fraction(5,4), Fraction(-1,2)]
])

b = np.array([[Fraction(-2,3)], [Fraction(5,6)], [Fraction(-11,72)], [Fraction(-773,426)]])

P, L, R = PLR(A)
# Löse nach x auf in Ax=b und printe ergebnis aus
x = gauss(A,b)

# Zur Kontrolle die Ergebnisse aus
L = np.array(( [Fraction(1,1),Fraction(0,1),Fraction(0,1),Fraction(0,1)],
               [Fraction(-9,4),Fraction(1,1),Fraction(0,1),Fraction(0,1)],
               [Fraction(-3,4), Fraction(23,48), Fraction(1,1), Fraction(0,1)],
               [Fraction(3,2), Fraction(-3,4), Fraction(-4,71), Fraction(1,1)]))
R = np.array(( [Fraction(-1,3), Fraction(-34,5), Fraction(-131,30), Fraction(-32,15)],
               [Fraction(0,1), Fraction(-40,3), Fraction(-107,10), Fraction(-19,5)],
               [Fraction(0,1),Fraction(0,1), Fraction(639,160), Fraction(-27,80)],
               [Fraction(0,1),Fraction(0,1),Fraction(0,1), Fraction(-12,71)]))

print('A:')
print(np.array(MatrixMul(L,R)))
