# linear_systems.py
"""Volume 1: Linear Systems.
<Name>
<Class>
<Date>
"""

import numpy as np
import scipy as sp
from scipy import linalg as la
import time
from matplotlib import pyplot as plt
from scipy import sparse
from scipy.sparse import linalg as spla


# Problem 1
def ref(A):
    """Reduce the square matrix A to REF. You may assume that A is invertible
    and that a 0 will never appear on the main diagonal. Avoid operating on
    entries that you know will be 0 before and after a row operation.

    Parameters:
        A ((n,n) ndarray): The square invertible matrix to be reduced.

    Returns:
        ((n,n) ndarray): The REF of A.
    """
    #change the type of A and get the size
    A = A.astype('float64')
    n = len(A[0,:])
    #iterate through each column and make all the values below the diagonal zero
    for j in range(n):
    	for i in range(j+1,n):
        	A[i,j:] -= (A[i,j] / A[j,j]) * A[j,j:]
    return A

# Problem 2
def lu(A):
    """Compute the LU decomposition of the square matrix A. You may
    assume that the decomposition exists and requires no row swaps.

    Parameters:
        A ((n,n) ndarray): The matrix to decompose.

    Returns:
        L ((n,n) ndarray): The lower-triangular part of the decomposition.
        U ((n,n) ndarray): The upper-triangular part of the decomposition.
    """
    #get the size of A and initialize L and U and change their types
    n = len(A[0,:])
    U = np.copy(A)
    L = np.eye(n)            
    L = L.astype('float64')
    U = U.astype('float64')
    #iterate through each column and row to calculate the values of L and U
    for j in range(n):
        for i in range(j+1,n):
            L[i,j] = U[i,j] / U[j,j]
            U[i,j:] -= L[i,j] * U[j,j:]
    return L,U


# Problem 3
def solve(A, b):
    """Use the LU decomposition and back substitution to solve the linear
    system Ax = b. You may again assume that no row swaps are required.

    Parameters:
        A ((n,n) ndarray)
        b ((n,) ndarray)

    Returns:
        x ((m,) ndarray): The solution to the linear system.
    """
    L,U = lu(A)
    n = len(A)
    y = np.zeros(n)
    x = np.zeros(n)
    
    for i in range(n):
        if i == 0:
            y[i] = b[i]
        else:
            y[i] = b[i] - np.sum(np.dot(L[i,:i], y[:i].T))
    
    for j in range(n-1, -1, -1):
        if j == n-1:
            x[j] = 1/U[j,j] * y[j]
        else:
            x[j] = 1/U[j,j] * (y[j] - np.sum(np.dot(U[j,j+1:], x[j+1:])))
    
    print(x)
    return x
            
    
    
    

# Problem 4
def prob4():
    """Time different scipy.linalg functions for solving square linear systems.

    For various values of n, generate a random nxn matrix A and a random
    n-vector b using np.random.random(). Time how long it takes to solve the
    system Ax = b with each of the following approaches:

        1. Invert A with la.inv() and left-multiply the inverse to b.
        2. Use la.solve().
        3. Use la.lu_factor() and la.lu_solve() to solve the system with the
            LU decomposition.
        4. Use la.lu_factor() and la.lu_solve(), but only time la.lu_solve()
            (not the time it takes to do the factorization).

    Plot the system size n versus the execution times. Use log scales if
    needed.
    """
    #initialize the domain of n values
    domain = np.array([1,2,4,8,16,32,64,128,256,512,1024])
    
    #invert A and left-multiply the inverse to b and record the times for each n 
    times1 = []
    for n in domain:
        A = np.random.random((n,n))
        b = np.random.random((n,1))
        start = time.time()
        Ai = la.inv(A)
        x = np.dot(Ai,b)
        end = time.time()
        times1.append(end - start)
        
    #use la.solve() and record the times for each n
    times2 = []
    for n in domain:
        A = np.random.random((n,n))
        b = np.random.random((n,1))
        start = time.time()
        x = la.solve(A,b)
        end = time.time()
        times2.append(end - start)
        
    #use la.lu_factor() and la.lu_solve() and record the times for each n
    times3 = []
    for n in domain:
        A = np.random.random((n,n))
        b = np.random.random((n,1))
        start = time.time()
        L, P = la.lu_factor(A)
        x = la.lu_solve((L,P), b)
        end = time.time()
        times3.append(end - start)
        
    #use la.lu_solve() and record the times for each n
    times4 = []
    for n in domain:
        A = np.random.random((n,n))
        b = np.random.random((n,1))
        L, P = la.lu_factor(A)
        start = time.time()
        x = la.lu_solve((L,P), b)
        end = time.time()
        times4.append(end - start)
        
    #create a graph with log scales and graph the four different functions 
    plt.loglog(domain, times1, 'b.-', basex=2, basey=2, linewidth=2, markersize=15, label="Matrix Inverse")
    plt.loglog(domain, times2, '.-', color='orange', basex=2, basey=2, linewidth=2, markersize=15, label="la.solve()")
    plt.loglog(domain, times3, 'm.-', basex=2, basey=2, linewidth=2, markersize=15, label="la.lu_factor and la.lu_solve")
    plt.loglog(domain, times4, 'g.-', basex=2, basey=2, linewidth=2, markersize=15, label="la.lu_solve")
    plt.xlabel("n", fontsize=14)
    plt.ylabel("Seconds", fontsize=14)
    plt.legend(loc="upper left")
    plt.show()


# Problem 5
def prob5(n):
    """Let I be the n × n identity matrix, and define
                    [B I        ]        [-4  1            ]
                    [I B I      ]        [ 1 -4  1         ]
                A = [  I . .    ]    B = [    1  .  .      ],
                    [      . . I]        [          .  .  1]
                    [        I B]        [             1 -4]
    where A is (n**2,n**2) and each block B is (n,n).
    Construct and returns A as a sparse matrix.

    Parameters:
        n (int): Dimensions of the sparse matrix B.

    Returns:
        A ((n**2,n**2) SciPy sparse matrix)
    """
    diagonals = [1, -4, 1]
    offsets = [-1, 0, 1]
    B = sparse.diags(diagonals, offsets, shape = (n,n))
    
    A = sparse.block_diag([B]*n)
    
    A.setdiag([1]*(n-1)*n, n)
    A.setdiag([1]*(n-1)*n, -n)
    
    #plt.spy(A, markersize = 1)
    #plt.show()
    
    return A


# Problem 6
def prob6():
    """Time regular and sparse linear system solvers.

    For various values of n, generate the (n**2,n**2) matrix A described of
    prob5() and vector b of length n**2. Time how long it takes to solve the
    system Ax = b with each of the following approaches:

        1. Convert A to CSR format and use scipy.sparse.linalg.spsolve()
        2. Convert A to a NumPy array and use scipy.linalg.solve().

    In each experiment, only time how long it takes to solve the system (not
    how long it takes to convert A to the appropriate format). Plot the system
    size n**2 versus the execution times. As always, use log scales where
    appropriate and use a legend to label each line.
    """
    
    #initialize the domain of n values
    domain = np.array([2,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64])
    
    #convert A to CSR format and time the solve function for each n
    times1 = []
    for n in domain:
        #print(n)
        A = prob5(n)
        Acsr = A.tocsr()
        b = np.random.random((n**2,1))
        start = time.time()
        x = spla.spsolve(Acsr,b)
        end = time.time()
        times1.append(end - start)
        
    #convert A to an array and time the solve function for each n
    times2 = []
    for n in domain:
        A = prob5(n)
        Ar = A.toarray()
        b = np.random.random((n**2,1))
        start = time.time()
        x = la.solve(Ar,b)
        end = time.time()
        times2.append(end - start)
        
    #create a graph with log scales and graph the two different functions 
    plt.loglog(domain, times1, 'b-', basex=2, basey=2, linewidth=2, markersize=15, label="CSR Format")
    plt.loglog(domain, times2, '-', color='orange', basex=2, basey=2, linewidth=2, markersize=15, label="Numpy Array")
    plt.xlabel("n", fontsize=14)
    plt.ylabel("Seconds", fontsize=14)
    plt.legend(loc="upper left")
    plt.show()