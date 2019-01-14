# qr_decomposition.py
"""Volume 1: The QR Decomposition.
<McKenna Pitts>
<Section 2>
<October 27, 2018>
"""

from scipy import linalg as la
import numpy as np

# Problem 1
def qr_gram_schmidt(A):
    """Compute the reduced QR decomposition of A via Modified Gram-Schmidt.

    Parameters:
        A ((m,n) ndarray): A matrix of rank n.

    Returns:
        Q ((m,n) ndarray): An orthonormal matrix.
        R ((n,n) ndarray): An upper triangular matrix.
    """
    
    m,n = A.shape										#create matrices Q and R
    Q = np.copy(A)
    R = np.zeros((n,n))
    for i in range(n):									#Normalize the ith column of Q
        R[i][i] = la.norm(Q[:,i])
        Q[:,i] = Q[:,i]/R[i][i]
        for j in range(i+1, n):							#Orthogonalize the jth column of Q
            R[i][j] = np.dot(Q[:,j].T, Q[:,i])
            Q[:,j] = Q[:,j] - R[i][j]*Q[:,i]

    return Q,R


# Problem 2
def abs_det(A):
    """Use the QR decomposition to efficiently compute the absolute value of
    the determinant of A.

    Parameters:
        A ((n,n) ndarray): A square matrix.

    Returns:
        (float) the absolute value of the determinant of A.
    """
    Q,R = qr_gram_schmidt(A)							#calcute Q and R
    return abs(np.prod(np.diag(R)))						#multiply R's diagonal 


# Problem 3
def solve(A, b):
    """Use the QR decomposition to efficiently solve the system Ax = b.

    Parameters:
        A ((n,n) ndarray): An invertible matrix.
        b ((n, ) ndarray): A vector of length n.

    Returns:
        x ((n, ) ndarray): The solution to the system Ax = b.
    """
    n = len(A)
    x = np.zeros(n)
    Q,R = qr_gram_schmidt(A)							#Compute Q and R
    y = np.dot(Q.T, b)									#Calculate y = Q.Tb
    
    for k in range(n-1, -1, -1):						#back subsitiution
        x[k] = 1/R[k,k] * (y[k] - np.sum([R[k,j] * x[j] for j in range(k+1,n)]))
        
    return x


# Problem 4
def qr_householder(A):
    """Compute the full QR decomposition of A via Householder reflections.

    Parameters:
        A ((m,n) ndarray): A matrix of rank n.

    Returns:
        Q ((m,m) ndarray): An orthonormal matrix.
        R ((m,n) ndarray): An upper triangular matrix.
    """
    sign = lambda x: 1 if x>= 0 else -1
    
    m,n = A.shape									#initialize Q and R
    R = np.copy(A)
    Q = np.eye(m)
    for k in range(n):								
        u = np.copy(R[k:,k])
        u[0] = u[0] + sign(u[0])*la.norm(u)			#normalize u
        u = u/la.norm(u)
        R[k:,k:] = R[k:,k:] - 2*np.outer(u, np.dot(u.T, R[k:,k:]))		#apply reflection to R
        Q[k:,:] = Q[k:,:] - 2*np.outer(u, np.dot(u.T, Q[k:,:]))			#apply reflection to Q
    return Q.T, R


# Problem 5
def hessenberg(A):
    """Compute the Hessenberg form H of A, along with the orthonormal matrix Q
    such that A = QHQ^T.

    Parameters:
        A ((n,n) ndarray): An invertible matrix.

    Returns:
        H ((n,n) ndarray): The upper Hessenberg form of A.
        Q ((n,n) ndarray): An orthonormal matrix.
    """
    sign = lambda x: 1 if x>= 0 else -1
    
    m,n = A.shape									#initialize Q and H
    H = np.copy(A)
    Q = np.eye(m)
    for k in range(n-2):
        u = np.copy(H[k+1:,k])	
        u[0] = u[0] + sign(u[0])*la.norm(u)			#normalize u
        u = u/la.norm(u)
        H[k+1:,k:] = H[k+1:,k:] - 2*np.outer(u, np.dot(u.T, H[k+1:,k:]))		#Apply Q[k] to H
        H[:,k+1:] = H[:,k+1:] - 2*np.outer(np.dot(H[:,k+1:], u), u.T)			#Apply Q[k].T to H
        Q[k+1:,:] = Q[k+1:,:] - 2*np.outer(u, np.dot(u.T, Q[k+1:,:]))			#Apply Q[k] to Q
    return H, Q.T
        
        
