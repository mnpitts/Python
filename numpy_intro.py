# numpy_intro.py
"""Python Essentials: Intro to NumPy.
<McKenna Pitts>
<Math 345 Section 2>
<September 4, 2018>
"""

import numpy as np

def prob1():
    """Define the matrices A and B as arrays. Return the matrix product AB."""
    A = np.array([[3, -1, 4],[1, 5, -9]])
    B = np.array([[2, 6, -5, 3],[5, -8, 9, 7],[9, -3, -2, -3]])
    
    return np.dot(A, B)


def prob2():
    """Define the matrix A as an array. Return the matrix -A^3 + 9A^2 - 15A."""
    A = np.array([[3, 1, 4],[1, 5, 9],[-5, 3, 1]])
    #return -1 * (A * A * A) + 9 * (A * A) - (15 * A)
    
    return -1 * np.dot(np.dot(A, A), A) + 9 * np.dot(A, A) - (15 * A)


def prob3():
    """Define the matrices A and B as arrays. Calculate the matrix product ABA,
    change its data type to np.int64, and return it.
    """
    A = np.triu(np.ones((7, 7), dtype = np.int))
    B = np.triu(np.full((7, 7), 5, dtype = np.int), 1) + np.tril(np.full((7, 7), -1, dtype = np.int), 0) 
    
    C = np.dot(np.dot(A, B), A)
    
    return C.astype(np.int64)


def prob4(A):
    """Make a copy of 'A' and set all negative entries of the copy to 0.
    Return the copy.

    Example:
        >>> A = np.array([-3,-1,3])
        >>> prob4(A)
        array([0, 0, 3])
    """
    B = np.copy(A)
    mask = B < 0
    B[mask] = 0
    
    return B


def prob5():
    """Define the matrices A, B, and C as arrays. Return the block matrix
                                | 0 A^T I |
                                | A  0  0 |,
                                | B  0  C |
    where I is the 3x3 identity matrix and each 0 is a matrix of all zeros
    of the appropriate size.
    """
    A = np.arange(6).reshape((3, 2)).T
    B = np.tril(np.full((3, 3), 3, dtype = np.int), 0)
    C = np.diag([-2, -2, -2])
    
    X = np.vstack((np.zeros((3, 3)), A, B))
    Y = np.vstack((A.T, np.zeros((5, 2))))
    Z = np.vstack((np.eye(3), np.zeros((2, 3)), C))
    
    D = np.hstack((X, Y, Z))
    
    return D


def prob6(A):
    """Divide each row of 'A' by the row sum and return the resulting array.

    Example:
        >>> A = np.array([[1,1,0],[0,1,0],[1,1,1]])
        >>> prob6(A)
        array([[ 0.5       ,  0.5       ,  0.        ],
               [ 0.        ,  1.        ,  0.        ],
               [ 0.33333333,  0.33333333,  0.33333333]])
    """
    return  A/A.sum(axis = 1)[:,None]
    


def prob7():
    """Given the array stored in grid.npy, return the greatest product of four
    adjacent numbers in the same direction (up, down, left, right, or
    diagonally) in the grid.
    """
    grid = np.load("grid.npy")
    
    right_max = np.max(grid[:,:-3] * grid[:,1:-2] * grid[:,2:-1] * grid[:,3:])
    vertical_max = np.max(grid[:-3,:] * grid[1:-2,:] * grid[2:-1,:] * grid[3:,:])
    rdiag_max = np.max(grid[:-3,:-3] * grid[1:-2,1:-2] * grid[2:-1,2:-1] * grid[3:,3:])
    ldiag_max = np.max(grid[:-3,3:] * grid[1:-2,2:-1] * grid[2:-1,1:-2] * grid[3:,:-3])
    
    return np.max([right_max, vertical_max, rdiag_max, ldiag_max])
    
    




