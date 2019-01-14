# linear_transformations.py
"""Volume 1: Linear Transformations.
<Name>
<Class>
<Date>
"""

from random import random
import numpy as np
from matplotlib import pyplot as plt
import time

data = np.load("horse.npy")
#plt.plot(data[0], data[1], 'k,')
#plt.axis([-1, 1, -1, 1])
#plt.gca().set_aspect("equal")
#plt.show()


# Problem 1
def stretch(A, a, b):
    """Scale the points in A by a in the x direction and b in the
    y direction.

    Parameters:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        a (float): scaling factor in the x direction.
        b (float): scaling factor in the y direction.
    """
    B = np.array([[a, 0],[0, b]])
    C = B @ A

    plt.plot(C[0], C[1], 'k,')
    plt.axis([-1, 1, -1, 1])
    plt.gca().set_aspect("equal")
    plt.show()

    return C
    

def shear(A, a, b):
    """Slant the points in A by a in the x direction and b in the
    y direction.

    Parameters:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        a (float): scaling factor in the x direction.
        b (float): scaling factor in the y direction.
    """
    
    B = np.array([[1, a],[b, 1]])
    C = B @ A
    
    plt.plot(C[0], C[1], 'k,')
    plt.axis([-1, 1, -1, 1])
    plt.gca().set_aspect("equal")
    plt.show()
    
    return C

def reflect(A, a, b):
    """Reflect the points in A about the line that passes through the origin
    and the point (a,b).

    Parameters:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        a (float): x-coordinate of a point on the reflecting line.
        b (float): y-coordinate of the same point on the reflecting line.
    """
    B = np.array([[a**2 - b**2, 2*a*b],[2*a*b, b**2 - a**2]])
    C = 1 / (a**2 + b**2) * B
    D = C @ A
    
    plt.plot(D[0], D[1], 'k,')
    plt.axis([-1, 1, -1, 1])
    plt.gca().set_aspect("equal")
    plt.show()
    
    return D

def rotate(A, theta):
    """Rotate the points in A about the origin by theta radians.

    Parameters:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        theta (float): The rotation angle in radians.
    """
    B = np.array([[np.cos(theta), (-1)*np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    C = B @ A

    #plt.plot(C[0], C[1], 'k,')
    #plt.axis([-1, 1, -1, 1])
    #plt.gca().set_aspect("equal")
    #plt.show()
    
    return C


# Problem 2
def solar_system(T, x_e, x_m, omega_e, omega_m):
    """Plot the trajectories of the earth and moon over the time interval [0,T]
    assuming the initial position of the earth is (x_e,0) and the initial
    position of the moon is (x_m,0).

    Parameters:
        T (int): The final time.
        x_e (float): The earth's initial x coordinate.
        x_m (float): The moon's initial x coordinate.
        omega_e (float): The earth's angular velocity.
        omega_m (float): The moon's angular velocity.
    """
    #initialize the different x, y values for the vectors
    y_e = 0
    y_m = 0
    p_e = np.array([x_e, y_e])
    p_m = np.array([x_m, y_m])
    earth = np.zeros([2, 1000])
    moon = np.zeros([2, 1000])
    
    #use for loops to create an array of vectors to represent the path of the moon and earth
    t = np.linspace(0, T, 1000)
    for i in range(1000):
        earth[:,i] = rotate(p_e, t[i]*omega_e)
    for i in range(1000):    
        moon[:,i] = rotate(p_m - p_e, t[i]*omega_m)
    moon = moon + earth        
    
    #plot the path of the moon and earth
    plt.plot(earth[0], earth[1], 'b-', linewidth=3, label="Earth")
    plt.plot(moon[0], moon[1], '-', color='orange', linewidth=3, label="Moon")
    plt.axis([-12,12,-12,12])
    plt.axis("equal")
    plt.legend(loc = "lower right")
    plt.show()

    

    
    
    


def random_vector(n):
    """Generate a random vector of length n as a list."""
    return [random() for i in range(n)]

def random_matrix(n):
    """Generate a random nxn matrix as a list of lists."""
    return [[random() for j in range(n)] for i in range(n)]

def matrix_vector_product(A, x):
    """Compute the matrix-vector product Ax as a list."""
    m, n = len(A), len(x)
    return [sum([A[i][k] * x[k] for k in range(n)]) for i in range(m)]

def matrix_matrix_product(A, B):
    """Compute the matrix-matrix product AB as a list of lists."""
    m, n, p = len(A), len(B), len(B[0])
    return [[sum([A[i][k] * B[k][j] for k in range(n)])
                                    for j in range(p) ]
                                    for i in range(m) ]

# Problem 3
def prob3():
    """Use time.time(), timeit.timeit(), or %timeit to time
    matrix_vector_product() and matrix-matrix-mult() with increasingly large
    inputs. Generate the inputs A, x, and B with random_matrix() and
    random_vector() (so each input will be nxn or nx1).
    Only time the multiplication functions, not the generating functions.

    Report your findings in a single figure with two subplots: one with matrix-
    vector times, and one with matrix-matrix times. Choose a domain for n so
    that your figure accurately describes the growth, but avoid values of n
    that lead to execution times of more than 1 minute.
    """
    #initialize the domain of n values
    domain = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256])
    
    #time the matrix_vector_product() function 
    times1 = []
    for i in domain:
        v = random_vector(i)
        M = random_matrix(i)
        start = time.time()
        matrix_vector_product(M, v)
        end = time.time()
        times1.append(end - start)
    
    #graph the times of the matrix_vector_product() function
    fun1 = plt.subplot(1, 2, 1)
    fun1.plot(domain, times1, 'b.-', linewidth=2, markersize=15)
    plt.xlabel("n", fontsize=14)
    plt.ylabel("Seconds", fontsize=14)
    
    #time the matrix_matrix_product() function
    times2 = []
    for i in domain:
        A = random_matrix(i)
        B = random_matrix(i)
        start = time.time()
        matrix_matrix_product(A, B)
        end = time.time()
        times2.append(end-start)
    
    #graph the matrix_matrix_product() function
    fun2 = plt.subplot(1, 2, 2)
    fun2.plot(domain, times2, '.-', color='orange', linewidth=2, markersize=15)
    plt.xlabel("n", fontsize=14)
    
    plt.show()
    
        
    

# Problem 4
def prob4():
    """Time matrix_vector_product(), matrix_matrix_product(), and np.dot().

    Report your findings in a single figure with two subplots: one with all
    four sets of execution times on a regular linear scale, and one with all
    four sets of exections times on a log-log scale.
    """
    #initialize the domain of n values
    domain = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256])
    
    #time the matrix_vector_product function
    times1 = []
    for i in domain:
        v = random_vector(i)
        M = random_matrix(i)
        start = time.time()
        matrix_vector_product(M, v)
        end = time.time()
        times1.append(end - start)
    
    #time the matrix_matrix_product function
    times2 = []
    for i in domain:
        A = random_matrix(i)
        B = random_matrix(i)
        start = time.time()
        matrix_matrix_product(A, B)
        end = time.time()
        times2.append(end-start)
        
    #time the matrix vector multiplication with np.dot()
    times3 = []
    for i in domain:
        w = random_vector(i)
        N = random_matrix(i)
        start = time.time()
        np.dot(N, w)
        end = time.time()
        times3.append(end-start)
    
    #time the matrix matrix multiplication with np.dot()
    times4 = []
    for i in domain:
       C = random_matrix(i)
       D = random_matrix(i)
       start = time.time()
       np.dot(C, D)
       end = time.time()
       times4.append(end-start)
        
        
    #graph the four different graphs by n and time
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(domain, times1, 'b.-', linewidth=2, markersize=15, label="Matrix-Vector")
    ax1.plot(domain, times2, '.-', color='orange', linewidth=2, markersize=15, label="Matrix-Matrix")
    ax1.plot(domain, times3, 'm.-', linewidth=2, markersize=15, label="np Matrix-Vector")
    ax1.plot(domain, times4, 'g.-', linewidth=2, markersize=15, label="np Matrix-Matrix")
    plt.xlabel("n", fontsize=14)
    plt.ylabel("Seconds", fontsize=14)
    ax1.legend(loc="upper left")    
    
    #graph the four different graphs on a log-log scale
    ax2 = plt.subplot(1, 2, 2)
    ax2.loglog(domain, times1, 'b.-', basex=2, basey=2, linewidth=2, markersize=15)
    ax2.loglog(domain, times2, '.-', color='orange', basex=2, basey=2, linewidth=2, markersize=15)
    ax2.loglog(domain, times3, 'm.-', basex=2, basey=2, linewidth=2, markersize=15)
    ax2.loglog(domain, times4, 'g.-', basex=2, basey=2, linewidth=2, markersize=15)
    plt.xlabel("n", fontsize=14)
    
    plt.show()  
        
