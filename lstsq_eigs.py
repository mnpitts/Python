# lstsq_eigs.py
"""Volume 1: Least Squares and Computing Eigenvalues.
<McKenna Pitts>
<Section 2>
<October 30>
"""

# (Optional) Import functions from your QR Decomposition lab.
# import sys
# sys.path.insert(1, "../QR_Decomposition")
# from qr_decomposition import qr_gram_schmidt, qr_householder, hessenberg

import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg as la


# Problem 1
def least_squares(A, b):
    """Calculate the least squares solutions to Ax = b by using the QR
    decomposition.

    Parameters:
        A ((m,n) ndarray): A matrix of rank n <= m.
        b ((m, ) ndarray): A vector of length m.

    Returns:
        x ((n, ) ndarray): The solution to the normal equations.
    """
    Q,R = la.qr(A, mode = "economic")					#Q, R decomposition
    y = np.dot(Q.T, b)									#dot product of Q.T and b
    x = la.solve_triangular(R, y)						#Solve triangular system
    
    return x
    
    
    

# Problem 2
def line_fit():
    """Find the least squares line that relates the year to the housing price
    index for the data in housing.npy. Plot both the data points and the least
    squares line.
    """
    data = np.load("housing.npy")						#load the data
    n = len(data)
    ones_col = np.ones((n,1))						
    x_col = data[:,0].reshape((n,1))					
    	
    A = np.column_stack((x_col, ones_col))				#Create A
    b = data[:,1]										#Create b
    
    x = least_squares(A,b)								#find the least squares solution
    
    m = data[:,0]										#plot the data points
    n = data[:,1]
    plt.plot(m, n, 'b*', label="Data Points")
    plt.plot(m, x[0]*m + x[1], 'g', label="Least Squares Fit")		#plot the least squares fit
    plt.legend(loc="lower right")
    plt.show()
    


# Problem 3
def polynomial_fit():
    """Find the least squares polynomials of degree 3, 6, 9, and 12 that relate
    the year to the housing price index for the data in housing.npy. Plot both
    the data points and the least squares polynomials in individual subplots.
    """
    data = np.load("housing.npy")						#load the data
    
    A_three = np.vander(data[:,0],3)					#create matrices for each degree
    A_six = np.vander(data[:,0],6)
    A_nine = np.vander(data[:,0],9)
    A_twelve = np.vander(data[:,0],12)
    b = data[:,1]
    
    x_three = la.lstsq(A_three, b)[0]					#find the least squares solution for each degree
    x_six = la.lstsq(A_six, b)[0]
    x_nine = la.lstsq(A_nine, b)[0]
    x_twelve = la.lstsq(A_twelve, b)[0]
    
    f_three = np.poly1d(x_three)						#Create a polynomial for each degree
    f_six = np.poly1d(x_six)							#using the least squares solution
    f_nine = np.poly1d(x_nine)
    f_twelve = np.poly1d(x_twelve)
    
    m = data[:,0]										#prepare data to plot
    n = data[:,1]
    x = np.linspace(0,16)
    
    three = plt.subplot(221)							#plot polynomial of degree 3
    three.plot(m, n, 'b*')
    three.plot(x, f_three(x), 'g')
    
    six = plt.subplot(222)								#plot polynomial of degree 6
    six.plot(m, n, 'b*')
    six.plot(x, f_six(x), 'g')
    
    nine = plt.subplot(223)								#plot polynomial of degree 9
    nine.plot(m, n, 'b*')
    nine.plot(x, f_nine(x), 'g')
    
    twelve = plt.subplot(224)							#plot polynomial of degree 12
    twelve.plot(m, n, 'b*', label="Data Points")
    twelve.plot(x, f_twelve(x), 'g', label="Least Squares Fit")
    
    plt.legend(loc="lower right")
    plt.show()


def plot_ellipse(a, b, c, d, e):
    """Plot an ellipse of the form ax^2 + bx + cxy + dy + ey^2 = 1."""
    theta = np.linspace(0, 2*np.pi, 200)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    A = a*(cos_t**2) + c*cos_t*sin_t + e*(sin_t**2)
    B = b*cos_t + d*sin_t
    r = (-B + np.sqrt(B**2 + 4*A)) / (2*A)

    plt.plot(r*cos_t, r*sin_t)
    plt.gca().set_aspect("equal", "datalim")

# Problem 4
def ellipse_fit():
    """Calculate the parameters for the ellipse that best fits the data in
    ellipse.npy. Plot the original data points and the ellipse together, using
    plot_ellipse() to plot the ellipse.
    """
    x_col, y_col = np.load("ellipse.npy").T				#load the data
    ones_col = np.ones_like(x_col)
    A = np.column_stack((x_col**2, x_col, x_col*y_col, y_col, y_col**2))	#Create A
    b = ones_col															#create b
    
    a, b, c, d, e = la.lstsq(A, b)[0]					#find least squares solution

    plt.plot(x_col, y_col, 'k*')						#plot the data points
    plot_ellipse(a,b,c,d,e)								#plot the ellipse
    plt.show()
    

# Problem 5
def power_method(A, N=20, tol=1e-12):
    """Compute the dominant eigenvalue of A and a corresponding eigenvector
    via the power method.

    Parameters:
        A ((n,n) ndarray): A square matrix.
        N (int): The maximum number of iterations.
        tol (float): The stopping tolerance.

    Returns:
        (float): The dominant eigenvalue of A.
        ((n,) ndarray): An eigenvector corresponding to the dominant
            eigenvalue of A.
    """
    m,n = A.shape									#initialize m,n
    xn = np.random.random(n)						#create a random vector of length n
    xn = xn/la.norm(xn)								#normalize vector
    for k in range(1,N):
        xk = A.dot(xn)								#create the next vector
        xk = xk/la.norm(xk)							#normalize the vector
        if (la.norm(xk - xn) < tol):				#check tolerance
            return xk.T.dot(A.dot(xk)), xk			#return eigenvalue and corresponding eigenvector
        xn = xk
    return xn.T.dot(A.dot(xn)), xn					#return eigenvalue and corresponding eigenvector
    


# Problem 6
def qr_algorithm(A, N=50, tol=1e-12):
    """Compute the eigenvalues of A via the QR algorithm.

    Parameters:
        A ((n,n) ndarray): A square matrix.
        N (int): The number of iterations to run the QR algorithm.
        tol (float): The threshold value for determining if a diagonal S_i
            block is 1x1 or 2x2.

    Returns:
        ((n,) ndarray): The eigenvalues of A.
    """
    
    m,n = A.shape									#initialize m,n
    S = la.hessenberg(A)							#Put A in upper Hessenberg form
    for k in range(N):
        Q,R = la.qr(S)								#Get QR decomposition of A
        S = R.dot(Q)								#Recombine R[k] and Q[k] into A[k+1]
    eigenvalues = []								#initialize empty list of eigenvalues
    i = 0
    while i < n:
        if i == n-1 or abs(S[i+1][i] < tol):
            eigenvalues.append(S[i][i])
        else:										#use quadratic formula to get two eigenvalues
            b = -1*(S[i,i] + S[i+1,i+1])	
            c = (S[i,i] * S[i+1,i+1] - S[i,i+1] * S[i+1,i])
            eigenvalues.append((-1*b + cmath.sqrt(b**2 - 4*c)) / 2)			
            eigenvalues.append((-1*b - cmath.sqrt(b**2 - 4*c)) / 2)
            i += 1
        i += 1										#move to the next S[i]
    return np.array(eigenvalues)					#return array of eigenvalues
    