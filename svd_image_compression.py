# solutions.py
"""Volume 1: The SVD and Image Compression. Solutions File."""

import numpy as np
import scipy as sp
from scipy import linalg as la
import math
import matplotlib.pyplot as plt
from imageio import imread



# Problem 1
def compact_svd(A, tol=1e-6):
    """Compute the truncated SVD of A.

    Parameters:
        A ((m,n) ndarray): The matrix (of rank r) to factor.
        tol (float): The tolerance for excluding singular values.

    Returns:
        ((m,r) ndarray): The orthonormal matrix U in the SVD.
        ((r,) ndarray): The singular values of A as a 1-D array.
        ((r,n) ndarray): The orthonormal matrix V^H in the SVD.
    """
    A_H = A.conj().T
    matrix = np.dot(A_H, A)							#Calculate A^HA
    eigval, e_vectors = la.eig(matrix)				#Calculate eigenvalues and eigenvectors
    e_vectors = e_vectors.T
    sigma = eigval**(1/2)							#Calculate singular values of A
    sort = np.argsort(sigma)
    sort = sort[::-1]								
    sigma = sigma[sort]								#Sort from greatest to least
    e_vectors = e_vectors[sort]						#Sort eigenvectors greatest to least
    r = np.sum(sigma > tol)							#Count number of nonsingular values
    if len(sigma) - r != 0:
        sigma = sigma[:r]							#Keep only positive values
        e_vectors = e_vectors[:r]					#Keep corresponding eigenvectors
    U = A.dot(e_vectors.T)/sigma					#Construct U
    
    return U, sigma, e_vectors


# Problem 2
def visualize_svd(A):
    """Plot the effect of the SVD of A as a sequence of linear transformations
    on the unit circle and the two standard basis vectors.
    """
    E = np.array([[1,0,0],[0,0,1]])				#Create matrix E
    theta = np.linspace(0,2*np.pi, 200)			#Create matrix S
    x = np.cos(theta)
    y = np.sin(theta)
    S = np.vstack((x,y))
    
    plt.subplot(221)							#Plot S and E
    plt.plot(x, y, 'b')
    plt.plot(E[0,:], E[1,:], 'g')
    plt.axis("equal")
    
    U, sigma, V_H = la.svd(A)					#Calculate SVD
    
    V_HS = np.dot(V_H, S)						#Calculate V_HS and V_HE
    V_HE = np.dot(V_H, E)
    plt.subplot(222)							#Plot V_HS and V_HE
    plt.plot(V_HS[0,:], V_HS[1,:], 'b')
    plt.plot(V_HE[0,:], V_HE[1,:], 'g')
    plt.axis("equal")

    plt.subplot(223)							#Plot SV_HS and SV_HE
    plt.plot(sigma[0]*V_HS[0,:], sigma[1]*V_HS[1,:], 'b')
    plt.plot(sigma[0]*V_HE[0,:], sigma[1]*V_HE[1,:], 'g')
    plt.axis("equal")
    	
    sV_HS = np.vstack((sigma[0]*V_HS[0,:], sigma[1]*V_HS[1,:]))		#Calculate SV_HS and SV_HE
    sV_HE = np.vstack((sigma[0]*V_HE[0,:], sigma[1]*V_HE[1,:]))
    
    UsV_HS = np.dot(U, sV_HS)					#Calcuate USV_HS and USV_HE
    UsV_HE = np.dot(U, sV_HE)
    plt.subplot(224)							#Plot USV_HS and USV_HE
    plt.plot(UsV_HS[0,:], UsV_HS[1,:], 'b')
    plt.plot(UsV_HE[0,:], UsV_HE[1,:], 'g')
    
    plt.show()
    

# Problem 3
def svd_approx(A, s):
    """Return the best rank s approximation to A with respect to the 2-norm
    and the Frobenius norm, along with the number of bytes needed to store
    the approximation via the truncated SVD.

    Parameters:
        A ((m,n), ndarray)
        s (int): The rank of the desired approximation.

    Returns:
        ((m,n), ndarray) The best rank s approximation of A.
        (int) The number of entries needed to store the truncated SVD.
    """

    U, sigma, V = compact_svd(A)					#Get compact SVD
    if s > np.size(sigma):							#Check s is greater than the amount of singular values
        raise ValueError("s is larger than rank(A)")
    U = U[:,0:s]									#Get each matrix in truncated form
    sigma = sigma[0:s]
    V = V[0:s]
    values = U.size + sigma.size + V.size			#Calculate number of entries
    return U.dot(np.diag(sigma).dot(V)), values


# Problem 4
def lowest_rank_approx(A, err):
    """Return the lowest rank approximation of A with error less than 'err'
    with respect to the matrix 2-norm, along with the number of bytes needed
    to store the approximation via the truncated SVD.

    Parameters:
        A ((m, n) ndarray)
        err (float): Desired maximum error.

    Returns:
        A_s ((m,n) ndarray) The lowest rank approximation of A satisfying
            ||A - A_s||_2 < err.
        (int) The number of entries needed to store the truncated SVD.
    """
    U, sigma, V = compact_svd(A)				#Calculate the SVD
    if err <= sigma[len(sigma)-1]:				#Verify error is greater than last singular value
        raise ValueError("A cannot be approximated within the tolerance by a matrix of a lesser rank.")
    s = np.argmax(np.where(sigma < err))		#Find s that gives lowest rank approximation
    return svd_approx(A,s)						#Return A and s
    
   

# Problem 5
def compress_image(filename, s):
    """Plot the original image found at 'filename' and the rank s approximation
    of the image found at 'filename.' State in the figure title the difference
    in the number of entries used to store the original image and the
    approximation.

    Parameters:
        filename (str): Image file path.
        s (int): Rank of new image.
    """
    fig, axes= plt.subplots(1,2)                            #Create two subplots
    image = imread(filename)/255							#Read in the image
    if len(np.shape(image)) == 2:
        new_image, values = svd_approx(image, s)            #Use the previously defined function to get the new A and amount of values neccessary to store A
        new_image = np.real(new_image)                      #Take only the real values of A
        axes[0].imshow(image, cmap='gray')                  #Show the original image                      
        axes[1].imshow(new_image, cmap='gray')				#Show the approximated image
    else:
        red_layer = image[:,:,0]                            #Split the color image into three different layers, red green and blue
        blue_layer = image[:,:,1]
        green_layer = image[:,:,2]
        red_layer, red_val = svd_approx(red_layer, s)       #Calculate the truncated version of each layer and the amount of values necessary to store it
        red_layer = np.real(red_layer)                      #Take only the real values of each layer
        red_layer = np.clip(red_layer, 0, 1)                #Clip any values that are less than zero and greater than one
        blue_layer, blue_val = svd_approx(blue_layer, s)
        blue_layer = np.real(blue_layer)
        blue_layer = np.clip(blue_layer, 0, 1)
        green_layer, green_val = svd_approx(green_layer, s)
        green_layer = np.real(green_layer)
        green_layer = np.clip(green_layer, 0, 1)
        axes[0].imshow(image)                               #Show the original image
        axes[1].imshow(np.dstack((red_layer, blue_layer, green_layer)))   #Show the new image by stacking the three layers on top of one another
        values = red_val+blue_val+green_val                 #Calculate the values needed to store all three layers (the new image)
    axes[0].axis("off") 
    axes[1].axis("off")
    plt.suptitle('Difference in Entries Stored: ' + str(image.size-values)) 
    plt.show()
    return
