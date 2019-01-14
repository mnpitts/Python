# image_segmentation.py
"""Volume 1: Image Segmentation.
<McKenna Pitts>
<Section 1>
<November 8>
"""

import numpy as np
from scipy import linalg as la
import scipy.sparse.linalg as lo
import scipy.sparse as sp
import scipy
from imageio import imread
import matplotlib.pyplot as plt


# Problem 1
def laplacian(A):
    """Compute the Laplacian matrix of the graph G that has adjacency matrix A.

    Parameters:
        A ((N,N) ndarray): The adjacency matrix of an undirected graph G.

    Returns:
        L ((N,N) ndarray): The Laplacian matrix of G.
    """
    D = np.diag(np.sum(A, axis=1))					#Sum the entries to get the diagonals
    L = D - A										#Subtract A from D to get L
    return L

# Problem 2
def connectivity(A, tol=1e-8):
    """Compute the number of connected components in the graph G and its
    algebraic connectivity, given the adjacency matrix A of G.

    Parameters:
        A ((N,N) ndarray): The adjacency matrix of an undirected graph G.
        tol (float): Eigenvalues that are less than this tolerance are
            considered zero.

    Returns:
        (int): The number of connected components in G.
        (float): the algebraic connectivity of G.
    """
    A = laplacian(A)
    eigenvalues = np.real(la.eigvals(A))			#Make an array of the real eigenvalues
    zeros = 0
    for i in range(len(eigenvalues)):
        if eigenvalues[i] < tol:
            zeros += 1								#Count all the zeros
    eigenvalues = np.sort(eigenvalues)				#Sort all the eigenvalues
    if eigenvalues[i] < tol:
        return zeros, 0
    else:
        return zeros, eigenvalues[1]				#Return zeros and second smallest eigenvalue
        


# Helper function for problem 4.
def get_neighbors(index, radius, height, width):
    """Calculate the flattened indices of the pixels that are within the given
    distance of a central pixel, and their distances from the central pixel.

    Parameters:
        index (int): The index of a central pixel in a flattened image array
            with original shape (radius, height).
        radius (float): Radius of the neighborhood around the central pixel.
        height (int): The height of the original image in pixels.
        width (int): The width of the original image in pixels.

    Returns:
        (1-D ndarray): the indices of the pixels that are within the specified
            radius of the central pixel, with respect to the flattened image.
        (1-D ndarray): the euclidean distances from the neighborhood pixels to
            the central pixel.
    """
    # Calculate the original 2-D coordinates of the central pixel.
    row, col = index // width, index % width

    # Get a grid of possible candidates that are close to the central pixel.
    r = int(radius)
    x = np.arange(max(col - r, 0), min(col + r + 1, width))
    y = np.arange(max(row - r, 0), min(row + r + 1, height))
    X, Y = np.meshgrid(x, y)

    # Determine which candidates are within the given radius of the pixel.
    R = np.sqrt(((X - col)**2 + (Y - row)**2))
    mask = R < radius
    return (X[mask] + Y[mask]*width).astype(np.int), R[mask]


# Problems 3-6
class ImageSegmenter:
    """Class for storing and segmenting images."""

    # Problem 3
    def __init__(self, filename):
        """Read the image file. Store its brightness values as a flat array."""
        
        image = imread(filename)							#Read in the image
        self.scaled = image/255								#Scale the image
        if len(self.scaled.shape) == 3:
            self.brightness = self.scaled.mean(axis=2)		#Take the mean to put it into a gray scale
        else:
            self.brightness = self.scaled
        self.m, self.n = np.shape(self.brightness)			#Save the dimensions of the image
        self.brightness = np.ravel(self.brightness)			#Save the brightness as 1-d array
        return

    # Problem 3
    def show_original(self):
        """Display the original image."""
        if len(self.scaled.shape) == 3:						#If RGB color scheme, 
            plt.imshow(self.scaled)							#use plt.imshow()
        else:
            plt.imshow(self.scaled, cmap = 'gray')			#For gray scale
        plt.axis('off')										#Turn off the axis					
        plt.show()
        return

    # Problem 4
    def adjacency(self, r=5., sigma_B2=.02, sigma_X2=3.):
        """Compute the Adjacency and Degree matrices for the image graph."""
        n = len(self.brightness)
        A = sp.lil_matrix((n,n))							#Create matrix A
        B = np.zeros((n,n))
        for i in range(n):
            #Find indices that need weights and distances for the weighted part
            neighbors, distances = get_neighbors(i, r, self.m, self.n)
            bright_vector = -1*np.abs(self.brightness[i] - self.brightness[neighbors])
            weights = np.exp(bright_vector/sigma_B2 - distances/sigma_X2)
            A[i, neighbors] = weights						#Set each row to weights
            B[i, neighbors] = weights
        D = np.diag(np.sum(B, axis=1))
        A = sp.csc_matrix(A)
        return(A,D)

    # Problem 5
    def cut(self, A, D):
        """Compute the boolean mask that segments the image."""
        L = sp.csgraph.laplacian(A)							#Compute L
        D = np.sum(D, axis=1)
        D = D**(-1/2)
        D0 = sp.diags(D)									#Create diagonal matrix
        mult = D0 @ L @ D0
        eigenvalue, eigenvector = lo.eigsh(mult, which = 'SM', k=2)		#find eigenvectors corresponding to smallest eigenvalues
        my_vectors = eigenvector[:,1]						#Take the second vector
        my_vectors = my_vectors.reshape(self.m, self.n)
        mask = my_vectors > 0									#Return mask where each value is greater than 0
        return(mask)
        

    # Problem 6
    def segment(self, r=5., sigma_B=.02, sigma_X=3.):
        """Display the original image and its segments."""
        A, D = self.adjacency(r, sigma_B, sigma_X)			#Find A and D using function
        mask = self.cut(A, D)								#Find mask using function
        mask_neg = ~mask									#Create mask for negative values
        figure, axes = plt.subplots(1, 3)
        if len(np.shape(self.scaled)) == 3:					#If the photo is RGB
            axes[0].imshow(self.scaled)						#Stack the mask three times			
            axes[1].imshow(self.scaled * np.dstack((mask,mask,mask)))
            axes[2].imshow(self.scaled * np.dstack((mask_neg, mask_neg, mask_neg)))
        else:												#If the photo is grayscale
            axes[0].imshow(self.scaled, cmap = 'gray')		#Plot mask*photo
            axes[1].imshow(self.scaled * mask, cmap = 'gray')
            axes[2].imshow(self.scaled * mask_neg, cmap = 'gray')
        axes[0].axis('off')
        axes[1].axis('off')									#Turn off the axis
        axes[2].axis('off')
        plt.show()
        return


#if __name__ == '__main__':
    #ImageSegmenter("dream_gray.png").segment()
    #ImageSegmenter("dream.png").segment()
    #ImageSegmenter("monument_gray.png").segment()
    #ImageSegmenter("monument.png").segment()
