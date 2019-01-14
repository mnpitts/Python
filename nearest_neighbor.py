# nearest_neighbor.py
"""Volume 2: Nearest Neighbor Search.
<McKenna Pitts>
<Section 2>
<October 24, 2018>
"""

import numpy as np
from scipy import linalg as la
from scipy.spatial import KDTree
from scipy import stats
from matplotlib import pyplot as plt


# Problem 1
def exhaustive_search(X, z):
    """Solve the nearest neighbor search problem with an exhaustive search.

    Parameters:
        X ((m,k) ndarray): a training set of m k-dimensional points.
        z ((k, ) ndarray): a k-dimensional target point.

    Returns:
        ((k,) ndarray) the element (row) of X that is nearest to z.
        (float) The Euclidean distance from the nearest neighbor to z.
    """
    t = z.T												#transpose z to perform subtraction
    dstar = min(la.norm(X-t, axis=1))					#find the minimum distance to z
    xstar = X[np.argmin(la.norm(X-t, axis=1))]			#find the element that is closest to z
    return xstar, dstar


# Problem 2: Write a KDTNode class.
class KDTNode:
    """A node class for k-d trees.  Contains an np array, references to two child nodes, 
    and a reference to the pivot.
    """
    
    def __init__(self, x):
        """Construct a new node and set the value attribute.  
        The other attributes will be set when the node is added to a tree.
        """
        if type(x) is not np.ndarray:
            raise TypeError(str(x) + " is not a NumPy array.")
        
        self.value = x
        self.prev = None
        self.left = None
        self.right = None
        self.pivot = 0


# Problems 3 and 4
class KDT:
    """A k-dimensional binary tree for solving the nearest neighbor problem.

    Attributes:
        root (KDTNode): the root node of the tree. Like all other nodes in
            the tree, the root has a NumPy array of shape (k,) as its value.
        k (int): the dimension of the data in the tree.
    """
    def __init__(self):
        """Initialize the root and k attributes."""
        self.root = None
        self.k = None

    def find(self, data):
        """Return the node containing the data. If there is no such node in
        the tree, or if the tree is empty, raise a ValueError.
        """
        def _step(current):
            """Recursively step through the tree until finding the node
            containing the data. If there is no such node, raise a ValueError.
            """
            if current is None:                     # Base case 1: dead end.
                raise ValueError(str(data) + " is not in the tree")
            elif np.allclose(data, current.value):
                return current                      # Base case 2: data found!
            elif data[current.pivot] < current.value[current.pivot]:
                return _step(current.left)          # Recursively search left.
            else:
                return _step(current.right)         # Recursively search right.

        # Start the recursive search at the root of the tree.
        return _step(self.root)

    # Problem 3
    def insert(self, data):
        """Insert a new node containing the specified data.

        Parameters:
            data ((k,) ndarray): a k-dimensional point to insert into the tree.

        Raises:
            ValueError: if data does not have the same dimensions as other
                values in the tree.
        """
        #if the tree is empty, create a node for the root of the tree
        if self.root == None:
            new_node = KDTNode(data)
            self.root = new_node
            self.k = data.shape[0]
            return
            
        #Define a recursive function to traverse the tree
        def my_step(data, current):
            if current == None:											#Add the node to the tree
                current = KDTNode(data)
                return current
            elif np.all(current.value == data):									#Data is already in the tree
                raise ValueError("The value is already in the tree.")
            elif data[current.pivot] < current.value[current.pivot]:	#Move to the left
                current.left = my_step(data, current.left)
                current.left.prev = current
                if current.pivot == self.k - 1:							#Set pivot value
                    current.left.pivot = 0
                else:
                   current.left.pivot = current.pivot + 1
                return current
            elif data[current.pivot] > current.value[current.pivot]:	#Move to the right
                current.right = my_step(data, current.right)
                current.right.prev = current
                if current.pivot == self.k - 1:							#Set pivot value
                    current.right.pivot = 0
                else:
                    current.right.pivot = current.pivot + 1
                return current
                
        #Start the recursion at the root of the tree
        my_step(data, self.root)
        return
        
        

    # Problem 4
    def query(self, z):
        """Find the value in the tree that is nearest to z.

        Parameters:
            z ((k,) ndarray): a k-dimensional target point.

        Returns:
            ((k,) ndarray) the value in the tree that is nearest to z.
            (float) The Euclidean distance from the nearest neighbor to z.
        """
        
        #Define a recursive function to traverse the tree and calculate the distance
        def KDSearch(current, nearest, dstar):
            if current is None:										#Base Case: Dead end
                return nearest, dstar
            x = current.value
            i = current.pivot
            if la.norm(x-z) < dstar:								#Check if current is closer to z than nearest
                nearest = current
                dstar = la.norm(x-z)
            if z[i] < x[i]:											#Search to the left
                nearest, dstar = KDSearch(current.left, nearest, dstar)
                if z[i] + dstar >= x[i]:							#Search to the right if needed
                    nearest, dstar = KDSearch(current.right, nearest, dstar)
            else:													#Search to the right
                nearest, dstar = KDSearch(current.right, nearest, dstar)
                if z[i] - dstar <= x[i]:							#Search to the left if needed
                    nearest, dstar = KDSearch(current.left, nearest, dstar)
            return nearest, dstar
        
        #Start the recursion at the root of the tree
        node, dstar = KDSearch(self.root, self.root, la.norm(self.root.value - z))
        return node.value, dstar
                    

    def __str__(self):
        """String representation: a hierarchical list of nodes and their axes.

        Example:                           'KDT(k=2)
                    [5,5]                   [5 5]   pivot = 0
                    /   \                   [3 2]   pivot = 1
                [3,2]   [8,4]               [8 4]   pivot = 1
                    \       \               [2 6]   pivot = 0
                    [2,6]   [7,5]           [7 5]   pivot = 0'
        """
        if self.root is None:
            return "Empty KDT"
        nodes, strs = [self.root], []
        while nodes:
            current = nodes.pop(0)
            strs.append("{}\tpivot = {}".format(current.value, current.pivot))
            for child in [current.left, current.right]:
                if child:
                    nodes.append(child)
        return "KDT(k={})\n".format(self.k) + "\n".join(strs)


# Problem 5: Write a KNeighborsClassifier class.
class KNeighborsClassifier:
    """A node class for k-nearest neighbors.  Accepts an integer n_neighbors, 
    the number of neighbors to include in the vote.
    """
    def __init__(self, n_neighbors):
        """Initialize the number of neighbors to include in the vote."""
        self.neighbors = n_neighbors
        
    def fit(self, X, y):
        """Load a SciPy KDTree with the data in X.  
        Save the tree and the labels as attributes.
        """
        tree = KDTree(X)
        self.tree = tree
        self.labels = y
        return
    
    def predict(self, z):
        """Query the KDTree for the n_neighbors elements of X that are nearest to z
        and return the most common label of those entries.
        """
        distances, indices = self.tree.query(z, k=self.neighbors)
        if self.neighbors == 1:
            return self.labels[indices]
        winner = stats.mode(self.labels[indices], axis=1)[0][:,0]
        return(winner)
        


# Problem 6
def prob6(n_neighbors, filename="mnist_subset.npz"):
    """Extract the data from the given file. Load a KNeighborsClassifier with
    the training data and the corresponding labels. Use the classifier to
    predict labels for the test data. Return the classification accuracy, the
    percentage of predictions that match the test labels.

    Parameters:
        n_neighbors (int): the number of neighbors to use for classification.
        filename (str): the name of the data file. Should be an npz file with
            keys 'X_train', 'y_train', 'X_test', and 'y_test'.

    Returns:
        (float): the classification accuracy.
    """
    data = np.load(filename)								#load the data
    X_train = data["X_train"].astype(np.float)
    y_train = data["y_train"]
    X_test = data["X_test"].astype(np.float)
    y_test = data["y_test"]
    
    model = KNeighborsClassifier(n_neighbors)
    model.fit(X_train, y_train)
    my_pred = model.predict(X_test)
    accuracy = (my_pred) == y_test							#check to see if predictions are correct
    return(np.mean(accuracy)*100)							#return percent accuracy
    
