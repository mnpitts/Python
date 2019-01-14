# markov_chains.py
"""Volume II: Markov Chains.
<McKenna Pitts>
<ACME>
<November 5>
"""

import numpy as np
import scipy.linalg as la


# Problem 1
def random_chain(n):
    """Create and return a transition matrix for a random Markov chain with
    'n' states. This should be stored as an nxn NumPy array.
    """
    A = np.random.rand(n,n)					#Make a random matrix A
    A = A/np.sum(A, axis=0)   				#Make it column stochastic
    return A


# Problem 2
def forecast(days):
    """Forecast tomorrow's weather given that today is hot."""
    transition = np.array([[0.7, 0.6], [0.3, 0.4]])

    predictions = []
    state = 0													#initialize weather to hot
    for i in range(days):
        state = np.random.binomial(1, transition[1,state])		#sample from a binomial distribution
        predictions.append(state)								#append predictions
    return predictions
    
    # Sample from a binomial distribution to choose a new state.
    #return np.binomial(1, transition[1, 0])


# Problem 3
def four_state_forecast(days):
    """Run a simulation for the weather over the specified number of days,
    with mild as the starting state, using the four-state Markov chain.
    Return a list containing the day-by-day results, not including the
    starting day.

    Examples:
        >>> four_state_forecast(3)
        [0, 1, 3]
        >>> four_state_forecast(5)
        [2, 1, 2, 1, 1]
    """
    transition = np.array([[.5,.3,.1,0],[.3,.3,.3,.3],[.2,.3,.4,.5],[.0,.1,.2,.2]])
    predictions = []
    state = 1										#initialize weather to mild
    for i in range(days):							#sample from a multinomial distribution
        state = np.argmax(np.random.multinomial(1, transition[:,state]))
        predictions.append(state)					#append predictions
    return predictions


# Problem 4
def steady_state(A, tol=1e-12, N=40):
    """Compute the steady state of the transition matrix A.

    Inputs:
        A ((n,n) ndarray): A column-stochastic transition matrix.
        tol (float): The convergence tolerance.
        N (int): The maximum number of iterations to compute.

    Raises:
        ValueError: if the iteration does not converge within N steps.

    Returns:
        x ((n,) ndarray): The steady state distribution vector of A.
    """
    x = np.random.rand(len(A))						#make a random state distribution vector
    x = x/np.sum(x)
    for i in range(N):
        x_n = A.dot(x)
        if la.norm(x_n-x) < tol:					#if norm < tol, return x_n
            return x_n
        x = x_n
    raise ValueError("A does not converge")			#if A takes more than N iterations, raise ValueError


# Problems 5 and 6
class SentenceGenerator(object):
    """Markov chain creator for simulating bad English.

    Attributes:
        (what attributes do you need to keep track of?)

    Example:
        >>> yoda = SentenceGenerator("Yoda.txt")
        >>> print(yoda.babble())
        The dark side of loss is a path as one with you.
    """
    def __init__(self, filename):
        """Read the specified file and build a transition matrix from its
        contents. You may assume that the file has one complete sentence
        written on each line.
        """
        self.filename = filename
        with open(filename, 'r') as myfile:
            mylist = list(myfile.read().split())
        with open(filename, 'r') as myfile:
            mylinelist = myfile.readlines()					#read lines into a list
        self.states = ['$tart']
        n = len(set(mylist)) + 2							#find number of unique words
        self.trans = np.zeros((n,n))						#initialize matrix
        for i in range(len(mylinelist)):
            mylinelist[i] = mylinelist[i].split()			#split lines into words
            for j in range(len(mylinelist[i])):
                if mylinelist[i][j] not in self.states:
                    self.states.append(mylinelist[i][j])	#put each word into the states list if not already theref
                mylinelist[i][j] = self.states.index(mylinelist[i][j])		#change word to the row/column number it is associated with
            self.trans[mylinelist[i][0],0] = 1
            self.trans[n-1, mylinelist[i][len(mylinelist[i])-1]] = 1		#initialize all transitions to 1
            for j in range(len(mylinelist[i])-1):
                self.trans[mylinelist[i][j+1], mylinelist[i][j]] = 1
        self.states.append('$top')
        self.trans[n-1, n-1] = 1
        self.trans = self.trans/np.sum(self.trans, axis=0)
        
        return

    def babble(self):
        """Begin at the start sate and use the strategy from
        four_state_forecast() to transition through the Markov chain.
        Keep track of the path through the chain and the corresponding words.
        When the stop state is reached, stop transitioning and terminate the
        sentence. Return the resulting sentence as a single string.
        """
        predictions = []
        state = 0										#initialize state to $tart
        while state != len(self.trans) - 1:
            state = np.argmax(np.random.multinomial(1, self.trans[:,state]))	#sample from multinomial distribution
            if state != len(self.trans) - 1:
                predictions.append(state)				#append state to predictions
            my_sentence = ""
            for i in range(len(predictions)):
                if i != len(predictions) -1:			#build the sentence by adding each word
                    my_sentence = my_sentence + self.states[predictions[i]] + " "
                else: 
                    my_sentence = my_sentence + self.states[predictions[i]]
        return my_sentence
                
                
                
                
