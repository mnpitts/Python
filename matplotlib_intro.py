# matplotlib_intro.py
"""Python Essentials: Intro to Matplotlib.
<Name>
<Class>
<Date>
"""

import numpy as np
import matplotlib.pyplot as plt

# Problem 1
def var_of_means(n):
    """Construct a random matrix A with values drawn from the standard normal
    distribution. Calculate the mean value of each row, then calculate the
    variance of these means. Return the variance.

    Parameters:
        n (int): The number of rows and columns in the matrix A.

    Returns:
        (float) The variance of the means of each row.
    """
    A = np.random.normal(size=(n,n))
    B = A.mean(axis=1)
    return B.var()
       

def prob1():
    """Create an array of the results of var_of_means() with inputs
    n = 100, 200, ..., 1000. Plot and show the resulting array.
    """
    y = [var_of_means(n) for n in range(100, 1100, 100)]
    plt.plot(y)
    plt.show()


# Problem 2
def prob2():
    """Plot the functions sin(x), cos(x), and arctan(x) on the domain
    [-2pi, 2pi]. Make sure the domain is refined enough to produce a figure
    with good resolution.
    """
    x = np.arange(-2*np.pi, 2*np.pi, np.pi/100)
    plt.plot(x, np.sin(x))
    plt.plot(x, np.cos(x))
    plt.plot(x, np.arctan(x))
    plt.show()
    


# Problem 3
def prob3():
    """Plot the curve f(x) = 1/(x-1) on the domain [-2,6].
        1. Split the domain so that the curve looks discontinuous.
        2. Plot both curves with a thick, dashed magenta line.
        3. Set the range of the x-axis to [-2,6] and the range of the
           y-axis to [-6,6].
    """
    x = np.linspace(-2, 1, 100)
    xx = np.linspace(1, 6, 100)
    x = np.delete(x, 99)
    xx = np.delete(xx, 0)
    plt.plot(x, 1/(x-1), 'm--', linewidth = 6)
    plt.plot(xx, 1/(xx-1), 'm--', linewidth = 6)
    plt.xlim(-2, 6)
    plt.ylim(-6, 6)
    plt.show()


# Problem 4
def prob4():
    """Plot the functions sin(x), sin(2x), 2sin(x), and 2sin(2x) on the
    domain [0, 2pi].
        1. Arrange the plots in a square grid of four subplots.
        2. Set the limits of each subplot to [0, 2pi]x[-2, 2].
        3. Give each subplot an appropriate title.
        4. Give the overall figure a title.
        5. Use the following line colors and styles.
              sin(x): green solid line.
             sin(2x): red dashed line.
             2sin(x): blue dashed line.
            2sin(2x): magenta dotted line.
    """
    x = np.arange(0, 2*np.pi, np.pi/100)
    
    fun1 = plt.subplot(2, 2, 1)
    fun1.plot(x, np.sin(x), 'g-')
    fun1.set_title("y = sin(x)", fontsize = 10)
    plt.axis([0, 2*np.pi, -2, 2])
    
    fun2 = plt.subplot(2, 2, 2)
    fun2.plot(x, np.sin(2*x), 'r--')
    fun2.set_title("y = sin(2x)", fontsize = 10)
    plt.axis([0, 2*np.pi, -2, 2])
    
    fun3 = plt.subplot(2, 2, 3)
    fun3.plot(x, 2*np.sin(x), 'b--')
    fun3.set_title("y = 2sin(x)", fontsize = 10)
    plt.axis([0, 2*np.pi, -2, 2])
    
    fun4 = plt.subplot(2, 2, 4)
    fun4.plot(x, 2*np.sin(2*x), 'm:')
    fun4.set_title("y = 2sin(2x)", fontsize = 10)
    plt.axis([0, 2*np.pi, -2, 2])
    
    plt.suptitle("Sine Functions", fontsize = 15)
    plt.show()


# Problem 5
def prob5():
    """Visualize the data in FARS.npy. Use np.load() to load the data, then
    create a single figure with two subplots:
        1. A scatter plot of longitudes against latitudes. Because of the
            large number of data points, use black pixel markers (use "k,"
            as the third argument to plt.plot()). Label both axes.
        2. A histogram of the hours of the day, with one bin per hour.
            Label and set the limits of the x-axis.
    """
    fars = np.load("FARS.npy")
    
    fun1 = plt.subplot(1, 2, 1)
    fun1.plot(fars[:,1], fars[:,2], 'k,', markersize = 8, alpha = .5)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.ylim(0, 80)
    plt.axis("equal")
    
    fun2 = plt.subplot(1, 2, 2)
    fun2.hist(fars[:,0], bins = np.arange(0, 25))
    plt.xlabel("Hour")
    plt.xlim(0, 24)
    
    plt.show()


# Problem 6
def prob6():
    """Plot the function f(x,y) = sin(x)sin(y)/xy on the domain
    [-2pi, 2pi]x[-2pi, 2pi].
        1. Create 2 subplots: one with a heat map of f, and one with a contour
            map of f. Choose an appropriate number of level curves, or specify
            the curves yourself.
        2. Set the limits of each subplot to [-2pi, 2pi]x[-2pi, 2pi].
        3. Choose a non-default color scheme.
        4. Add a colorbar to each subplot.
    """
    x = np.linspace(-2*np.pi, 2*np.pi, 100)
    y = x.copy()
    X,Y = np.meshgrid(x, y)
    Z = (np.sin(X) * np.sin(Y))/(X*Y)
    
    plt.subplot(1, 2, 1)
    plt.pcolormesh(X, Y, Z, cmap = "viridis")
    plt.colorbar()
    plt.xlim(-2*np.pi, 2*np.pi)
    plt.ylim(-2*np.pi, 2*np.pi)
    #plt.axis("equal")
    
    
    fun2 = plt.subplot(1, 2, 2)
    plt.contour(X, Y, Z, cmap = "coolwarm")
    plt.colorbar()
    plt.xlim(-2*np.pi, 2*np.pi)
    plt.ylim(-2*np.pi, 2*np.pi)
    #plt.axis("equal")    
    
    plt.show()
    
