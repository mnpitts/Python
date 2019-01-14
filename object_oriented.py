# object_oriented.py
"""Python Essentials: Object Oriented Programming.
<McKenna Pitts>
<Math 321 Section 2>
<September 19, 2018>
"""
import math

class Backpack:
    """A Backpack object class. Has a name, a list of contents, a color, and a max size.

    Attributes:
        name (str): the name of the backpack's owner.
        contents (list): the contents of the backpack.
        color (str): the color of the backpack.
        max_size (int): the holding capacity of the backpack.
        
    """

    # Problem 1: Modify __init__() and put(), and write dump().
    def __init__(self, name, color, max_size = 5):
        """Set the name, color, max_size and initialize an empty list of contents.

        Parameters:
            name (str): the name of the backpack's owner.
            color (str): the color of the backpack.
            max_size (int): the maximum size of the backpack
            contents (list): the contents of the backpack
        """
        self.name = name
        self.color = color
        self.max_size = max_size
        self.contents = []

    def put(self, item):
        """Add an item to the backpack's list of contents."""
        if len(self.contents) < self.max_size:
            self.contents.append(item)
        else:
            print("No Room!")
            
    def dump(self):
        """Remove all items from the backpack"""
        self.contents.clear()            

    def take(self, item):
        """Remove an item from the backpack's list of contents."""
        self.contents.remove(item)

    # Magic Methods -----------------------------------------------------------

    # Problem 3: Write __eq__() and __str__().
    def __add__(self, other):
        """Add the number of contents of each Backpack."""
        return len(self.contents) + len(other.contents)

    def __lt__(self, other):
        """Compare two backpacks. If 'self' has fewer contents
        than 'other', return True. Otherwise, return False.
        """
        return len(self.contents) < len(other.contents)
        
    def __eq__(self, other):
        """Compare two backpacks.  Returns True if the two objects have the same
        name, color, and number of contents.
        """
        return(self.name == other.name and self.color == other.color and len(self.contents) == len(other.contents))
        
            
    def __str__(self):
        """Returns a string represnetation of an object. 
        It will return the owner's name, backpack color, current size, maximum size, 
        and list of contents.
        """
        return("Owner:\t\t"+self.name+"\nColor:\t\t"+self.color+"\nSize:\t\t"+str(len(self.contents))+"\nMax Size:\t"+str(self.max_size)+"\nContents:\t"+str(self.contents))
        


# An example of inheritance. You are not required to modify this class.
class Knapsack(Backpack):
    """A Knapsack object class. Inherits from the Backpack class.
    A knapsack is smaller than a backpack and can be tied closed.

    Attributes:
        name (str): the name of the knapsack's owner.
        color (str): the color of the knapsack.
        max_size (int): the maximum number of items that can fit inside.
        contents (list): the contents of the backpack.
        closed (bool): whether or not the knapsack is tied shut.
    """
    def __init__(self, name, color):
        """Use the Backpack constructor to initialize the name, color,
        and max_size attributes. A knapsack only holds 3 item by default.

        Parameters:
            name (str): the name of the knapsack's owner.
            color (str): the color of the knapsack.
            max_size (int): the maximum number of items that can fit inside.
        """
        Backpack.__init__(self, name, color, max_size=3)
        self.closed = True

    def put(self, item):
        """If the knapsack is untied, use the Backpack.put() method."""
        if self.closed:
            print("I'm closed!")
        else:
            Backpack.put(self, item)

    def take(self, item):
        """If the knapsack is untied, use the Backpack.take() method."""
        if self.closed:
            print("I'm closed!")
        else:
            Backpack.take(self, item)

    def weight(self):
        """Calculate the weight of the knapsack by counting the length of the
        string representations of each item in the contents list.
        """
        return sum(len(str(item)) for item in self.contents)


# Problem 2: Write a 'Jetpack' class that inherits from the 'Backpack' class.
class Jetpack(Backpack):
    """A Jetpack object class.  Inherits from the Backpack class.
    A jetpack is smaller than a backpack and accepts an amount of fuel.
    
    Attributes:
        name (str): the name of the jetpack's owner
        color (str): the color of the jetpack
        max_size (int): the maximum number of items that can fit inside
        fuel_amount (int): the amount of fuel accepted by the jetpack
        contents (list): the contents of the jetpack
    """
    
    def __init__(self, name, color, max_siz = 2, fuel_amount = 10):
        """Use the Backpack construcktor to initialize the name, color, 
        and max_size attributes.  A Jetpack only holds 2 items by default.
        
        Parameters:
            name (str): the name of the jetpack's owner.
            color (str): the color of the jetpack.
            max_size (int): the maximum number of items that can fit inside.
            fuel_amount (int): the amount of fuel int he jetpack
            contents (list): the contents of the backpack    
        """
        self.name = name
        self.color = color
        self.max_size = max_size
        self.fuel_amount = fuel_amount
        self.contents = []
        
        
    def fly(self, fuel_used):
        """If there is sufficient fuel, 
        decrements the fuel amount by the fuel to be burned during flight.
        """
        if fuel_used > fuel_amount:
           print("Not enough fuel!")
        else: 
            self.fuel_amount -= fuel_used
            
    def dump(self):
        """Removes all contents and fuel from the jetpack."""
        Backpack.dump(self)
        self.fuel_amount = 0
        
    
        

# Problem 4: Write a 'ComplexNumber' class.

class ComplexNumber:
    """A Complex Number class.  Has a real and imaginary number.
    
    Attributes:
        real (int): the real number
        imag (int): the imaginary number
    """
    
    def __init__(self, real, imag):
        """Set the real and imaginary numbers.
        
        Parameters:
            real (int): the real number
            imag (int): the imaginary number
        """
        
        self.real = real
        self.imag = imag
        
    def conjugate(self):
        """Returns the object's complex conjugate as a new ComplexNumber object."""
        
        conj = -self.imag
        
        return(ComplexNumber(self.real, conj))
        
    def __str__(self):
        """Prints a + bi as (a + bj)"""
        if self.imag >= 0:
            return("(" + str(self.real) + "+" + str(self.imag) + "j)")
        else:
            return("(" + str(self.real) + str(self.imag) + "j)")
        
    def __abs__(self):
        """Returns the magnitude of the complex number"""
        return(math.sqrt(self.real**2 + self.imag**2))
        
    def __eq__(self, other):
        """Two complex numbers are equal if and only if 
        they have the same real and imaginary parts."""
        return(self.real == other.real and self.imag == other.imag)
            
    def __add__(self, other):
        """Returns the sum of two Complex Numbers"""
        return(ComplexNumber(self.real + other.real, self.imag + other.imag))
    
    def __sub__(self, other):
        """Returns the difference of two Complex Numbers"""
        return(ComplexNumber(self.real - other.real, self.imag - other.imag))
    
    def __mul__(self, other):
        """Returns the product of two Complex Numbers"""
        a = self.real * other.real
        b = self.real * other.imag
        c = self.imag * other.real
        d = self.imag * other.imag
        
        real_number = a - d
        imag_number = b + c
        
        return(ComplexNumber(real_number, imag_number))
        
    
    def __truediv__(self, other):
        """Returns the division of two Complex Numbers"""
        top = self * other.conjugate()
        bottom = other * other.conjugate()
        real_number = top.real/bottom.real
        imag_number = top.imag/bottom.real
        
        return(ComplexNumber(real_number, imag_number))
        
        
    
        
        
        
        



 

    
