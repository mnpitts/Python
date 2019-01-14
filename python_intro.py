# python_intro.py
"""Python Essentials: Introduction to Python.
<McKenna Pitts>
<Math 345 Section 2>
<September 4, 2018>
"""

pi = 3.14159

def sphere_volume(r):
    """Return the volume of a sphere for a determined radius"""
    return 4/3 * pi * r**3

def isolate(a, b, c, d, e):
    """Return five spaces between the first three numbers 
    and one space between the last two"""
    print(a, b, sep = "     ", end = "     ")
    print (c, d, e, sep = " ")

def first_half(str):
    """Return the first half of a string. 
    Does not include the middle character there are an odd number of characters"""
    return str[:(len(str)//2)]

def backward(str):
    """Return the string written backwards"""
    return str[::-1]

def list_ops():
    """Modifies a given list"""
    my_list = ["bear", "ant", "cat", "dog"]
    my_list.append("eagle")
    my_list[2] = "fox"
    my_list.pop(1)
    my_list.sort()
    my_list = my_list[::-1]
    my_list[my_list.index("eagle")] = "hawk"
    my_list[-1] = my_list[-1] + "hunter"
    return my_list
    
def pig_latin(word):
    """Changes a word into pig latin"""
    if word[0] in "aeiou":
        return word + "hay"
    else:
        return word[1:] + word[0] + "ay"

def palindrome():
    """"Finds and returns the largest palindromic number 
    made from the product of two 3-digit numbers"""
    m = 0
    for i in range(999,1,-1):
        for j in range(999,1,-1):
            x = i * j
            y = str(x)
            if y == y[::-1]:
                if x > m:
                    m = x
    return m

def alt_harmonic(n):
    """Returns sum of alternating harmonic series"""
    return sum([1/n * (-1)**(n+1) for n in range(1, n + 1)])

if __name__ == "__main__":
    print("Hello, world!")
    print(sphere_volume(4))
    isolate(1, 2, 3, 4, 5)
    print(first_half("Hello, world!"))
    print(backward("Hello"))
    print(list_ops())
    print(pig_latin('racecar'))
    print(palindrome())
    print(alt_harmonic(500000))
    