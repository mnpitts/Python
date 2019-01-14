# standard_library.py
"""Python Essentials: The Standard Library.
<McKenna Pitts>
<Math 320 Section 2>
<September 4, 2018>
"""
import sys
import random
import box
import time
import calculator
from itertools import combinations

# Problem 1
def prob1(L):
    """Return the minimum, maximum, and average of the entries of L
    (in that order).
    """
    return min(L), max(L), sum(L)/len(L)
    


# Problem 2
def prob2():
    """Determine which Python objects are mutable and which are immutable.
    Test numbers, strings, lists, tuples, and sets. Print your results.
    """
    int1 = 1
    int2 = int1
    int2 = int2 + 1
    if int2 == int1:
    	print("Integers are mutable")
    else:
        print("Integers are immutable")
    
    str1 = "Hello"
    str2 = str1
    str2 = "Goodbye"
    if str2 == str1:
        print("Strings are mutable")
    else:
        print("Strings are immutable")
    
    list1 = [1, 2, 3, 4]
    list2 = list1
    list2[0] = 0
    if list2 == list1:
        print("Lists are mutable")
    else:
        print("Lists are immutable")
    
    tuple1 = (1, 2, 3, 4)
    tuple2 = tuple1
    tuple2 += (1,)
    if tuple2 == tuple1:
        print("Tuples are mutable")
    else:
        print("Tuples are immutable")
    
    set1 = {"apple", "bear", "cottage"}
    set2 = set1
    set2.add("dog")
    if set2 == set1:
        print("Sets are mutable")
    else:
        print("Sets are immutable")


# Problem 3
def hypot(a, b):
    """Calculate and return the length of the hypotenuse of a right triangle.
    Do not use any functions other than those that are imported from your
    'calculator' module.

    Parameters:
        a: the length one of the sides of the triangle.
        b: the length the other non-hypotenuse side of the triangle.
    Returns:
        The length of the triangle's hypotenuse.
    """
      
    return calculator.math.sqrt(calculator.sum(calculator.product(a, a), calculator.product(b, b)))


# Problem 4
def power_set(A):
    """Use itertools to compute the power set of A.

    Parameters:
        A (iterable): a str, list, set, tuple, or other iterable collection.

    Returns:
        (list(sets)): The power set of A as a list of sets.
    """

    my_list = []
    
    for i in range(0, len(A) + 1):
        new_list = set(combinations(A, i))
        my_list.extend(new_list)
    for i in range(len(my_list)):
    	my_list[i] = set(my_list[i])
    
    #print(my_list)
    return(my_list)
    


# Problem 5: Implement shut the box.
def shut_the_box(name, time_left):

    
    list_numbers = list(range(1, 10))
    dice_numbers = list(range(1, 6))
    roll_dice = 0
    time_left = int(time_left) 
    possible = True
    
    while(possible):
        start = time.time()
        good = True
        if min(list_numbers) + max(list_numbers) < 7:
            roll = random.choice(dice_numbers)
        else:
            roll = random.choice(dice_numbers) + random.choice(dice_numbers)
        print("Numbers left:", list_numbers)
        print("Roll:", roll)
        
        if(box.isvalid(roll, list_numbers) == False):
            print("Game over!", "\n")
            print("Score for player", name, ":", sum(list_numbers), " points")
            print("Time played:", round(time_left, 2), " seconds")
            print("Better luck next time!")
            return
        
        print("Seconds left", round(time_left, 2))
        while(good == True):
            eliminate = input("Numbers to Eliminate: ")
            print("")
            remove_numbers = box.parse_input(eliminate, list_numbers)
            if not remove_numbers:
                print("Invalid input")
            elif not sum(remove_numbers) == roll:
                print("Invalid input")
            else:
                for i in remove_numbers:
                    list_numbers.remove(i)
                good = False
            end = time.time()
            time_left = time_left - (end - start)
            if(time_left <= 0):
                print("Game over!", "\n")
                print("Score for player", name, ":", sum(list_numbers), " points")
                print("Time played:", round(time_left, 2), " seconds")
                print("Better luck next time!")
                possible = False
                return
            if not list_numbers:
                possible = False
                print("Score for player", name, ":", sum(list_numbers), " points")
                print("Time played:", round(time_left, 2), " seconds")
                print("Congratulations! You shut the box :)")
                return
                

if __name__ == "__main__":   
    
    if len(sys.argv) == 3:
        shut_the_box(sys.argv[1], sys.argv[2])



    
    
    
    
    
    