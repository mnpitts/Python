# exceptions_fileIO.py
"""Python Essentials: Exceptions and File Input/Output.
<Name>
<Class>
<Date>
"""

from random import choice


# Problem 1
def arithmagic():
    step_1 = input("Enter a 3-digit number where the first and last "
                                           "digits differ by 2 or more: ")
    if len(step_1) != 3:
        raise ValueError("Your input is not a three digit number")
    if abs(int(step_1[0]) - int(step_1[2])) < 2:
        raise ValueError("The first and last digits do not differ by 2 or more")
    step_2 = input("Enter the reverse of the first number, obtained "
                                              "by reading it backwards: ")
    if step_2 != step_1[::-1]:
        raise ValueError("That is not the reverse of the first number")                                        
    step_3 = input("Enter the positive difference of these numbers: ")
    if abs(int(step_1) - int(step_2)) != int(step_3):
        raise ValueError("That is not the difference of the two numbers")
    step_4 = input("Enter the reverse of the previous result: ")
    if step_4 != step_3[::-1]:
        raise ValueError("That is not the reverse of the difference")
    print(str(step_3), "+", str(step_4), "= 1089 (ta-da!)")


# Problem 2
def random_walk(max_iters=1e12):
        walk = 0
        directions = [1, -1]
        try:
            for i in range(int(max_iters)):
                walk += choice(directions)
        except KeyboardInterrupt:
            print("Process terminated at iteration", i)
            return walk
        print("Process completed.")
        return walk


# Problems 3 and 4: Write a 'ContentFilter' class.

class ContentFilter:
    """A ContentFilter class.  Accepts a name of a file to be read.
    
    Parameters:
        file_name (str): the name of a file to be read
        access (str): the access given to the file
    """
    
    def __init__(self, file_name):
        while True:
            try:
                my_file = open(file_name, 'r')
                self.file_name = file_name
                self.contents = my_file.read()
                my_file.close()
                break
            except Exception as e:
                file_name = input("Please enter a valid file name: ")
    
    def uniform(self, outfile, mode = 'w', case = "upper"):
        """Writes the data to an outfile with uniform case"""
        if mode != 'w' and mode != 'x' and mode != 'a':
            raise ValueError("Your input must be 'w', 'x', or 'a'")
        if case == "upper":
            with open(outfile, mode) as my_file:
                my_file.write(self.contents.upper())
        elif case == "lower":
            with open(outfile, mode) as my_file:
                my_file.write(self.contents.lower())
        else:
            raise ValueError("Your input must be 'upper' or 'lower'")
        
    def reverse(self, outfile, mode = 'w', unit = "line"):
        """Writes the data to an outfile in reverse order"""
        if mode != 'w' and mode != 'x' and mode != 'a':
            raise ValueError("Your input must be w, x, or a")
        if unit == "line":
            with open(outfile, mode) as my_file:
                r = self.contents.splitlines()
                print(r)
                for i in r[::-1]:
                    if i != '\n':
                        my_file.write(i + '\n')
        elif unit == "word":
            with open(outfile, mode) as my_file:
                t = self.contents.splitlines()
                for line in t:
                    j = line.split(' ')
                    for word in j[::-1]:
                        my_file.write(word)
                        my_file.write(' ')
                    if line != '\n':
                        my_file.write('\n')      
        else:
            raise ValueError("Your input must be 'line' or 'word'")
            
            
    def transpose(self, outfile, mode = 'w'):
        """Writes a transposed version of the data to an outfile"""
        if mode != 'w' and mode != 'x' and mode != 'a':
            raise ValueError("Your input must be 'w', 'x', or 'a'")
        else:
            with open(outfile, mode) as my_file:
                r = self.contents.splitlines()
                for i in range(len(r)):
                    r[i] = r[i].split(' ')
                for i in range(len(r[0])):
                    for j in range(len(r)):
                        my_file.write(r[j][i] + ' ')
                    my_file.write('\n')
        
        
    def __str__(self):
        """Implements the string magic method to print the name of the source file, and
        the number of total characters, alphabeitic characters, numerical characters, 
        whitespace characters, and the number of lines"""
        r = self.contents.splitlines()
        total_lines = len(r)
        white_space = sum(c.isspace() for c in self.contents)
        num_char = sum(c.isdigit() for c in self.contents)
        alpha_char = sum(c.isalpha() for c in self.contents)
        total_char = num_char + alpha_char + white_space
        file_name = self.file_name
            
        return("Source file:\t\t\t" + file_name + '\n'
            + "Total characters:\t\t" + str(total_char) + '\n'
            + "Alphabetic characters:\t\t" + str(alpha_char) + '\n'
            + "Numerical characters:\t\t" + str(num_char) + '\n'
            + "Whitespace characters:\t\t" + str(white_space) + '\n'
            + "Number of lines:\t\t" + str(total_lines))
        
        
        
        
        
        
        
        
        
                