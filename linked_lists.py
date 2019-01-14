# linked_lists.py
"""Volume 2: Linked Lists.
<Name>
<Class>
<Date>
"""


# Problem 1
class Node:
    """A basic node class for storing data."""
    def __init__(self, data):
        """Store the data in the value attribute only if it is a string, int, or float."""
        if type(data) != str and type(data) != int and type(data) != float:
            raise TypeError("Data must be an integer, float, or string.")
        else:
            self.value = data


class LinkedListNode(Node):
    """A node class for doubly linked lists. Inherits from the Node class.
    Contains references to the next and previous nodes in the linked list.
    """
    def __init__(self, data):
        """Store the data in the value attribute and initialize
        attributes for the next and previous nodes in the list.
        """
        Node.__init__(self, data)       # Use inheritance to set self.value.
        self.next = None                # Reference to the next node.
        self.prev = None                # Reference to the previous node.


# Problems 2-5
class LinkedList:
    """Doubly linked list data structure class.

    Attributes:
        head (LinkedListNode): the first node in the list.
        tail (LinkedListNode): the last node in the list.
    """
    def __init__(self):
        """Initialize the head and tail attributes by setting
        them to None, since the list is empty initially.
        """
        #initialize the head, tail, and size
        self.head = None                
        self.tail = None
        self.size = 0

    def append(self, data):
        """Append a new node containing the data to the end of the list."""
        # Create a new node to store the input data.
        new_node = LinkedListNode(data)
        if self.head is None:
            # If the list is empty, assign the head and tail attributes to
            # new_node, since it becomes the first and last node in the list.
            self.head = new_node
            self.tail = new_node
            self.size = 1
        else:
            # If the list is not empty, place new_node after the tail.
            self.tail.next = new_node               # tail --> new_node
            new_node.prev = self.tail               # tail <-- new_node
            # Now the last node in the list is new_node, so reassign the tail.
            self.tail = new_node
            self.size += 1

    # Problem 2
    def find(self, data):
        """Return the first node in the list containing the data.

        Raises:
            ValueError: if the list does not contain the data.

        Examples:
            >>> l = LinkedList()
            >>> for x in ['a', 'b', 'c', 'd', 'e']:
            ...     l.append(x)
            ...
            >>> node = l.find('b')
            >>> node.value
            'b'
            >>> l.find('f')
            ValueError: <message>
        """
        #create a new node to iterate through the linkedlist
        current_node = self.head
        if current_node == None:
            raise ValueError("Data is not found in the list.")
        while current_node is not None:
            #iterate through the list until you find the data
            if current_node.value == data:
                return current_node
            else:
                current_node = current_node.next
        raise ValueError("Data is not found in the list.")

    # Problem 2
    def get(self, i):
        """Return the i-th node in the list.

        Raises:
            IndexError: if i is negative or greater than or equal to the
                current number of nodes.

        Examples:
            >>> l = LinkedList()
            >>> for x in ['a', 'b', 'c', 'd', 'e']:
            ...     l.append(x)
            ...
            >>> node = l.get(3)
            >>> node.value
            'd'
            >>> l.get(5)
            IndexError: <message>
        """
        if i >= self.size or i < 0:
           raise IndexError("There is not a node at that position.")
        else:
            #create a new node to iterate through the linkedlist
            count = 0
            current_node = self.head

            while current_node is not None:
                #iterate through the list until you get to the i-th node
                if count == i:
                    return current_node
                else:
                    current_node = current_node.next
                    count += 1


    # Problem 3
    def __len__(self):
        """Return the number of nodes in the list.

        Examples:
            >>> l = LinkedList()
            >>> for i in (1, 3, 5):
            ...     l.append(i)
            ...
            >>> len(l)
            3
            >>> l.append(7)
            >>> len(l)
            4
        """
        #return the size of the list
        return self.size

    # Problem 3
    def __str__(self):
        """String representation: the same as a standard Python list.

        Examples:
            >>> l1 = LinkedList()       |   >>> l2 = LinkedList()
            >>> for i in [1,3,5]:       |   >>> for i in ['a','b',"c"]:
            ...     l1.append(i)        |   ...     l2.append(i)
            ...                         |   ...
            >>> print(l1)               |   >>> print(l2)
            [1, 3, 5]                   |   ['a', 'b', 'c']
        """
        #start the string
        str = "["
        #create a new node to iterate through the list
        current_node = self.head
        while current_node is not None:
            #iterate through the list and add each node value to the string
            if current_node.next == None:
                #end of the list
                str += repr(current_node.value)
            else:
                #add all the nodes followed by a '
                str += repr(current_node.value)
                str += ", "
            current_node = current_node.next
        #finish the string
        str += "]"
        return str

    # Problem 4
    def remove(self, data):
        """Remove the first node in the list containing the data.

        Raises:
            ValueError: if the list is empty or does not contain the data.

        Examples:
            >>> print(l1)               |   >>> print(l2)
            ['a', 'e', 'i', 'o', 'u']   |   [2, 4, 6, 8]
            >>> l1.remove('i')          |   >>> l2.remove(10)
            >>> l1.remove('a')          |   ValueError: <message>
            >>> l1.remove('u')          |   >>> l3 = LinkedList()
            >>> print(l1)               |   >>> l3.remove(10)
            ['e', 'o']                  |   ValueError: <message>
        """
        #create a new node using the find function
        my_node = self.find(data)
        if my_node == self.head:
            #remove the head
            self.head = self.head.next
            if self.size == 1:
                self.tail = self.head
            self.size -= 1
            return data
        elif my_node == self.tail:
            #remove the tail
            self.tail = my_node.prev
            self.tail.next = None
            self.size -= 1
            return data
        else:
            #remove a node within the list
            my_node.prev.next = my_node.next
            my_node.next.prev = my_node.prev
            self.size -= 1
            return data


    # Problem 5
    def insert(self, index, data):
        """Insert a node containing data into the list immediately before the
        node at the index-th location.

        Raises:
            IndexError: if index is negative or strictly greater than the
                current number of nodes.

        Examples:
            >>> print(l1)               |   >>> len(l2)
            ['b']                       |   5
            >>> l1.insert(0, 'a')       |   >>> l2.insert(6, 'z')
            >>> print(l1)               |   IndexError: <message>
            ['a', 'b']                  |
            >>> l1.insert(2, 'd')       |   >>> l3 = LinkedList()
            >>> print(l1)               |   >>> l3.insert(1, 'a')
            ['a', 'b', 'd']             |   IndexError: <message>
            >>> l1.insert(2, 'c')       |
            >>> print(l1)               |
            ['a', 'b', 'c', 'd']        |
        """
        if index < self.size and index > 0:
            #insert a node in the middle of the list
            insert_node = self.get(index)
            new_node = LinkedListNode(data)
            new_node.next = insert_node
            new_node.prev = insert_node.prev
            insert_node.prev.next = new_node
            insert_node.prev = new_node
            self.size += 1
        elif index == 0:
            #insert a node a the beginning of a list
            new_node = LinkedListNode(data)
            if self.head != None:
            	new_node.next = self.head
            	self.head.prev = new_node
            	self.head = new_node
            else:
            	self.head = new_node
            self.size += 1
            	
        elif index == self.size:
            #insert node at the end of a list
            self.append(data)
        else:
            raise IndexError("Index is not found in the list")
            
        


# Problem 6: Deque class.
class Deque(LinkedList):
    
    def __init__(self):
        #initialize the Deque class
        #use inheritance
        LinkedList.__init__(self)
        
        
    def pop(self):
        #remove the last value (on the right) of the Deque
        return(LinkedList.remove(self, self.tail.value))
    
    def popleft(self):
        #remove the first valune (on the left0 of the Deque
        return(LinkedList.remove(self, self.head.value))
    
    def appendleft(self, data):
        #add a value to the bottomw of the Deque
        LinkedList.insert(self, 0, data)
    
    def remove(*args, **kwargs):
        #disable the remove funcion
        raise NotImplementedError("Use pop() or popleft() for removal")
    
    def insert(*args, **kwargs):
        #disable the insert function
        raise NotImplementedError("Use append() or appendleft() for insertion")

# Problem 7
def prob7(infile, outfile):
    """Reverse the contents of a file by line and write the results to
    another file.

    Parameters:
        infile (str): the file to read from.
        outfile (str): the file to write to.
    """
    #create a new deque to store in lines in the file
    mydeque = Deque()
    with open(infile, 'r') as my_file:
        #for each line append it to the deque
        for line in my_file:
            mydeque.append(line)
            
    with open(outfile, 'w') as new_file:
        while mydeque.size > 0:
            #add each line from the deque to the new file
            new_file.write(str(mydeque.pop()))
    return  

