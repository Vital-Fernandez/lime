class Person:

    # Constructor
    def __init__(self, name):
        self.name = name
        self.age = 30

    # To get name
    def getName(self):
        return self.name

    # To check if this person is an employee
    def isEmployee(self):
        return False

class Soul:

    def __init__(self, person):

        self.identity = person
        self._name = person.name
        self._age = person.age

        return

# Inherited or Subclass (Note Person in bracket)
class Employee(Person):

    def __init__(self, name):

        super().__init__(name)
        self.job = 'a'
        self._soul = Soul(self)

        return

    # Here we return true
    def isEmployee(self):
        return True



def changing_my_dict(input_dict):
    input_dict['A'] = 'a'
    input_dict['B'] = 'b'
    return input_dict


# # Driver code
# emp = Person("Geek1")  # An Object of Person
# print(emp.getName(), emp.isEmployee())

emp = Employee("Geek2")  # An Object of Employee
print(emp.getName(), emp.isEmployee(), emp.job)

print('Emp name', emp.name)
print('Soul name', emp._soul._name)
print('Soul identity', emp._soul.identity.name)

emp.name = 'Pedro'
print('New emp name', emp.name)
print('New Soul name', emp._soul._name)
print('New Soul identity', emp._soul.identity.name)

myDict = {'caso':'CASO', 'A': 'AAAAA'}
print(myDict)
myDict.setdefault('A', 'a')
myDict.setdefault('B', 'b')
print(myDict)
