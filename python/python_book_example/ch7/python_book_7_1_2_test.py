class Cat:
    breed = 'Mix'
    def __init__(self,n,a):
        self.name = n
        self.age = a


myCat = Cat('Tom',8)


print(Cat.breed)
print(myCat.name)
print(myCat.age)