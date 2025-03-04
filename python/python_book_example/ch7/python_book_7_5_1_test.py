# 成員私有化
class Circle():
    def __init__(self,r=1,c='red'):
        self.__rad=r
        self.color=c
    def __area(self):
        return 3.14*self.__rad**2
    def print__area(self):
        print('area=',self.__area())

c1 = Circle(2,'yellow')

print(c1.color)

print(c1.print__area())