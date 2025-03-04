# 使用Getter和Setter的完整範例
class Circle():
    def __init__(self,r=1):
        print('__init__() called')
        self.radius = r
    def print_area(self):
        print('area = ',3.14*self.rad**2)

    @property
    def radius(self):
        print('Getter radius() called')
        return self.rad
    
    @property
    def area(self):
        print('Getter area() called')
        return 3.14*self.rad**2
    
    @radius.setter
    def radius(self,r):
        print('Setter radius() called')
        if r > 0:
            self.rad = r
        else:
            print('input error')
            self.rad = 0



c1 = Circle(2)

c1.radius
c1.print_area()
c1.area


c1.radius = -10

c1.radius = 10

print(c1.radius)