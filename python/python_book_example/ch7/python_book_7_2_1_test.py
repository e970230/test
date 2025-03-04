class Circle:
    cnt = 0 #類別屬性
    def __init__(self,r=1):
        self.rad=r
        Circle.cnt+=1   #將類別屬性cnt值+1
    def area(self):
        return 3.14*self.rad**2
    

c1=Circle(300)

print(c1.rad)

print(c1.area())

c2 = Circle(50)

print('有多少圓',c1.cnt)
print('有多少圓',c2.cnt)



print([Circle().cnt for _ in range(5)])

