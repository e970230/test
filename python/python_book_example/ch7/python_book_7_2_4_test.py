class Circle:
    def __init__(self,r):
        self.rad=r
    def __repr__(self):     #當物件需被顯示時，會呼叫此函數
        return f'Circle(r={self.rad})'
    def compare(self,obj):
        if self.rad > obj.rad:
            return self
        else:
            return obj
        

c1 = Circle(5)

c2 = Circle(13)

print(c1.compare(c2))
print(c1.rad)