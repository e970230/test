# 添加比較功能
class Circle:
    def __init__(self,r):
        self.rad = r
    def compare(self,obj):
        if self.rad > obj.rad:
            return self
        else:
            return obj
        

c1=Circle(3)
c2=Circle(5)

result = c1.compare(c2)

print(result)