# 使用 @property 修飾子

class Circle():
    @property       #getter
    def radius(self):
        print('Getter called')
        return self.rad
    @radius.setter      #setter, 名稱必須和 getter 一樣
    def radius(self,r):
        print('Setter called')
        self.rad = r


c1 = Circle()

c1.radius = 10

print(c1.radius)