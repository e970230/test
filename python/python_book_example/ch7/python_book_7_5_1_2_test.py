# 撰寫 Getter 與 Setter
class Circle():
    def __init__(self,r=1):
        self.set_rad(r)     #直接呼叫 Setter
    def get_rad(self):      #定義公有的 Getter
        print('Getter called')
        return self.__rad
    def set_rad(self,r):    #定義公有的 Setter
        print('Setter called')
        if r > 0:
            self.__rad = r
        else :
            print('Input error')
            self.__rad=0

c1 = Circle(6)

print(c1.get_rad())

c1.set_rad(12)

print(c1.get_rad())

c1.set_rad(-12)

print(c1.get_rad())

print("巧克力厚片到此一遊")