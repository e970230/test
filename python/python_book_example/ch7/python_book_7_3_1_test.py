# 類別函數
class Circle:
    cnt = 0
    def __init__(self):
        __class__.cnt+=1
    @classmethod
    def show_count(cls):
        print(cls.cnt,'obj(s) created')
    

Circle.show_count()

c1 = Circle()

c1.show_count()

cirs = [Circle() for _ in range(5)]

Circle.show_count()
