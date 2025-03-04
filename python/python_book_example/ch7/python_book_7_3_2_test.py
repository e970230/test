# 靜態函數
class Circle:
    def __init__(self,r=1):
        self.rad = r
    def area(self):
        return 3.14*self.rad**2
    @staticmethod  #decorator
    def show_info(greeting):
        print(greeting,'Circle')
        

Circle.show_info('Hello_1')
