#class 定義範例，
class Student:
    grade = 70
    def show(self):
        print('grade=', self.grade)



tom = Student()
print(tom.grade)
tom.grade = 90
tom.show()


