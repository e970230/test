class Sphere():
    def __init__(self,r):
        self.rad = r
    def __repr__(self):
        return f'Sphere object,rad = {self.rad}'
    def volume(self):
        return 4/3*3.14*self.rad**3
    def surface_area(self):
        return 4*3.14*self.rad**2
    
s0 = Sphere(2)

print(s0)
print('體積',round(s0.volume(),5))
print('表面積',s0.surface_area())
