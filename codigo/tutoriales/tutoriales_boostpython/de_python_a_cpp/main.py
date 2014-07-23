
from cpp_main import mostrar

class Poto:
    def __init__(self, string):
        self.string = string
    
    def show(self):
        print self.string


if __name__ == '__main__':
    p = Poto('pocholo')
    mostrar(p)
