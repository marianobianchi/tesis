

import funciones_y_tipos

if __name__ == "__main__":
    ip = funciones_y_tipos.IntPair()
    ip.first = 10
    ip.second = 5
    
    fp = funciones_y_tipos.dame_par("Hola tarolas", ip)
    
    print "(", fp.first, fp.second, ") == ( 20.0 2.5 )"
