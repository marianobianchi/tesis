#!/usr/bin/python
# -*- coding: utf-8 -*-
#Con esto, todos los strings literales son unicode (no hace falta poner u'algo')
from __future__ import unicode_literals


import numpy as np
import cv2


def espiral_desde((x, y), filas, columnas):
    sum_x = 2
    sum_y = 2
    for j in range(50):
        x += sum_x
        if (0 <= x <= filas) and (0 <= y <= columnas):
            yield (x, y)

        sum_x *= -2

        y += sum_y
        if (0 <= x <= filas) and (0 <= y <= columnas):
            yield (x, y)

        sum_y *= -2



def seguir_circulo():

    img_name = 'videos/moving_circle/mc_{i:03d}.jpg'
    img = cv2.imread(img_name.format(i=0), cv2.IMREAD_GRAYSCALE)

    filas, columnas, canales = img.shape

    # Tamaño de la región de imagen que se usará para detectar y seguir
    tam_region = 80 # Pixeles cada lado

    # Detectar objeto
    centro = (80, 80)

    # Cuadrado donde esta el objeto
    columna_izq = centro[1] - (tam_region/2)
    columna_der = columna_izq + tam_region
    fila_arriba = centro[0] - (tam_region/2)
    fila_abajo = fila_arriba + tam_region

    # Este es el objeto a seguir
    img_objeto = img[fila_arriba:fila_abajo,columna_izq:columna_der]

    # Mascara del objeto
    mask = cv2.bitwise_not(img_objeto) # Da vuelta los valores (0->255 y 255->0)



    for i in range(1, 100):
        # Imagen en escala de grises
        img = cv2.imread(img_name.format(i=i), cv2.IMREAD_GRAYSCALE)

        # Muestro el frame
        #cv2.imshow('frame', img)
        #cv2.waitKey(1)


        # Cantidad de pixeles distintos
        comp_imagenes = filas * columnas

        # Ubicacion del objeto
        obj_col_izq = columna_izq
        obj_col_der = columna_der
        obj_fil_arr = fila_arriba
        obj_fil_aba = fila_abajo


        # Seguimiento (busqueda/deteccion acotada)
        for x, y in espiral_desde(centro, filas, columnas):
            columna_izq = centro[1] - (tam_region/2)
            columna_der = columna_izq + tam_region
            fila_arriba = centro[0] - (tam_region/2)
            fila_abajo = fila_arriba + tam_region

            # Tomo una region de la imagen donde se busca el objeto
            roi = img[fila_arriba:fila_abajo,columna_izq:columna_der]

            # Hago una comparacion bit a bit de la imagen original
            # Compara solo en la zona de la máscara y deja 0's en donde hay
            # coincidencias y 255's en donde no coinciden
            xor = cv2.bitwise_xor(img_objeto, roi, mask=mask)

            # Cuento la cantidad de 0's y me quedo con la mejor comparacion
            non_zeros = cv2.countNonZero(xor)

            if non_zeros < comp_imagenes:







    cv2.destroyAllWindows()



if __name__ == '__main__':
    seguir_circulo()




