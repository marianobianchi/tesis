#!/usr/bin/python
# -*- coding: utf-8 -*-
#Con esto, todos los strings literales son unicode (no hace falta poner u'algo')
from __future__ import unicode_literals


import numpy as np
import cv2


def espiral_desde((x, y), tam_region, filas, columnas):
    sum_x = 2
    sum_y = 2
    for j in range(50):
        # Hago 2 busquedas por cada nuevo X
        x += sum_x/2
        if (0 <= x <= (x+tam_region) <= filas) and (0 <= y <= (y+tam_region) <= columnas):
            yield (x, y)

        x += sum_x/2
        if (0 <= x <= (x+tam_region) <= filas) and (0 <= y <= (y+tam_region) <= columnas):
            yield (x, y)

        sum_x *= -2

        # Hago 2 busquedas por cada nuevo Y
        y += sum_y/2
        if (0 <= x <= (x+tam_region) <= filas) and (0 <= y <= (y+tam_region) <= columnas):
            yield (x, y)

        y += sum_y/2
        if (0 <= x <= (x+tam_region) <= filas) and (0 <= y <= (y+tam_region) <= columnas):
            yield (x, y)

        sum_y *= -2


def dibujar_cuadrado(img, esq_izq_arriba, tam_region, color=(0,0,0)):
    esq_der_abajo = (esq_izq_arriba[0] + tam_region, esq_izq_arriba[1] + tam_region)
    cv2.rectangle(img, esq_izq_arriba, esq_der_abajo, color, 3)
    return img


def seguir_circulo():

    img_name = 'videos/moving_circle/mc_{i:03d}.jpg'
    img = cv2.imread(img_name.format(i=0), cv2.IMREAD_GRAYSCALE)

    filas, columnas = img.shape

    # Tamaño de la región de imagen que se usará para detectar y seguir
    tam_region = 80 # Pixeles cada lado

    # Detectar objeto: Cuadrado donde esta el objeto
    columna_izq = 40
    fila_arriba = 40

    # Este es el objeto a seguir
    img_objeto = img[fila_arriba:fila_arriba+tam_region,columna_izq:columna_izq+tam_region]

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

        # Nueva ubicacion
        nueva_ubicacion = (fila_arriba, columna_izq)

        import rpdb2
        rpdb2.start_embedded_debugger("pass")


        # Seguimiento (busqueda/deteccion acotada)
        for x, y in espiral_desde((fila_arriba, columna_izq), tam_region, filas, columnas):
            col_izq = y
            col_der = col_izq + tam_region
            fil_arr = x
            fil_aba = fil_arr + tam_region

            # Tomo una region de la imagen donde se busca el objeto
            roi = img[fil_arr:fil_aba,col_izq:col_der]

            # Hago una comparacion bit a bit de la imagen original
            # Compara solo en la zona de la máscara y deja 0's en donde hay
            # coincidencias y 255's en donde no coinciden
            xor = cv2.bitwise_xor(img_objeto, roi, mask=mask)

            # Cuento la cantidad de 0's y me quedo con la mejor comparacion
            non_zeros = cv2.countNonZero(xor)

            if non_zeros < comp_imagenes:
                # Nueva ubicacion del objeto (esquina superior izquierda del cuadrado)
                nueva_ubicacion = (x, y)

                # Actualizo la cantidad de pixeles distintos
                comp_imagenes = non_zeros


        # Cuadrado verde si hubo una coincidencia/cambio de lugar
        # Rojo si no hubo coincidencia alguna
        color_img = np.zeros((3, filas, columnas))
        color_img[0] = img[:,:]
        color_img[1] = img[:,:]
        color_img[2] = img[:,:]
        if nueva_ubicacion == (fila_arriba, columna_izq):
            color_img = dibujar_cuadrado(color_img, nueva_ubicacion, tam_region, color=(0,0,255))
        else:
            color_img = dibujar_cuadrado(color_img, nueva_ubicacion, tam_region, color=(0,255,0))


        fila_arriba, columna_izq = nueva_ubicacion


        # Muestro el resultado y espero que se apriete la tecla q
        cv2.imshow('Lo que se encontro', color_img)
        while cv2.waitKey(1) & 0xFF != ord('q'):
            pass









    cv2.destroyAllWindows()



if __name__ == '__main__':
    seguir_circulo()
