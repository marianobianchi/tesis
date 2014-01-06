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


def dibujar_cuadrado(img, (fila_borde_sup_izq, col_borde_sup_izq), tam_region, color=(0,0,0)):
    cv2.rectangle(
        img,
        (col_borde_sup_izq, fila_borde_sup_izq),
        (col_borde_sup_izq+tam_region, fila_borde_sup_izq+tam_region),
        color,
        3
    )
    return img


def ver_seguimiento(img,
                    frame_title,
                    nueva_ubicacion,
                    tam_region,
                    vieja_ubicacion,
                    frenar=False):

    filas, columnas = img.shape

    # Convierto a imagen a color para dibujar un cuadrado
    color_img = np.zeros((filas, columnas, 3), dtype=np.uint8)
    color_img[:,:,0] = img[:,:]
    color_img[:,:,1] = img[:,:]
    color_img[:,:,2] = img[:,:]

    # Cuadrado verde si hubo una coincidencia/cambio de lugar
    # Rojo si no hubo coincidencia alguna
    if nueva_ubicacion == vieja_ubicacion:
        color_img = dibujar_cuadrado(color_img, nueva_ubicacion, tam_region, color=(0,0,255))
    else:
        color_img = dibujar_cuadrado(color_img, nueva_ubicacion, tam_region, color=(0,255,0))

    # Muestro el resultado y espero que se apriete la tecla q
    cv2.imshow(frame_title, color_img)
    if frenar:
        while cv2.waitKey(1) & 0xFF != ord('q'):
            pass


def seguir_circulo():

    img_name = 'videos/moving_circle/mc_{i:03d}.jpg'
    img = cv2.imread(img_name.format(i=0), cv2.IMREAD_GRAYSCALE)

    filas, columnas = img.shape

    # Tamaño de la región de imagen que se usará para detectar y seguir
    tam_region = 80 # Pixeles cada lado

    # Detectar objeto: Cuadrado donde esta el objeto
    vieja_ubicacion = (40, 40) # Fila, columna
    nueva_ubicacion = vieja_ubicacion

    # Este es el objeto a seguir
    img_objeto = img[vieja_ubicacion[0]:vieja_ubicacion[0]+tam_region,
                     vieja_ubicacion[1]:vieja_ubicacion[1]+tam_region]

    # Mascara del objeto
    mask = cv2.bitwise_not(img_objeto) # Da vuelta los valores (0->255 y 255->0)



    for i in range(1, 100):
        # Imagen en escala de grises
        img = cv2.imread(img_name.format(i=i), cv2.IMREAD_GRAYSCALE)

        # Cantidad de pixeles distintos
        comp_imagenes = filas * columnas

        # Seguimiento (busqueda/deteccion acotada)
        for x, y in espiral_desde(vieja_ubicacion, tam_region, filas, columnas):
            col_izq = y
            col_der = col_izq + tam_region
            fil_arr = x
            fil_aba = fil_arr + tam_region

            # Tomo una region de la imagen donde se busca el objeto
            roi = img[fil_arr:fil_aba,col_izq:col_der]

            # Si se quiere ver como va buscando, descomentar la siguiente linea
            # ver_seguimiento(img, 'Buscando el objeto', (x,y), tam_region, vieja_ubicacion)

            # Hago una comparacion bit a bit de la imagen original
            # Compara solo en la zona de la máscara y deja 0's en donde hay
            # coincidencias y 255's en donde no coinciden
            xor = cv2.bitwise_xor(img_objeto, roi, mask=mask)

            # Cuento la cantidad de 0's y me quedo con la mejor comparacion
            non_zeros = cv2.countNonZero(xor)

            if non_zeros < comp_imagenes:
                print "NUEVA UBICACION: x={x} y={y}".format(x=x, y=y)
                # Nueva ubicacion del objeto (esquina superior izquierda del cuadrado)
                nueva_ubicacion = (x, y)

                # Actualizo la cantidad de pixeles distintos
                comp_imagenes = non_zeros

        # Muestro el seguimiento para hacer pruebas
        ver_seguimiento(
            img,
            'Seguimiento',
            nueva_ubicacion,
            tam_region,
            vieja_ubicacion,
            True
        )

# si son igual volver a detectar
        vieja_ubicacion = nueva_ubicacion
        print "Vieja Ubicacion: x={x} y={y}".format(x=vieja_ubicacion[0], y=vieja_ubicacion[1])









    cv2.destroyAllWindows()



if __name__ == '__main__':
    seguir_circulo()
