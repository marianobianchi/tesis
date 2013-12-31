#!/usr/bin/python
# -*- coding: utf-8 -*-

#Con esto, todos los strings literales son unicode (no hace falta poner u'algo')
from __future__ import unicode_literals

import numpy as np
import cv2


def tomar_y_cambiar_valor_de_pixel():
    img = cv2.imread('RSCN1018.JPG')

    # Tomar los pixeles BGR
    blue, green, red = img[200,100]
    print "Azul: {a}, Verde: {v}, Rojo: {r}".format(a=blue, v=green, r=red)

    # Editar pixeles BGR
    img[200,100] = [12, 30, 240]
    print "Azul: {a}, Verde: {v}, Rojo: {r}".format(a=blue, v=green, r=red)

    # Tomar de a un pixel, es mejor usando "item"
    blue = img.item(200,100,0)
    print "Azul: {a}".format(a=blue)

    # Setear un pixel
    img.itemset((200,100,0), 30)
    print "Azul: {a}".format(a=img.item(200,100,0))


def tomar_datos_de_imagen():
    img = cv2.imread('RSCN1018.JPG')

    filas, columnas, canales = img.shape
    print "Alto: {a}, Ancho: {v}, Canales: {r}".format(a=filas, v=columnas, r=canales)

    cant_pixeles = img.size
    print "Pixeles totales: {p}".format(p=cant_pixeles)

    print "IMPORTANTE:"
    print "Tipo de datos de la imagen: {t}".format(t=img.dtype)


def sumar_imagenes():
    x = np.uint8([250])
    y = np.uint8([10])

    print "La suma segun opencv satura: {s}".format(s=cv2.add(x,y))
    print "La suma segun numpy funciona como modulo: {s}".format(s=x+y)


def pegar_logo_en_imagen():
    # Cargo la imagen y el logo
    img = cv2.imread('RSCN1018.JPG')
    logo = cv2.imread('wiki_logo_p.JPG')

    # I want to put logo on top-left corner, So I create a ROI
    # Vamos a poner el logo en la esquina izquierda-arriba. Creamos una region de imagen (ROI)
    rows,cols,channels = logo.shape
    roi = img[0:rows, 0:cols ]

    # Now create a mask of logo and create its inverse mask also
    logogray = cv2.cvtColor(logo,cv2.COLOR_BGR2GRAY) # Escala de grises
    ret, mask = cv2.threshold(logogray, 10, 255, cv2.THRESH_BINARY) # Deja 0 en valores <= 10 y 255 en los mayores
    mask_inv = cv2.bitwise_not(mask) # Da vuelta los valores (0->255 y 255->0)

    # En mask queda el logo con blanco en la parte del dibujo y negro en el fondo
    # inv mask queda al reves de mask

    # Con esto dejamos en negro el logo dentro de la ROI
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

    # Con esto dejamos en img2_fg solo la parte del logo
    img2_fg = cv2.bitwise_and(logo, logo, mask=mask)

    # Ponemos el logo en ROI
    dst = cv2.add(img1_bg, img2_fg)

    # Modificamos la imagen principal
    img[0:rows, 0:cols ] = dst

    cv2.imshow('res',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == '__main__':
    pegar_logo_en_imagen()