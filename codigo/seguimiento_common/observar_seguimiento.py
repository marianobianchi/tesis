#!/usr/bin/python
# -*- coding: utf-8 -*-
#Con esto, todos los strings literales son unicode (no hace falta poner u'algo')
from __future__ import unicode_literals


import numpy as np
import cv2

from metodos_comunes import dibujar_cuadrado




class MuestraDelSeguimiento(object):

    def __init__(self, nombre):
        self.name = nombre

    def dibujar_seguimiento(self, img, ubicacion, tam_region, lo_siguio):
        if len(img.shape) == 2:
            # Convierto a imagen a color para dibujar un cuadrado
            color_img = np.zeros((filas, columnas, 3), dtype=np.uint8)
            color_img[:,:,0] = img[:,:]
            color_img[:,:,1] = img[:,:]
            color_img[:,:,2] = img[:,:]
        else:
            color_img = img

        # Cuadrado verde si proviene del seguimiento
        # Rojo si proviene de una deteccion (ya sea porque se perdio el objeto o
        # porque recien comienza el algoritmo)
        if not lo_siguio:
            color_img = dibujar_cuadrado(color_img, ubicacion, tam_region, color=(0,0,255))
        else:
            color_img = dibujar_cuadrado(color_img, ubicacion, tam_region, color=(0,255,0))

        return color_img

    def run(self, img, nombre, ubicacion, tam_region, lo_siguio, frenar=False):
        """
        Deben implementarlo las subclases
        """
        pass

    def close(self):
        """
        En caso que se requiera tomar alguna medida al terminar de capturar las
        imagenes, este es el metodo que se debe utilizar
        """
        pass


class MuestraSeguimientoEnVivo(MuestraDelSeguimiento):

    def run(self, img, ubicacion, tam_region, lo_siguio, frenar=False):
        img_with_rectangle = self.dibujar_seguimiento(img, ubicacion, tam_region, lo_siguio)

        # Muestro el resultado y espero que se apriete la tecla q
        cv2.imshow(self.name, img_with_rectangle)
        if frenar:
            while cv2.waitKey(1) & 0xFF != ord('q'):
                pass


class GrabaSeguimientoEnArchivo(MuestraDelSeguimiento):

    def __init__(self, nombre):
        self.name = nombre
        self.grabador = None

    def run(self, img, ubicacion, tam_region, lo_siguio, frenar=False):
        img_with_rectangle = self.dibujar_seguimiento(img, ubicacion, tam_region, lo_siguio)

        if self.grabador is None:
            # Defino el codec y armo el objeto VideoWriter
            fourcc = cv2.cv.CV_FOURCC(b'X',b'V',b'I',b'D')

            self.grabador = cv2.VideoWriter(
                self.name, # Video file name
                fourcc, # Codec
                6.0, # Frames por segundo
                (len(img[0]),len(img)), # Tamaño de los frames
                True, # Flag de color. True = color, False = gris
            )

        self.grabador.write(img_with_rectangle)

    def close(self):
        if self.grabador is not None:
            self.grabador.release()


class MuestraBusquedaEnVivo(MuestraSeguimientoEnVivo):
    def dibujar_seguimiento(self, img, ubicacion, tam_region, lo_siguio):
        if len(img.shape) == 2:
            # Convierto a imagen a color para dibujar un cuadrado
            color_img = np.zeros((filas, columnas, 3), dtype=np.uint8)
            color_img[:,:,0] = img[:,:]
            color_img[:,:,1] = img[:,:]
            color_img[:,:,2] = img[:,:]
        else:
            color_img = img

        # Cuadrado azul
        color_img = dibujar_cuadrado(color_img, ubicacion, tam_region, color=(255,0,0))

        return color_img