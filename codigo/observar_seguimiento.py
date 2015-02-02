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

    def dibujar_seguimiento(self, img, topleft, bottomright, fue_exitoso,
                            es_deteccion):
        if len(img.shape) == 2:
            # Convierto a imagen a color para dibujar un cuadrado
            filas, columnas = img.shape
            color_img = np.zeros((filas, columnas, 3), dtype=np.uint8)
            color_img[:, :, 0] = img[:, :]
            color_img[:, :, 1] = img[:, :]
            color_img[:, :, 2] = img[:, :]
        else:
            color_img = img

        # Cuadrado verde si proviene del seguimiento
        # Rojo si proviene de una deteccion (ya sea porque se perdio el objeto o
        # porque recien comienza el algoritmo)
        if fue_exitoso:
            if es_deteccion:
                color_img = dibujar_cuadrado(
                    color_img,
                    topleft,
                    bottomright,
                    color=(0, 0, 255)
                )
            else:
                color_img = dibujar_cuadrado(
                    color_img,
                    topleft,
                    bottomright,
                    color=(0, 255, 0)
                )

        return color_img

    def run(self, img_provider, topleft, bottomright, fue_exitoso, es_deteccion,
            frenar=True):
        """
        Deben implementarlo las subclases
        """
        pass

    def close(self):
        """
        En caso que se requiera tomar alguna medida al terminar de capturar las
        imagenes, este es el metodo que se debe utilizar
        """
        cv2.destroyAllWindows()


class MuestraSeguimientoEnVivo(MuestraDelSeguimiento):

    def run(self, img_provider, topleft, bottomright, fue_exitoso,
            es_deteccion, frenar=True):

        img_list = img_provider.image_list()

        for i, img in enumerate(img_list):
            img_with_rectangle = self.dibujar_seguimiento(
                img,
                topleft,
                bottomright,
                fue_exitoso,
                es_deteccion,
            )

            # Muestro el resultado y espero que se apriete la tecla q
            cv2.imshow(self.name + ' ' + unicode(i), img_with_rectangle)

        if frenar:
            while cv2.waitKey(1) & 0xFF != ord('q'):
                pass


class MuestraBusquedaEnVivo(MuestraDelSeguimiento):
    def dibujar_seguimiento(self, img, topleft, bottomright, *args, **kwargs):
        if len(img.shape) == 2:
            # Convierto a imagen a color para dibujar un cuadrado
            filas, columnas = img.shape
            color_img = np.zeros((filas, columnas, 3), dtype=np.uint8)
            color_img[:, :, 0] = img[:, :]
            color_img[:, :, 1] = img[:, :]
            color_img[:, :, 2] = img[:, :]
        else:
            color_img = img

        # Cuadrado azul
        color_img = dibujar_cuadrado(
            color_img,
            topleft,
            bottomright,
            color=(255, 0, 0)
        )

        return color_img

    def run(self, img, topleft, bottomright, frenar=True, *args, **kwargs):
        # Necesario para no pisar la imagen original
        img_copy = img.copy()

        img_with_rectangle = self.dibujar_seguimiento(
            img_copy,
            topleft,
            bottomright,
        )

        # Muestro el resultado y espero que se apriete la tecla q
        cv2.imshow(self.name, img_with_rectangle)

        if frenar:
            while cv2.waitKey(1) & 0xFF != ord('q'):
                pass
