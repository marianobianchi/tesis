#!/usr/bin/python
# -*- coding: utf-8 -*-
#Con esto, todos los strings literales son unicode (no hace falta poner u'algo')
from __future__ import unicode_literals


import os

import cv2


class ImageProvider(object):
    """
    Esta clase se va a encargar de proveer imágenes, ya sea provenientes
    de un video o tiras de frames guardadas en el disco.

    Idea: copiar parte de la API de cv2.VideoCapture (lo que haga falta)
    """
    def read(self):
        """
        Cada subclase implementa este método. Debe devolver una imagen
        distinta cada vez, simulando los frames de un video.
        """
        pass


class FramesAsVideo(ImageProvider):

    def __init__(self, path):
        """
        La carpeta apuntada por 'path' debe contener solo imagenes que seran
        devueltas por el metodo 'read'.
        Se devolveran ordenadas alfabéticamente.
        """
        self.path = path
        self.img_filenames = os.listdir(path)
        self.img_filenames.sort()
        self.img_filenames = [os.path.join(path, fn) for fn in self.img_filenames]

    def read(self):
        have_images = len(self.img_filenames) > 0
        img = None
        if have_images:
            # guardo proxima imagen
            img = cv2.imread(self.img_filenames[0])

            self.img_filenames = self.img_filenames[1:] # quito la imagen de la lista

        return (have_images, img)