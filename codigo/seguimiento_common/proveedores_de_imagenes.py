#!/usr/bin/python
# -*- coding: utf-8 -*-
#Con esto, todos los strings literales son unicode (no hace falta poner u'algo')
from __future__ import unicode_literals


import os

import cv2
import pcl


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

    def _actual_read(self, filename):
        img = cv2.imread(filename, cv2.IMREAD_COLOR)
        return img

    def read(self):
        have_images = len(self.img_filenames) > 0
        img = None
        if have_images:
            # guardo proxima imagen
            img = self._actual_read(self.img_filenames[0])

            self.img_filenames = self.img_filenames[1:] # quito la imagen de la lista

        return (have_images, img)


class GrayFramesAsVideo(FramesAsVideo):

    def _actual_read(self, filename):
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        return img


class DepthFramesAsVideo(ImageProvider):
    def __init__(self, path, objname, number):
        path = os.path.join(path, objname)
        path = os.path.join(path, objname + '_' + number)
        self.path = path
        self.objname = objname
        self.number = number

        # get the number of frames available
        filenames = os.listdir(path)

        last_frame_number = 1

        for filename in filenames:
            frame_number = self._get_fnumber(filename)
            if frame_number > last_frame_number:
                last_frame_number = frame_number

        self.next_frame_number = 1
        self.last_frame_number = last_frame_number

    def _get_fnumber(self, fname):
        parts = fname.split('.')[0].split('_')
        fnumber = None
        for p in reversed(parts):
            if p.isdigit():
                fnumber = int(p)
                break

        if fnumber is None:
            raise Exception('No se encontro el numero de frame')

        return fnumber

    def _actual_read(self, nframe):
        generic_fname = '{objname}_{number}_{nframe}'

        fname_no_extension = generic_fname.format(
            objname=self.objname,
            number=self.number,
            nframe=nframe,
        )
        img_filename = os.path.join(self.path, fname_no_extension + '.png')
        img = cv2.imread(img_filename, cv2.IMREAD_COLOR)

        pcd_filename = os.path.join(self.path, fname_no_extension + '.pcd')
        pcd = pcl.PointCloud()
        pcd.from_file(pcd_filename)

        return img, pcd

    def read(self):
        have_images = self.next_frame_number <= self.last_frame_number
        img = None
        pcd = None
        if have_images:
            # guardo proxima imagen
            img, pcd = self._actual_read(self.next_frame_number)

            self.next_frame_number +=1

        return (have_images, img, pcd)

    def nframe(self):
        """
        Returns the last frame number that was read
        """
        return self.next_frame_number - 1