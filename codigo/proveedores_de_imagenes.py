#!/usr/bin/python
# -*- coding: utf-8 -*-
#Con esto, todos los strings literales son unicode (no hace falta poner u'algo')
from __future__ import unicode_literals

import os
import cv2
import numpy as np
import pcl

from cpp.depth_to_rgb import *


class FrameNamesAndImageProvider(object):
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

    def _frame_fname(self, nframe, is_depth=False, is_rgb=False):

        generic_fname = '{objname}_{number}_{nframe}'
        if is_depth:
            generic_fname += '_depth'

        fname_no_extension = generic_fname.format(
            objname=self.objname,
            number=self.number,
            nframe=nframe,
        )
        fname = os.path.join(self.path, fname_no_extension)
        if is_depth or is_rgb:
            fname += '.png'
        else:
            fname += '.pcd'
        return fname

    def have_images(self):
        return self.next_frame_number <= self.last_frame_number

    # Images filenames
    def rgb_fname(self):
        assert self.have_images(), 'El proveedor de imágenes no tiene más imágenes disponibles'
        return self._frame_fname(self.next_frame_number, is_rgb=True)

    def depth_fname(self):
        assert self.have_images(), 'El proveedor de imágenes no tiene más imágenes disponibles'
        return self._frame_fname(self.next_frame_number, is_depth=True)

    def pcd_fname(self):
        assert self.have_images(), 'El proveedor de imágenes no tiene más imágenes disponibles'
        return self._frame_fname(self.next_frame_number)

    # Images
    def rgb_img(self):
        fname = self.rgb_fname()
        img = cv2.imread(fname, cv2.IMREAD_COLOR)
        return img

    def rgbdepth_img(self):
        fname = self.depth_fname()
        depth_img = cv2.imread(fname, cv2.IMREAD_ANYDEPTH)

        height = len(depth_img)
        width = len(depth_img[0])
        rgbdepth_img = np.zeros((height, width,3), np.uint8)
        for r in range(height):
            for c in range(width):
                char_rgb = depth_to_rgb(int(depth_img[r][c]))
                rgbdepth_img[r][c] = [char_rgb.blue, char_rgb.green, char_rgb.red]

        return rgbdepth_img

    def depth_img(self):
        fname = self.depth_fname()
        depth_img = cv2.imread(fname, cv2.IMREAD_ANYDEPTH)
        return depth_img

    def pcd(self):
        fname = self.pcd_fname()
        pc = pcl.PointCloud()
        pc.from_file(fname)
        return pc

    def image_list(self):
        return [self.rgb_img(), self.depth_img()]

    def next(self):
        self.next_frame_number += 1

    def image_size(self):
        img = self.rgb_img()
        return len(img), len(img[0]) # filas, columnas

    def current_frame_number(self):
        """
        Gives numbers from 0 to #frames-1
        """
        return self.next_frame_number - 1
