#!/usr/bin/python
# -*- coding: utf-8 -*-
#Con esto, todos los strings literales son unicode (no hace falta poner u'algo')
from __future__ import unicode_literals

import sys
import os

import cv2
import numpy as np

from cpp.common import read_pcd
from cpp.depth_to_rgb import *


class FrameNamesAndImageProvider(object):
    def __init__(self, scene_path, scene, scene_number, obj_path, obj, obj_number):
        scene_path = os.path.join(scene_path, scene)
        scene_path = os.path.join(scene_path, scene + '_' + scene_number)
        self.scene_path = scene_path
        self.scene = scene
        self.scene_number = scene_number

        obj_path = os.path.join(obj_path, obj)
        obj_path = os.path.join(obj_path, obj + '_' + obj_number)
        self.obj_path = obj_path
        self.obj = obj
        self.obj_number = obj_number

        # get the number of frames available
        filenames = os.listdir(scene_path)
        last_frame_number = 1

        for filename in filenames:
            frame_number = self._get_fnumber(filename)
            if frame_number > last_frame_number:
                last_frame_number = frame_number

        # Set initial and last frame number
        print "CAMBIAR INICIO"
        self.offset_frame_count = 5
        self.next_frame_number = self.offset_frame_count
        self.last_frame_number = last_frame_number

    def _obj_fname(self, obj_scene_number=1, frame_number=1, suffix='.pcd'):
        generic_fname = '{obj}_{obj_number}_{obj_scene_number}_{frame_number}{suffix}'

        fname = generic_fname.format(
            obj=self.obj,
            obj_number=self.obj_number,
            obj_scene_number=obj_scene_number,
            frame_number=frame_number,
            suffix=suffix,
        )
        return os.path.join(self.obj_path, fname)

    def obj_pcd(self, n=1):
        fname = self._obj_fname(frame_number=n)
        pc = read_pcd(str(fname))
        return pc

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

        generic_fname = '{scene}_{scene_number}_{nframe}'
        if is_depth:
            generic_fname += '_depth'

        fname_no_extension = generic_fname.format(
            scene=self.scene,
            scene_number=self.scene_number,
            nframe=nframe,
        )
        fname = os.path.join(self.scene_path, fname_no_extension)
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
        pc = read_pcd(str(fname))
        return pc

    def image_list(self):
        return [self.rgb_img()]#, self.rgbdepth_img()]

    def next(self):
        self.next_frame_number += 1

    def image_size(self):
        img = self.rgb_img()
        return len(img), len(img[0]) # filas, columnas


class FrameNamesAndImageProviderPreCharged(FrameNamesAndImageProvider):
    def __init__(self, scene_path, scene, scene_number, obj_path, obj, obj_number):
        (super(FrameNamesAndImageProviderPreCharged, self)
         .__init__(scene_path, scene, scene_number, obj_path, obj, obj_number))

        self._pcd_images = []

        total_files = self.last_frame_number - self.next_frame_number + 1

        for i in range(self.next_frame_number, self.last_frame_number + 1):
            sys.stdout.write(
                "Reading pcd file number " +
                str(i - self.offset_frame_count + 1) +
                "/" +
                str(total_files) +
                "\r"
            )
            sys.stdout.flush()
            fname = self._frame_fname(i)
            pc = read_pcd(str(fname))
            self._pcd_images.append(pc)
        sys.stdout.write('\n')

    def pcd(self):
        return self._pcd_images[self.next_frame_number - self.offset_frame_count]