#!/usr/bin/python
# -*- coding: utf-8 -*-
#Con esto, todos los strings literales son unicode (no hace falta poner u'algo')
from __future__ import unicode_literals

import os
import cv2


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

    def current(self):
        """
        target es la imagen en la que se está buscando al objeto
        """
        assert self.have_images(), 'El proveedor de imágenes no tiene más imágenes disponibles'
        name_dict = {}

        target_rgb_fname = self._frame_fname(
            self.next_frame_number,
            is_rgb=True,
        )
        target_depth_fname = self._frame_fname(
            self.next_frame_number,
            is_depth=True,
        )
        target_pcd_fname = self._frame_fname(self.next_frame_number)

        name_dict.update({
            'target_rgb_fname': target_rgb_fname,
            'target_pcd_fname': target_pcd_fname,
            'target_depth_fname': target_depth_fname,
        })

        if self.next_frame_number > 1:
            source_rgb_fname = self._frame_fname(
                self.next_frame_number - 1,
                is_rgb=True,
            )
            source_pcd_fname = self._frame_fname(self.next_frame_number - 1)
            source_depth_fname = self._frame_fname(
                self.next_frame_number - 1,
                is_depth=True
            )
            name_dict.update({
                'source_rgb_fname': source_rgb_fname,
                'source_pcd_fname': source_pcd_fname,
                'source_depth_fname': source_depth_fname,
            })

        return name_dict

    def next(self):
        name_dict = self.current()
        self.next_frame_number += 1
        return name_dict

    def target_img(self):
        img_filename = self.current()['target_rgb_fname']
        img = cv2.imread(img_filename, cv2.IMREAD_COLOR)
        return img

    def source_img(self):
        img_filename = self.current()['source_rgb_fname']
        img = cv2.imread(img_filename, cv2.IMREAD_COLOR)
        return img

    def image_size(self):
        img = self.target_img()
        return len(img), len(img[0]) # filas, columnas

    def current_frame_number(self):
        """
        Gives numbers from 0 to #frames-1
        """
        if self.next_frame_number > 1:
            return self.next_frame_number - 1 - 1
        else:
            return self.next_frame_number - 1