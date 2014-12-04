#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import sys
import os
import re

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
        self.obj_scene_nums = self._get_obj_scene_numbers()

        # get the number of frames available
        filenames = os.listdir(scene_path)
        filenames = [fname for fname in filenames if fname.endswith('.pcd')]
        last_frame_number = 1

        for filename in filenames:
            frame_number = self._get_fnumber(filename)
            if frame_number > last_frame_number:
                last_frame_number = frame_number

        # Set initial and last frame number
        self.offset_frame_count = 1
        self.next_frame_number = self.offset_frame_count
        self.last_frame_number = last_frame_number

    def _get_obj_scene_numbers(self):
        reg_text = '{o}_{n}_(?P<scene>\d+)_\d+.*'.format(
            o=self.obj,
            n=self.obj_number,
        )
        reg_exp = re.compile(reg_text)
        filenames = os.listdir(self.obj_path)
        matchings = [reg_exp.match(fname) for fname in filenames]
        scene_nums = set(
            [m.groupdict()['scene'] for m in matchings if m is not None]
        )
        return list(scene_nums)

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

    @staticmethod
    def imread(fname, color_scheme):
        if not os.path.isfile(fname):
            raise Exception("La imagen que quiere cargar no existe")
        return cv2.imread(fname, color_scheme)

    def obj_rgb_templates_and_masks(self, num_diff_images=3,
                                    sizes=None, start_in_frame=1):

        if sizes is None:
            sizes = [0.25, 0.5, 2]

        if 1 in sizes:
            sizes.remove(1)

        # Lista con los numeros de las escenas
        scenes_list = self.obj_scene_nums[:]
        # Lo que tengo que estirar la lista para tener "diff_imgs" distintas
        # rotando las distintas escenas
        mult = (num_diff_images / len(self.obj_scene_nums)) + 1
        scenes_rotation = (scenes_list * mult)[:num_diff_images]
        scene_sample = self.rgb_img()
        scene_height, scene_width = scene_sample.shape[0], scene_sample.shape[1]

        # Lista de nros de frames que se van a usar potencialmente
        frames_list = range(start_in_frame, num_diff_images + start_in_frame)
        frames_list *= len(self.obj_scene_nums)
        # Me quedo solo con aquellos que si se van a usar
        frames_rotation = sorted(frames_list)[:num_diff_images]

        # Listas resultantes
        templates = []
        masks = []

        for scene_num, frame_num in zip(scenes_rotation, frames_rotation):
            template_fname = self._obj_fname(
                obj_scene_number=scene_num,
                frame_number=frame_num,
                suffix='_crop.png',
            )
            mask_fname = self._obj_fname(
                obj_scene_number=scene_num,
                frame_number=frame_num,
                suffix='_maskcrop.png',
            )
            template = self.imread(template_fname, cv2.IMREAD_COLOR)
            mask = self.imread(mask_fname, cv2.IMREAD_GRAYSCALE)

            templates.append(template)
            masks.append(mask)

            for size in sizes:
                resized_template = cv2.resize(
                    template,
                    (0, 0),
                    fx=size,
                    fy=size,
                )
                if (resized_template.shape[0] < scene_height and
                        resized_template.shape[1] < scene_width):
                    resized_mask = cv2.resize(
                        mask,
                        (0, 0),
                        fx=size,
                        fy=size,
                    )
                    templates.append(resized_template)
                    masks.append(resized_mask)

        return templates, masks

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
        return self.imread(fname, cv2.IMREAD_COLOR)

    def rgbdepth_img(self):
        fname = self.depth_fname()
        depth_img = self.imread(fname, cv2.IMREAD_ANYDEPTH)

        height = len(depth_img)
        width = len(depth_img[0])
        rgbdepth_img = np.zeros((height, width, 3), np.uint8)
        for r in range(height):
            for c in range(width):
                char_rgb = depth_to_rgb(int(depth_img[r][c]))
                rgbdepth_img[r][c] = [char_rgb.blue, char_rgb.green, char_rgb.red]

        return rgbdepth_img

    def depth_img(self):
        fname = self.depth_fname()
        return self.imread(fname, cv2.IMREAD_ANYDEPTH)

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


class FrameNamesAndImageProviderPreChargedForPCD(FrameNamesAndImageProvider):
    def __init__(self, scene_path, scene, scene_number, obj_path, obj, obj_number):
        (super(FrameNamesAndImageProviderPreChargedForPCD, self)
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

    def restart(self):
        self.next_frame_number = self.offset_frame_count

    def pcd(self):
        return self._pcd_images[self.next_frame_number - self.offset_frame_count]


class FrameNamesAndImageProviderPreChargedForRGB(FrameNamesAndImageProvider):
    def __init__(self, scene_path, scene, scene_number, obj_path, obj, obj_number):
        (super(FrameNamesAndImageProviderPreChargedForRGB, self)
         .__init__(scene_path, scene, scene_number, obj_path, obj, obj_number))

        self._images = []

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
            fname = self._frame_fname(i, is_rgb=True)
            pc = self.imread(str(fname), cv2.IMREAD_COLOR)
            self._images.append(pc)
        sys.stdout.write('\n')

    def restart(self):
        self.next_frame_number = self.offset_frame_count

    def rgb_img(self):
        return self._images[self.next_frame_number - self.offset_frame_count]


class TemplateAndImageProviderFromVideo(object):

    def __init__(self, video_path, template_path):
        self.video_capture = cv2.VideoCapture(video_path)
        self.template_path = template_path

        self.next_frame_number = 1
        self._have_imgs, self._img = self.video_capture.read()

    def next(self):
        self.next_frame_number += 1
        self._have_imgs, self._img = self.video_capture.read()

    def have_images(self):
        return self._have_imgs

    def restart(self):
        self.next_frame_number = 1
        self.video_capture.release()
        self.video_capture.open()

    def rgb_img(self):
        return self._img

    def obj_rgb_templates_and_masks(self):
        return self.imread(self.template_path, cv2.IMREAD_COLOR)

    def image_list(self):
        return [self.rgb_img()]