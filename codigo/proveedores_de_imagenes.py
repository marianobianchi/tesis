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


scene_obj_frames_to_use = {
    'desk_1': {
        'coffee_mug_5': [(1, 3), (33, 51), (77, 98)],  # 22/22 (presente/ausente)
        'soda_can_6': [(1, 2), (31, 58), (69, 98)],  # 30/30
        'cap_4': [(17, 98)],  # 41/41
    },
    'desk_2': {
        'bowl_3': [(1, 21), (52, 190)],  # 80/80
        'soda_can_4': [(1, 20), (55, 190)],  # 78/78
        'flashlight_1': [(1, 26), (127, 190)],  # 60/60
    },
    'desk_3': {
        'flashlight_5': [(1, 67), (126, 228)],  # 85/85
        'flashlight_3': [(1, 129), (194, 228)],  # 82/82
        'cereal_box_1': [(1, 113), (186, 228)],  # 78/78
        'bowl_4': [(1, 66), (129, 228)],  # 83/83
    },
    'table_1': {
        'coffee_mug_1': [(1, 31), (51, 87), (110, 125)],  # 53/53
        'coffee_mug_4': [(1, 24)] * 2 + [(25, 125)] + [(1, 14)],  # 96/96
        'bowl_2': [(1, 10)] * 10 + [(12, 125)] + [(8, 11)],  # 114/114
        'cap_1': [(1, 4)] * 27 + [(5, 125)] + [(1, 3)],  # 116/116
        'cap_4': [(1, 20)] * 4 + [(21, 125)] + [(1, 9)],  # 97/97
        'cereal_box_4': [(1, 6)] * 19 + [(7, 125)] + [(1, 5)], # 119, 119
        'soda_can_4': [(1, 89), (111, 112)],  # 52/52
        'flashlight_3': [(1, 43), (57, 125)],  # 53/53
    },
    'table_small_1': {
        'coffee_mug_1': [(1, 7), (109, 199)] + [(1, 6)] * 11,  # 82/82
        'cereal_box_1': [(1, 199)],  # 199/0
        'soda_can_3': [(1, 4)] * 15 + [(96, 199)],  # 84/84
        'bowl_4': [(1, 9)] * 7 + [(100, 199)],  # 86/86
    },
    'table_small_2': {
        'cap_4': [],
        'soda_can_1': [],
        'cereal_box_4': [],
    },
    'meeting_small_1': {
        'bowl_3': [],
        'bowl_2': [],
        'cap_3': [],
        'soda_can_5': [],
        'cap_1': [],
        'cereal_box_2': [],
        'cereal_box_1': [],
        'coffee_mug_5': [],
        'flashlight_5': [],
        'coffee_mug_6': [],
        'soda_can_3': [],
        'flashlight_2': [],
        'soda_can_1': [],
    },
    'kitchen_small_1': {
        'cap_1': [],
        'bowl_4': [],
        'cereal_box_2': [],
        'coffee_mug_5': [],
        'soda_can_6': [],
        'flashlight_5': [],
        'flashlight_2': [],
        'soda_can_1': [],
    },

}


class FrameNamesAndImageProvider(object):
    def __init__(self, scene_path, scene, scene_number, obj_path, obj, obj_number):
        scene_path = os.path.join(scene_path, scene)
        scene_path = os.path.join(scene_path, scene + '_' + scene_number)
        self.scene_path = scene_path
        self.scene = scene
        self.scene_number = scene_number

        self._initialize_object(obj_path, obj, obj_number)

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

    def _initialize_object(self, obj_path, obj, obj_number):
        self.base_objs_path = obj_path
        obj_path = os.path.join(obj_path, obj)
        obj_path = os.path.join(obj_path, obj + '_' + obj_number)
        self.obj_path = obj_path
        self.obj = obj
        self.obj_number = obj_number
        self.obj_scene_nums = self._get_obj_scene_numbers()

    def reinitialize_object(self, obj, obj_number):
        self._initialize_object(self.base_objs_path, obj, obj_number)

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

    def _obj_pcd_fname(self):
        generic_fname = '{obj}_{obj_number}_entero.pcd'
        print "Levantando el modelo entero"

        fname = generic_fname.format(
            obj=self.obj,
            obj_number=self.obj_number,
        )
        return os.path.join(self.obj_path, fname)

    def obj_pcd(self, n=1):
        fname = self._obj_pcd_fname()
        pc = read_pcd(str(fname))
        return pc

    @staticmethod
    def imread(fname, color_scheme):
        if not os.path.isfile(fname):
            raise Exception("La imagen que quiere cargar no existe")
        return cv2.imread(fname, color_scheme)

    def obj_rgb_templates_and_masks(self, sizes=None):
        path = os.path.join(self.obj_path, 'templates_and_masks')

        if sizes is None:
            sizes = [0.5, 0.75, 1.25, 1.5]

        # Saco si tiene escala 1 porque por defecto más adelante ese lo agrego
        # siempre
        if 1 in sizes:
            sizes.remove(1)

        # Nombres de templates
        templates_fnames = [fname for fname in os.listdir(path)
                            if 'mask' not in fname and not os.path.isdir(os.path.join(path, fname))]
        templates_fnames.sort()

        # Nombres de mascaras
        masks_fnames = [fname for fname in os.listdir(path)
                        if 'mask' in fname and not os.path.isdir(os.path.join(path, fname))]
        masks_fnames.sort()

        # Tamaño de la escena
        scene_sample = self.rgb_img()
        scene_height, scene_width = scene_sample.shape[0], scene_sample.shape[1]

        # Listas resultantes
        templates = []
        masks = []

        for tmp_fname, msk_fname in zip(templates_fnames, masks_fnames):
            template_fname = os.path.join(path, tmp_fname)
            mask_fname = os.path.join(path, msk_fname)

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
                if (0 < resized_template.shape[0] < scene_height and
                        0 < resized_template.shape[1] < scene_width):
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

    def nframe(self):
        return self.next_frame_number

    # Images filenames
    def rgb_fname(self):
        assert self.have_images(), 'El proveedor de imágenes no tiene más imágenes disponibles'
        return self._frame_fname(self.nframe(), is_rgb=True)

    def depth_fname(self):
        assert self.have_images(), 'El proveedor de imágenes no tiene más imágenes disponibles'
        return self._frame_fname(self.nframe(), is_depth=True)

    def pcd_fname(self):
        assert self.have_images(), 'El proveedor de imágenes no tiene más imágenes disponibles'
        return self._frame_fname(self.nframe())

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
        return [self.rgb_img()]  # , self.rgbdepth_img()]

    def next(self):
        self.next_frame_number += 1

    def image_size(self):
        img = self.rgb_img()
        return len(img), len(img[0])  # filas, columnas


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


class SelectedFrameNamesAndImageProviderPreChargedForPCD(FrameNamesAndImageProvider):
    def __init__(self, scene_path, scene, scene_number, obj_path, obj, obj_number):
        scene_path = os.path.join(scene_path, scene)
        scene_path = os.path.join(scene_path, scene + '_' + scene_number)
        self.scene_path = scene_path
        self.scene = scene
        self.scene_number = scene_number

        self._pcd_images_dict = {}
        self._initialize_object(obj_path, obj, obj_number)

    def _initialize_object(self, obj_path, obj, obj_number):
        (super(SelectedFrameNamesAndImageProviderPreChargedForPCD, self)
         ._initialize_object(obj_path, obj, obj_number))

        self._initialize_nframes(self.scene, self.scene_number, obj, obj_number)
        self.next_frame = 0
        self._load_frames()

    def _initialize_nframes(self, scene, scene_number, obj, obj_number):
        scenenamenum = '{s}_{n}'.format(s=scene, n=scene_number)
        objnamenum = '{o}_{n}'.format(o=obj, n=obj_number)
        tuple_frames = scene_obj_frames_to_use[scenenamenum][objnamenum]
        nframes = []
        for t in tuple_frames:
            l = range(t[0], t[1] + 1)
            nframes.extend(l)

        self.nframes = nframes
        return nframes

    def _load_frames(self):
        total_files = len(self.nframes)
        for i in range(total_files):
            sys.stdout.write(
                "Reading pcd file number " +
                str(i + 1) +
                "/" +
                str(total_files) +
                "\r"
            )
            sys.stdout.flush()
            nframe = self.nframes[i]
            if nframe not in self._pcd_images_dict:
                fname = self._frame_fname(nframe)
                pc = read_pcd(str(fname))
                self._pcd_images_dict[nframe] = pc

        sys.stdout.write('\n')

    def restart(self):
        self.next_frame = 0

    def next(self):
        self.next_frame += 1

    def have_images(self):
        return self.next_frame < len(self.nframes)

    def nframe(self):
        try:
            n = self.nframes[self.next_frame]
        except:
            pass

        return n

    def pcd(self):
        return self._pcd_images_dict[self.nframe()]


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
