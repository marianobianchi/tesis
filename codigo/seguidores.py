#coding=utf-8

from __future__ import (unicode_literals, division)


class Follower(object):
    """
    Es la clase base para los seguidores.

    Almacena los descriptores que son utilizados y actualizados por el detector
    de objetos y por el buscador.
    """

    def __init__(self, image_provider, detector, finder):

        self.img_provider = image_provider

        # Following helpers
        self.detector = detector
        self.finder = finder

        # Object descriptors
        self._obj_location = (0, 0)  # (Fila, columna)
        self._obj_frame_size = 0
        self._obj_descriptors = {}

    ########################
    # Descriptores comunes
    ########################
    def descriptors(self):
        desc = self._obj_descriptors.copy()
        desc.update({
            'location': self._obj_location,
            'size': self._obj_frame_size,
        })
        return desc

    ###########################
    # Funcion de entrenamiento
    ###########################
    def train(self):
        pass

    #######################
    # Funcion de deteccion
    #######################
    def detect(self):
        # Actualizo descriptores e imagen en detector
        self.detector.update(self.descriptors())

        # Detectar
        fue_exitoso, descriptors = self.detector.detect()
        tam_region = 0
        location = (0,0)

        if fue_exitoso:
            # Calculo y actualizo los descriptores con los valores encontrados
            self.upgrade_detected_descriptors(descriptors)
            tam_region = self.descriptors()['size']
            location = self.descriptors()['location']

        return fue_exitoso, tam_region, location

    ######################
    # Funcion de busqueda
    ######################
    def follow(self):
        # Actualizo descriptores e imagen en comparador
        self.finder.update(self.descriptors())

        # Busco el objeto
        fue_exitoso, descriptors = self.finder.find()
        tam_region = 0
        location = (0,0)

        if fue_exitoso:
            # Calculo y actualizo los descriptores con los valores encontrados
            self.upgrade_followed_descriptors(descriptors)
            tam_region = self.descriptors()['size']
            location = self.descriptors()['location']

        return fue_exitoso, tam_region, location


    ##########################
    # Actualizar descriptores
    ##########################
    def set_object_descriptors(self, obj_descriptors):
        self._obj_location = obj_descriptors.pop('location')
        self._obj_frame_size = obj_descriptors.pop('size')
        self._obj_descriptors.update(obj_descriptors)

    def upgrade_detected_descriptors(self, descriptors):
        desc = self.detector.calculate_descriptors(descriptors)
        self.set_object_descriptors(desc)

    def upgrade_followed_descriptors(self, descriptors):
        desc = self.finder.calculate_descriptors(descriptors)
        self.set_object_descriptors(desc)


class FollowerWithStaticDetection(Follower):
    def descriptors(self):
        desc = super(FollowerWithStaticDetection, self).descriptors()
        desc.update({
            'nframe': self.img_provider.current_frame_number(),
        })
        return desc


class FollowerWithStaticDetectionAndPCD(FollowerWithStaticDetection):
    def descriptors(self):
        desc = super(FollowerWithStaticDetectionAndPCD, self).descriptors()
        desc.update({
            'depth_img': self.img_provider.depth_img(),
            'pcd': self.img_provider.pcd(),
        })
        return desc


class FollowerStaticICPAndObjectModel(FollowerWithStaticDetectionAndPCD):
    def train(self):
        obj_model = self.img_provider.obj_pcd()
        self._obj_descriptors.update({'obj_model': obj_model})