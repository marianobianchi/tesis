# coding=utf-8

from __future__ import (unicode_literals, division)

import cv2

from cpp.common import points

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
        self._obj_topleft = (0, 0)  # (Fila, columna)
        self._obj_bottomright = (0, 0)
        self._obj_descriptors = {}

    ########################
    # Descriptores comunes
    ########################
    def descriptors(self):
        desc = self._obj_descriptors.copy()
        desc.update({
            'topleft': self._obj_topleft,
            'bottomright': self._obj_bottomright,
            'nframe': self.img_provider.next_frame_number,
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
        topleft = (0, 0)
        bottomright = (0, 0)

        if fue_exitoso:
            # Calculo y actualizo los descriptores con los valores encontrados
            self.upgrade_detected_descriptors(descriptors)
            topleft = self.descriptors()['topleft']
            bottomright = self.descriptors()['bottomright']

        return fue_exitoso, topleft, bottomright

    ######################
    # Funcion de busqueda
    ######################
    def follow(self, es_deteccion):
        """
        es_deteccion indica si en el frame anterior se realizo una deteccion
        """
        # Actualizo descriptores e imagen en comparador
        self.finder.update(self.descriptors())

        # Busco el objeto
        fue_exitoso, descriptors = self.finder.find(es_deteccion)
        topleft = (0, 0)
        bottomright = (0, 0)

        if fue_exitoso:
            # Calculo y actualizo los descriptores con los valores encontrados
            self.upgrade_followed_descriptors(descriptors)
            topleft = self.descriptors()['topleft']
            bottomright = self.descriptors()['bottomright']

        return fue_exitoso, topleft, bottomright

    ##########################
    # Actualizar descriptores
    ##########################
    def set_object_descriptors(self, obj_descriptors):
        self._obj_topleft = obj_descriptors.pop('topleft')
        self._obj_bottomright = obj_descriptors.pop('bottomright')
        self._obj_descriptors.update(obj_descriptors)

    def upgrade_detected_descriptors(self, descriptors):
        desc = self.detector.calculate_descriptors(descriptors)
        self.set_object_descriptors(desc)

    def upgrade_followed_descriptors(self, descriptors):
        desc = self.finder.calculate_descriptors(descriptors)
        self.set_object_descriptors(desc)


class DepthFollower(Follower):
    def train(self):
        obj_model = self.img_provider.obj_pcd()
        pts = points(obj_model)
        self._obj_descriptors.update(
            {
                'obj_model': obj_model,
                'obj_model_points': pts,
            }
        )


    def descriptors(self):
        desc = super(DepthFollower, self).descriptors()
        desc.update({
            'depth_img': self.img_provider.depth_img(),
            'pcd': self.img_provider.pcd(),
        })
        return desc


#################
# Seguidores RGB
#################
class RGBFollower(Follower):
    def train(self):
        obj_templates, obj_masks = self.img_provider.obj_rgb_templates_and_masks()
        self._obj_descriptors.update(
            {
                'object_templates': obj_templates,
                'object_masks': obj_masks,
            }
        )

    def descriptors(self):
        desc = super(RGBFollower, self).descriptors()
        desc.update({
            'scene_rgb': self.img_provider.rgb_img(),
        })
        return desc


###################
# Seguidores RGB-D
###################

class RGBDPreferDFollowerWithStaticDetection(Follower):
    """
    Combina los seguidores RGB y D usando la deteccion estatica de profundidad,
    ya que inserta en los descriptores a las nubes de puntos
    """

    def __init__(self, image_provider, depth_static_detector,
                 rgb_finder, depth_finder):

        self.img_provider = image_provider

        # Following helpers
        self.depth_static_detector = depth_static_detector
        self.rgb_finder = rgb_finder
        self.depth_finder = depth_finder

        # Object descriptors
        self._obj_topleft = (0, 0)  # (Fila, columna)
        self._obj_bottomright = (0, 0)
        self._obj_descriptors = {}

    ########################
    # Descriptores comunes
    ########################
    def descriptors(self):
        desc = super(RGBDPreferDFollowerWithStaticDetection, self).descriptors()
        desc.update({
            'scene_rgb': self.img_provider.rgb_img(),
            'depth_img': self.img_provider.depth_img(),
            'pcd': self.img_provider.pcd(),
        })
        return desc

    ###########################
    # Funcion de entrenamiento
    ###########################
    def train(self):
        # Depth
        obj_model = self.img_provider.obj_pcd()
        pts = points(obj_model)
        self._obj_descriptors.update(
            {
                'obj_model': obj_model,
                'obj_model_points': pts,
            }
        )

        # RGB
        obj_templates, obj_masks = self.img_provider.obj_rgb_templates_and_masks()
        self._obj_descriptors.update(
            {
                'object_templates': obj_templates,
                'object_masks': obj_masks,
            }
        )

    #######################
    # Funcion de deteccion
    #######################
    def detect(self):
        # Actualizo descriptores e imagen en detector
        self.depth_static_detector.update(self.descriptors())

        # Detectar
        fue_exitoso, descriptors = self.depth_static_detector.detect()

        topleft = (0, 0)
        bottomright = (0, 0)

        if fue_exitoso:
            # Calculo y actualizo los descriptores con los valores encontrados
            self.upgrade_detected_descriptors(descriptors)
            topleft = self.descriptors()['topleft']
            bottomright = self.descriptors()['bottomright']

        return fue_exitoso, topleft, bottomright

    ######################
    # Funcion de busqueda
    ######################
    def follow(self, es_deteccion):
        """
        es_deteccion indica si en el frame anterior se realizo una deteccion
        """
        # Actualizo descriptores e imagen en comparador
        self.depth_finder.update(self.descriptors())

        # Busco el objeto
        fue_exitoso, descriptors = self.depth_finder.find(es_deteccion)
        topleft = (0, 0)
        bottomright = (0, 0)

        if fue_exitoso:
            # Me quedo con el resultado del depth finder para la nube de puntos
            # y corro el detector RGB buscando solo con diferentes tamaños, pero
            # no moviendo el centro del cuadrante
            self.upgrade_depth_followed_descriptors(descriptors)
            self.rgb_finder.update(self.descriptors())
            mejora_fue_exitosa, new_descriptors = self.rgb_finder.find(es_deteccion=False)

            if mejora_fue_exitosa:
                # Calculo y actualizo los descriptores con los valores encontrados
                self.upgrade_rgb_followed_descriptors(new_descriptors)

            topleft = self.descriptors()['topleft']
            bottomright = self.descriptors()['bottomright']

        return fue_exitoso, topleft, bottomright

    ##########################
    # Actualizar descriptores
    ##########################
    def upgrade_detected_descriptors(self, descriptors):
        desc = self.depth_static_detector.calculate_descriptors(descriptors)
        self.set_object_descriptors(desc)

    def upgrade_depth_followed_descriptors(self, descriptors):
        desc = self.depth_finder.calculate_descriptors(descriptors)
        self.set_object_descriptors(desc)

    def upgrade_rgb_followed_descriptors(self, descriptors):
        desc = self.rgb_finder.calculate_descriptors(descriptors)
        self.set_object_descriptors(desc)