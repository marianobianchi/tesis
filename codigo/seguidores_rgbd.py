#!/usr/bin/python
#coding=utf-8

from __future__ import (unicode_literals, division)


import cv2
import scipy.io


from esquemas_seguimiento import NameBasedFollowingScheme
from metodos_comunes import (from_flat_to_cloud, filter_cloud)
from observar_seguimiento import MuestraSeguimientoEnVivo
from proveedores_de_imagenes import FrameNamesAndImageProvider
from cpp.icp_follow import *


#####################
# Objetos seguidores
#####################
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

    #######################
    # Funcion de deteccion
    #######################
    def detect(self):
        # Actualizo descriptores e imagen en detector
        self.detector.update(self.descriptors())

        # Detectar
        fue_exitoso, tam_region, location = self.detector.detect()

        if fue_exitoso:
            # Calculo y actualizo los descriptores con los valores encontrados
            self.upgrade_detected_descriptors(location, tam_region)
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
        fue_exitoso, tam_region, location = self.finder.find()

        if fue_exitoso:
            # Calculo y actualizo los descriptores con los valores encontrados
            self.upgrade_followed_descriptors(location, tam_region)
            tam_region = self.descriptors()['size']
            location = self.descriptors()['location']

        return fue_exitoso, tam_region, location


    ##########################
    # Actualizar descriptores
    ##########################
    def set_object_descriptors(self, ubicacion, tam_region, obj_descriptors):
        self._obj_location = ubicacion
        self._obj_frame_size = tam_region
        self._obj_descriptors.update(obj_descriptors)

    def upgrade_detected_descriptors(self, ubicacion, tam_region):
        desc = self.detector.calculate_descriptors(ubicacion, tam_region)
        self.set_object_descriptors(ubicacion, tam_region, desc)

    def upgrade_followed_descriptors(self, ubicacion, tam_region):
        desc = self.finder.calculate_descriptors(ubicacion, tam_region)
        self.set_object_descriptors(ubicacion, tam_region, desc)


################################
# Clases para detectar y buscar
################################
class Finder(object):
    """
    Es la clase que se encarga de buscar el objeto
    """
    def __init__(self):
        self._descriptors = {}

    def update(self, descriptors):
        self._descriptors.update(descriptors)

    def calculate_descriptors(self, ubicacion, tam_region):
        """
        Calcula los descriptores en base al objeto encontrado para que
        los almacene el Follower
        """
        return {}

    def base_comparisson(self):
        """
        Comparacion base: sirve como umbral para las comparaciones que se
        realizan durante el seguimiento
        """
        return 0

    def comparisson(self, roi):
        return 0

    def is_best_match(self, new_value, old_value):
        return False

    #####################################
    # Esquema de seguimiento del objeto
    #####################################
    def find(self):
        return (False, 0, (0, 0))


class Detector(object):
    """
    Es la clase que se encarga de detectar el objeto buscado
    """
    def __init__(self):
        self._descriptors = {}

    def update(self, desc):
        self._descriptors.update(desc)

    def calculate_descriptors(self, ubicacion, tam_region):
        """
        Calcula los descriptores en base al objeto encontrado para que
        los almacene el Follower
        """
        return {}

    def detect(self):
        pass



class FollowerWithStaticDetection(Follower):
    def descriptors(self):
        desc = super(StaticFollower, self).descriptors()
        desc.update({
            'nframe': self.img_provider.current_frame_number(),
            'depth_img': self.img_provider.depth_img(),
            'pcd': self.img_provider.pcd(),
        })
        return desc


class StaticDetector(Detector):
    """
    Esta clase se encarga de definir la ubicación del objeto buscado en la
    imagen valiendose de los datos provistos por la base de datos RGBD.
    Los datos se encuentran almacenados en un archivo ".mat".
    """
    def __init__(self, matfile_path, obj_rgbd_name):
        super(StaticDetector, self).__init__()
        self._matfile = scipy.io.loadmat(matfile_path)['bboxes']
        self._obj_rgbd_name = obj_rgbd_name

    def detect(self):
        nframe = self._descriptors['nframe']

        objs = self._matfile[0][nframe][0]

        fue_exitoso = False
        tam_region = 0
        location = (0, 0)

        for obj in objs:
            if obj[0][0] == self._obj_rgbd_name:
                fue_exitoso = True
                location = (int(obj[2][0][0]), int(obj[4][0][0]))
                tam_region = max(int(obj[3][0][0]) - int(obj[2][0][0]),
                                 int(obj[5][0][0]) - int(obj[4][0][0]))
                break

        return fue_exitoso, tam_region, location #location=(fila, columna)

    def calculate_descriptors(self, ubicacion, tam_region):
        """
        Obtengo la nube de puntos correspondiente a la ubicacion y region
        pasadas por parametro.
        """
        depth_img = self._descriptors['depth_img']

        rows_cols_limits = from_flat_to_cloud_limits(
            ubicacion,
            (ubicacion[0] + tam_region, ubicacion[1] + tam_region),
            depth_img,
        );

        r_top_limit = rows_cols_limits.first.first;
        r_bottom_limit = rows_cols_limits.first.second;
        c_left_limit = rows_cols_limits.second.first;
        c_right_limit = rows_cols_limits.second.second;

        cloud = self._descriptors['pcd']

        cloud = filter_cloud(cloud, "y", r_top_limit, r_bottom_limit)
        cloud = filter_cloud(cloud, "x", c_left_limit, c_right_limit)

        return {'object_cloud': cloud}


class ICPFinder(Finder):

    def find(self):
        #TODO: filter target_cloud (por ejemplo, una zona 4 veces mayor)
        #size = self._descriptors['size']
        #im_c_left = self._descriptors['location'][1]
        #im_c_right = im_c_left + size
        #im_r_top = self._descriptors['location'][0]
        #im_r_bottom = im_r_top + size
        #
        #topleft = IntPair(im_r_top, im_c_left)
        #bottomright = IntPair(im_r_bottom, im_c_right)

        # CODIGO DE C++
        #// Define row and column limits for the zone to search the object
        #// In this case, we look on a box N times the size of the original
        #int N = 4;
        #r_top_limit = r_top_limit - ( (r_bottom_limit - r_top_limit) * N);
        #r_bottom_limit = r_bottom_limit + ( (r_bottom_limit - r_top_limit) * N);
        #c_left_limit = c_left_limit - ( (c_right_limit - c_left_limit) * N);
        #c_right_limit = c_right_limit + ( (c_right_limit - c_left_limit) * N);
        #
        #// Filter points corresponding to the zone where the object being followed is supposed to be
        #filter_cloud(         target_cloud, filtered_target_cloud, "y", r_top_limit, r_bottom_limit);
        #filter_cloud(filtered_target_cloud, filtered_target_cloud, "x", c_left_limit, c_right_limit);



        object_cloud = self._descriptors['object_cloud']
        target_cloud = self._descriptors['pcd']

        icp_result = follow(object_cloud, target_cloud)

        fue_exitoso = icp_result.has_converged
        tam_region = icp_result.size
        location = (icp_result.top, icp_result.left)

        return fue_exitoso, tam_region, location


def prueba_de_deteccion_estatica():
    img_provider = FrameNamesAndImageProvider(
        'videos/rgbd/scenes/', 'desk', '1'
    )  # path, objname, number

    detector = StaticDetector(
        'videos/rgbd/scenes/desk/desk_1.mat',
        'coffee_mug'
    )

    finder = Finder()

    follower = Follower(img_provider, detector, finder)

    show_following = MuestraSeguimientoEnVivo('Deteccion estatica - Sin seguidor')

    NameBasedFollowingScheme(
        img_provider,
        follower,
        show_following,
    ).run()


def prueba_seguimiento_ICP():
    img_provider = DepthAndRGBImageProvider(
        'videos/rgbd/scenes/', 'desk', '1'
    )  # path, objname, number

    detector = StaticDetector(
        'videos/rgbd/scenes/desk/desk_1.mat',
        'coffee_mug'
    )

    finder = ICPFinder()

    follower = Follower(img_provider, detector, finder)

    show_following = MuestraSeguimientoEnVivo('Seguidor ICP')

    NameBasedFollowingScheme(
        img_provider,
        follower,
        show_following,
    ).run()

if __name__ == '__main__':
    prueba_seguimiento_ICP()