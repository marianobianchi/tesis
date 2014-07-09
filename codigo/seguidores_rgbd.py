#!/usr/bin/python
#coding=utf-8

from __future__ import (unicode_literals, division)


import cv2
import scipy.io


from esquemas_seguimiento import NameBasedFollowingScheme
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
            'nframe': self.img_provider.current_frame_number(),
        })
        return desc

    #######################
    # Funcion de deteccion
    #######################
    def detect(self, name_dict):
        # Actualizo descriptores e imagen en detector
        self.detector.update(name_dict, self.descriptors())

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
    def follow(self, name_dict):
        # Actualizo descriptores e imagen en comparador
        self.finder.update(name_dict, self.descriptors())

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

    def update(self, name_dict, descriptors):
        self._descriptors.update(descriptors)
        self._descriptors.update(name_dict)

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

    #def simple_follow(self,
    #                  name_dict,
    #                  ubicacion,
    #                  valor_comparativo,
    #                  tam_region_inicial):
    #    """
    #    Esta funcion es el esquema de seguimiento del objeto.
    #    """
    #    filas, columnas = self.img_provider.image_size()
    #
    #    nueva_ubicacion = ubicacion
    #    nueva_comparacion = None
    #    tam_region_final = tam_region_inicial
    #
    #    # Seguimiento (busqueda/deteccion acotada)
    #    for fila, columna, tam_region in (self.metodo_de_busqueda
    #                                      .get_positions_and_framesizes(
    #                                        ubicacion,
    #                                        tam_region_inicial,
    #                                        filas,
    #                                        columnas)):
    #
    #        # Si se quiere ver como va buscando, descomentar la siguiente linea
    #        #MuestraBusquedaEnVivo('Buscando el objeto').run(
    #        #    img_copy,
    #        #    (x, y),
    #        #    tam_region,
    #        #    None,
    #        #    frenar=True,
    #        #)
    #
    #        nueva_comparacion = self.compare.comparisson(fila, columna, tam_region)
    #
    #        # Si hubo coincidencia
    #        if self.compare.is_best_match(nueva_comparacion, valor_comparativo):
    #            # Nueva ubicacion del objeto (esq. superior izq. del cuadrado)
    #            nueva_ubicacion = (fila, columna)
    #
    #            # Actualizo el valor de la comparacion
    #            valor_comparativo = nueva_comparacion
    #
    #            # Actualizo el tamaño de la region
    #            tam_region_final = tam_region
    #
    #    return nueva_ubicacion, valor_comparativo, tam_region_final


class Detector(object):
    """
    Es la clase que se encarga de detectar el objeto buscado
    """
    def __init__(self):
        self._descriptors = {}

    def update(self, name_dict, desc):
        self._descriptors.update(name_dict)
        self._descriptors.update(desc)

    def calculate_descriptors(self, ubicacion, tam_region):
        """
        Calcula los descriptores en base al objeto encontrado para que
        los almacene el Follower
        """
        return {}

    def detect(self):
        pass



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


class ICPFinder(Finder):

    def find(self):
        size = self._descriptors['size']
        im_c_left = self._descriptors['location'][1]
        im_c_right = im_c_left + size
        im_r_top = self._descriptors['location'][0]
        im_r_bottom = im_r_top + size

        topleft = IntPair(im_r_top, im_c_left)
        bottomright = IntPair(im_r_bottom, im_c_right)

        depth_fname = str(self._descriptors['source_depth_fname'])
        source_cloud_fname = str(self._descriptors['source_pcd_fname'])
        target_cloud_fname = str(self._descriptors['target_pcd_fname'])

        icp_result = follow(
            topleft,
            bottomright,
            depth_fname,
            source_cloud_fname,
            target_cloud_fname
        )

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
    img_provider = FrameNamesAndImageProvider(
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