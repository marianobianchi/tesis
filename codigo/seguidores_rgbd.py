#!/usr/bin/python
# -*- coding: utf-8 -*-
#Con esto, todos los strings literales son unicode (no hace falta poner u'algo')
from __future__ import unicode_literals


import numpy as np
import cv2
import scipy.io


from seguimiento_common.esquemas_seguimiento import (FollowingSchema,
                                                     FollowingSchemaCountingFrames)
from seguimiento_common.observar_seguimiento import (MuestraSeguimientoEnVivo, MuestraBusquedaEnVivo,
                                  GrabaSeguimientoEnArchivo)
from seguimiento_common.proveedores_de_imagenes import (FramesAsVideo,
                                                        GrayFramesAsVideo,
                                                        RGBDDatabaseFramesAsVideo)
from seguimiento_common.metodos_de_busqueda import *

from metodos_comunes import *



#####################
# Objetos seguidores
#####################
class Follower(object):
    """
    Es la clase base para los seguidores.

    Almacena los descriptores que son utilizados y actualizados por el detector
    de objetos y por el seguidor.

    Es conocido por el esquema de seguimiento que se encarga de utilizarlo como
    corresponde para lograr el objetivo final.
    """

    def __init__(self, image_provider, detector, compare, metodo_de_busqueda=BusquedaEnEspiral()):
        self.img_provider = image_provider
        self.metodo_de_busqueda = metodo_de_busqueda

        # Following helpers
        self.detector = detector
        self.compare = compare

        # Object descriptors
        self._obj_location = (0, 0) # (Fila, columna)
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

    #def object_roi(self):
    #    return self._obj_descriptors['frame']
    #
    #def object_mask(self):
    #    return self._obj_descriptors['mask']

    def object_frame_size(self):
        """
        Devuelve el tamaño de un lado del cuadrado que contiene al objeto
        """
        return self._obj_frame_size

    def object_location(self):
        return self._obj_location


    #####################################
    # Esquema de seguimiento del objeto
    #####################################
    def simple_follow(self, img, ubicacion, valor_comparativo, tam_region_inicial):
        """
        Esta funcion es el esquema de seguimiento del objeto.
        """
        filas, columnas = len(img), len(img[0])

        nueva_ubicacion = ubicacion
        nueva_comparacion = None
        tam_region_final = tam_region_inicial

        # Seguimiento (busqueda/deteccion acotada)
        for x, y, tam_region in self.metodo_de_busqueda.get_positions_and_framesizes(ubicacion,
                                                                                   tam_region_inicial,
                                                                                   filas,
                                                                                   columnas):
            col_izq = y
            col_der = col_izq + tam_region
            fil_arr = x
            fil_aba = fil_arr + tam_region

            # Tomo una region de la imagen donde se busca el objeto
            roi = img[fil_arr:fil_aba,col_izq:col_der]

            # Si se quiere ver como va buscando, descomentar la siguiente linea
            #MuestraBusquedaEnVivo('Buscando el objeto').run(
            #    img_copy,
            #    (x, y),
            #    tam_region,
            #    None,
            #    frenar=True,
            #)

            nueva_comparacion = self.compare.comparisson(roi)

            # Si hubo coincidencia
            if self.compare.is_best_match(nueva_comparacion, valor_comparativo):
                # Nueva ubicacion del objeto (esquina superior izquierda del cuadrado)
                nueva_ubicacion = (x, y)

                # Actualizo el valor de la comparacion
                valor_comparativo = nueva_comparacion

                # Actualizo el tamaño de la region
                tam_region_final = tam_region

        return nueva_ubicacion, valor_comparativo, tam_region_final

    def follow(self, img):
        """
        Esta funcion utiliza al esquema de seguimiento del objeto (simple_follow)
        """
        # Descomentar si se quiere ver la busqueda
        #img_copy = img.copy()

        vieja_ubicacion = self.object_location()
        nueva_ubicacion = vieja_ubicacion

        tam_region = self.object_frame_size()
        tam_region_final = tam_region

        # Actualizo descriptores e imagen en comparador
        self.compare.update(img, self.descriptors())

        # Valor comparativo base
        valor_comparativo = self.compare.base_comparisson()

        # Repito 3 veces (cantidad arbitraria) una busqueda, partiendo siempre
        # de la ultima mejor ubicacion del objeto encontrada
        for i in range(3):
            nueva_ubicacion, valor_comparativo, tam_region_final = self.simple_follow(
                img,
                nueva_ubicacion,
                valor_comparativo,
                tam_region_final
            )

        fue_exitoso = (vieja_ubicacion != nueva_ubicacion)
        nueva_ubicacion = nueva_ubicacion if fue_exitoso else None

        if fue_exitoso:
            # Calculo y actualizo los descriptores con los valores encontrados
            self.upgrade_followed_descriptors(img, nueva_ubicacion, tam_region_final)

        # Devuelvo self.object_frame_size() porque puede cambiar en "upgrade_descriptors"
        # Idem con self.object_location()
        return fue_exitoso, self.object_frame_size(), self.object_location()

    ##########################
    # Actualizar descriptores
    ##########################
    def set_object_descriptors(self, ubicacion, tam_region, obj_descriptors):
        self._obj_location = ubicacion
        self._obj_frame_size = tam_region
        self._obj_descriptors.update(obj_descriptors)

    def upgrade_detected_descriptors(self, img, ubicacion, tam_region):
        desc = self.detector.calculate_descriptors(img, ubicacion, tam_region)
        self.set_object_descriptors(ubicacion, tam_region, desc)

    def upgrade_followed_descriptors(self, img, ubicacion, tam_region):
        desc = self.compare.calculate_descriptors(img, ubicacion, tam_region)
        self.set_object_descriptors(ubicacion, tam_region, desc)

    #######################
    # Funcion de deteccion
    #######################
    def detect(self, img):
        # Actualizo descriptores e imagen en detector
        self.detector.update(img, self.descriptors())

        # Detectar
        fue_exitoso, tam_region, location = self.detector.detect()

        if fue_exitoso:
            # Calculo y actualizo los descriptores con los valores encontrados
            self.upgrade_detected_descriptors(img, location, tam_region)
            tam_region = self.object_frame_size()
            location = self.object_location()

        return fue_exitoso, tam_region, location


class FollowerWithStaticDetection(Follower):
    #######################
    # Funcion de deteccion
    #######################


    def detect(self, img, nframe):
        """
        nframe es el numero de frame del video en el que se está haciendo la
        deteccion
        """
        # Actualizo descriptores e imagen en detector
        descriptors = self.descriptors()
        descriptors.update({'nframe': nframe})
        self.detector.update(img, descriptors)

        # Detectar
        fue_exitoso, tam_region, location = self.detector.detect()

        if fue_exitoso:
            # Calculo y actualizo los descriptores con los valores encontrados
            self.upgrade_detected_descriptors(img, location, tam_region)
            tam_region = self.object_frame_size()
            location = self.object_location()

        return fue_exitoso, tam_region, location



################################
# Clases para detectar y seguir
################################
class Compare(object):
    """
    Es la clase que se encarga de hacer las comparaciones al momento de
    realizar el seguimiento para poder decidir a donde se movio el objeto
    """
    def __init__(self):
        self._descriptors = {}
        self._img = None

    def update(self, img, descriptors):
        self._img = img
        self._descriptors.update(descriptors)

    def calculate_descriptors(self, img, ubicacion, tam_region):
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


class Detector(object):
    """
    Es la clase que se encarga de detectar el objeto buscado
    """
    def __init__(self):
        self._descriptors = {}
        self._img = None

    def update(self, img, descriptors):
        self._img = img
        self._descriptors.update(descriptors)

    def calculate_descriptors(self, img, ubicacion, tam_region):
        """
        Calcula los descriptores en base al objeto encontrado para que
        los almacene el Follower
        """
        return {}

    def detect(self):
        pass


class SimpleCircleCompare(Compare):
    def calculate_descriptors(self, img, ubicacion, tam_region):
        """
        Calcula los descriptores en base al objeto encontrado para que
        los almacene el Follower
        """
        desc = {}

        frame = img[ubicacion[0]:ubicacion[0]+tam_region,
                    ubicacion[1]:ubicacion[1]+tam_region]

        desc['frame'] = frame
        desc['mask'] = self.calculate_mask(frame)
        return desc

    def base_comparisson(self):
        """
        Comparacion base: sirve como umbral para las comparaciones que se
        realizan durante el seguimiento
        """
        filas, columnas = len(self._img), len(self._img[0])
        return filas * columnas

    def comparisson(self, roi):
        # Hago una comparacion bit a bit de la imagen original
        # Compara solo en la zona de la máscara y deja 0's en donde hay
        # coincidencias y 255's en donde no coinciden
        past_obj_roi = self._descriptors['frame']

        mask = self._descriptors['mask']

        xor = cv2.bitwise_xor(past_obj_roi, roi, mask=mask)

        # Cuento la cantidad de 0's y me quedo con la mejor comparacion
        return cv2.countNonZero(xor)

    def is_best_match(self, new_value, old_value):
        return new_value < old_value

    def calculate_mask(self, img):
        # Da vuelta los valores (0->255 y 255->0)
        mask = cv2.bitwise_not(img)
        return mask


class SimpleCircleDetector(Detector):
    def calculate_descriptors(self, img, ubicacion, tam_region):
        """
        Calcula los descriptores en base al objeto encontrado para que
        los almacene el Follower
        """
        desc = {}

        frame = img[ubicacion[0]:ubicacion[0]+tam_region,
                    ubicacion[1]:ubicacion[1]+tam_region]

        desc['frame'] = frame
        desc['mask'] = self.calculate_mask(frame)
        return desc

    def detect(self):
        fue_exitoso = True
        tam_region = 80
        ubicacion = (40,40)
        return fue_exitoso, tam_region, ubicacion

    def calculate_mask(self, img):
        # Da vuelta los valores (0->255 y 255->0)
        mask = cv2.bitwise_not(img)
        return mask



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
        location = (0,0)

        for obj in objs:
            if obj[0][0] == self._obj_rgbd_name:
                fue_exitoso = True
                location = (obj[2][0][0], obj[4][0][0])
                print "Frame {n}: (top, left)=({t},{l})".format(
                    n=nframe,
                    t=location[0],
                    l=location[1],
                )
                tam_region = max(obj[3][0][0]-obj[2][0][0], obj[5][0][0]-obj[4][0][0])
                break

        #TODO: ver que conviene devolver
        return fue_exitoso, tam_region, location


class DumbDepthCompare(Compare):
    pass


def seguir_pelota_negra():
    img_provider = GrayFramesAsVideo('videos/moving_circle')
    detector = SimpleCircleDetector()
    compare = SimpleCircleCompare()
    follower = Follower(img_provider, detector, compare)

    muestra_seguimiento = MuestraSeguimientoEnVivo('Seguimiento')
    FollowingSchema(img_provider, follower, muestra_seguimiento).run()


if __name__ == '__main__':
    img_provider = RGBDDatabaseFramesAsVideo('videos/rgbd/scenes/','desk','1') # path, objname, number
    detector = StaticDetector('videos/rgbd/scenes/desk/desk_1.mat', 'coffee_mug')
    compare = DumbDepthCompare()
    follower = FollowerWithStaticDetection(img_provider, detector, compare)

    muestra_seguimiento = MuestraSeguimientoEnVivo('Seguimiento')
    FollowingSchemaCountingFrames(img_provider, follower, muestra_seguimiento).run()