# coding=utf-8

from __future__ import (unicode_literals, division)

import cv2


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
    def follow(self):
        # Actualizo descriptores e imagen en comparador
        self.finder.update(self.descriptors())

        # Busco el objeto
        fue_exitoso, descriptors = self.finder.find()
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


class FollowerWithStaticDetection(Follower):
    def descriptors(self):
        desc = super(FollowerWithStaticDetection, self).descriptors()
        desc.update({
            'nframe': self.img_provider.next_frame_number,
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


#################
# Seguidores RGB
#################
class FollowerStaticAndRGBTemplate(FollowerWithStaticDetection):
    def train(self):
        obj_templates, obj_masks = self.img_provider.obj_rgb_templates_and_masks(
            num_diff_images=self.detector.templates_to_use,
            sizes=self.detector.template_sizes,
            start_in_frame=self.detector.templates_from_frame,
        )
        self._obj_descriptors.update(
            {
                'object_templates': obj_templates,
                'object_masks': obj_masks,
            }
        )

    def descriptors(self):
        desc = super(FollowerStaticAndRGBTemplate, self).descriptors()
        desc.update({
            'scene_rgb': self.img_provider.rgb_img(),
        })
        return desc


class FollowerStaticDetectionAndRGBTemplate(FollowerStaticAndRGBTemplate):
    def train(self):
        obj_templates, obj_masks = self.img_provider.obj_rgb_templates_and_masks()
        self._obj_descriptors.update(
            {
                'object_templates': obj_templates,
                'object_masks': obj_masks,
            }
        )












##########################
# BORRAR TODO LO DE ABAJO
##########################
from metodos_de_busqueda import BusquedaAlrededor

class ObjectDetectorAndFollower(object):
    def __init__(self, image_provider, metodo_de_busqueda=BusquedaAlrededor()):
        self.img_provider = image_provider
        self.metodo_de_busqueda = metodo_de_busqueda

        # Object descriptors
        self._obj_location = (40, 40)
        self._obj_frame_size = 80
        self._obj_descriptors = {}

    ########################
    # Descriptores comunes
    ########################
    def object_roi(self):
        return self._obj_descriptors['frame']

    def object_mask(self):
        return self._obj_descriptors['mask']

    def object_frame_size(self):
        """
        Devuelve el tamaño de un lado del cuadrado que contiene al objeto
        """
        return self._obj_frame_size

    def object_location(self):
        return self._obj_location

    #########################
    # Metodos de comparacion
    #########################
    def object_comparisson_base(self, img):
        """
        Comparacion base: sirve como umbral para las comparaciones que se
        realizan durante el seguimiento
        """
        filas, columnas = len(img), len(img[0])
        return filas * columnas

    def object_comparisson(self, roi):
        # Hago una comparacion bit a bit de la imagen original
        # Compara solo en la zona de la máscara y deja 0's en donde hay
        # coincidencias y 255's en donde no coinciden
        xor = cv2.bitwise_xor(self.object_roi(), roi, mask=self.object_mask())

        # Cuento la cantidad de 0's y me quedo con la mejor comparacion
        return cv2.countNonZero(xor)

    def is_best_match(self, new_value, old_value):
        return new_value < old_value

    #####################################
    # Esquema de seguimiento del objeto
    #####################################
    def simple_follow(self, img, ubicacion, valor_comparativo,
                      tam_region_inicial):
        """
        Esta funcion es el esquema de seguimiento del objeto.
        """
        filas, columnas = len(img), len(img[0])

        nueva_ubicacion = ubicacion
        tam_region_final = tam_region_inicial

        # Seguimiento (busqueda/deteccion acotada)
        for x, y, tam_region in (self.metodo_de_busqueda
                                 .get_positions_and_framesizes(
                                     ubicacion, tam_region_inicial, filas,
                                     columnas)):
            col_izq = y
            col_der = col_izq + tam_region
            fil_arr = x
            fil_aba = fil_arr + tam_region

            # Tomo una region de la imagen donde se busca el objeto
            roi = img[fil_arr:fil_aba, col_izq:col_der]

            # Si se quiere ver como va buscando, descomentar la siguiente linea
            # MuestraBusquedaEnVivo('Buscando el objeto').run(
            #    img_copy,
            #    (x, y),
            #    tam_region,
            #    None,
            #    frenar=True,
            # )
            nueva_comparacion = self.object_comparisson(roi)

            # Si hubo coincidencia
            if self.is_best_match(nueva_comparacion, valor_comparativo):
                # Nueva ubicacion del objeto (esquina superior izquierda del
                # cuadrado)
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
        # img_copy = img.copy()

        vieja_ubicacion = self.object_location()
        nueva_ubicacion = vieja_ubicacion

        tam_region = self.object_frame_size()
        tam_region_final = tam_region

        # Cantidad de pixeles distintos
        valor_comparativo = self.object_comparisson_base(img)

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

        # Devuelvo self.object_frame_size() porque puede cambiar en
        # "upgrade_descriptors". Idem con self.object_location()
        return fue_exitoso, self.object_frame_size(), self.object_location()

    ##########################
    # Actualizar descriptores
    ##########################
    def set_object_descriptors(self, ubicacion, tam_region, obj_descriptors):
        self._obj_location = ubicacion
        self._obj_frame_size = tam_region
        self._obj_descriptors.update(obj_descriptors)

    def _upgrade_descriptors(self, img, ubicacion, tam_region):
        frame = img[ubicacion[0]:ubicacion[0]+tam_region,
                    ubicacion[1]:ubicacion[1]+tam_region]
        mask = self.calculate_mask(frame)
        obj_descriptors = {'frame': frame, 'mask': mask}

        self.set_object_descriptors(ubicacion, tam_region, obj_descriptors)

    def upgrade_detected_descriptors(self, img, ubicacion, tam_region):
        self._upgrade_descriptors(img, ubicacion, tam_region)

    def upgrade_followed_descriptors(self, img, ubicacion, tam_region):
        self._upgrade_descriptors(img, ubicacion, tam_region)

    #######################
    # Funcion de deteccion
    #######################
    def detect(self, img):
        # Los valores estan harcodeados en el __init__
        self.upgrade_detected_descriptors(
            img,
            self.object_location(),
            self.object_frame_size()
        )
        fue_exitoso = self.object_frame_size() > 0
        return fue_exitoso, self.object_frame_size(), self.object_location()

    ##########################
    # Funciones de cada clase
    ##########################
    def calculate_mask(self, img):
        # Da vuelta los valores (0->255 y 255->0)
        return cv2.bitwise_not(img)


class CalculaSurfMixin(object):
    def detect_features(self, roi):

        # Creo el objeto SURF, pasando el valor del "Hessian threshold"
        surf = cv2.SURF(400)

        # Encontrar keypoints y descriptores
        keypoints, descriptors = surf.detectAndCompute(roi, None)
        return keypoints, descriptors


class ComparacionDeSurfMixin(object):

    def is_best_match(self, new_value, old_value):
        return new_value > old_value

    def object_comparisson_base(self, img):
        """
        Devuelve la cantidad minima de keypoints a matchear
        """
        return 1

    def object_comparisson(self, roi):
        keypoints, descriptors = self.detect_features(roi)
        good = []

        if descriptors is not None:

            # Tomo los keypoints y descriptores guardados
            train_keypoints = self._obj_descriptors['keypoints']
            train_descriptors = self._obj_descriptors['descriptors']

            # Calcular matching entre descriptores
            # BFMatcher with default params
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(train_descriptors, descriptors, k=2)

            if all([len(m) == 2 for m in matches]):
                # Apply ratio test
                for m,n in matches:
                    if m.distance < 0.75*n.distance:
                        good.append([m])

        return len(good)


class MatchingTemplateDetectionMixin(object):
    def detect(self, img):

        template = self._obj_descriptors['template']
        template_filas, template_columnas = len(template), len(template[0])

        # Aplico el template Matching
        res = cv2.matchTemplate(img, template, cv2.TM_SQDIFF_NORMED)

        # Busco la posición
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        # min_loc tiene primero las columnas y despues las filas, entonces lo
        # doy vuelta
        ubicacion = (min_loc[1], min_loc[0])

        tam_region = max(template_columnas, template_filas)

        self.upgrade_detected_descriptors(img, ubicacion, tam_region)

        fue_exitoso = self.object_frame_size() > 0

        return fue_exitoso, self.object_frame_size(), self.object_location()


# class ALittleGeneralObjectDetectorAndFollower(
#         CalculaHistogramaMixin, ComparacionDeHistogramasPorBhattacharyya,
#         MatchingTemplateDetectionMixin, ObjectDetectorAndFollower):
#
#     def __init__(self, image_provider, template,
#                  metodo_de_busqueda=BusquedaAlrededor()):
#         self.img_provider = image_provider
#         self.metodo_de_busqueda = metodo_de_busqueda
#
#         # Object descriptors
#         self._obj_location = None  # Fila, columna
#         self._obj_frame_size = None
#         self._obj_descriptors = {
#             'template': template,
#         }
#
#     def upgrade_detected_descriptors(self, img, ubicacion, tam_region):
#         print "UBICACION:", ubicacion, tam_region
#
#         frame = img[ubicacion[0]:ubicacion[0] + tam_region,
#                     ubicacion[1]:ubicacion[1] + tam_region]
#
#         # Actualizo el histograma
#         hist = self.calculate_histogram(frame)
#
#         obj_descriptors = {'frame': frame, 'hist': hist}
#         self.set_object_descriptors(ubicacion, tam_region, obj_descriptors)
#
#     def upgrade_followed_descriptors(self, img, ubicacion, tam_region):
#         frame = img[ubicacion[0]:ubicacion[0] + tam_region,
#                     ubicacion[1]:ubicacion[1] + tam_region]
#
#         # Actualizo el histograma
#         hist = self.calculate_histogram(frame)
#
#         # TODO: actualizo el template?
#         obj_descriptors = {'frame': frame, 'hist': hist, 'template': frame}
#         self.set_object_descriptors(ubicacion, tam_region, obj_descriptors)
#
#
# class TemplateMatchingAndSURFFollowing(CalculaSurfMixin,
#                                        ComparacionDeSurfMixin,
#                                        MatchingTemplateDetectionMixin,
#                                        ObjectDetectorAndFollower):
#     def __init__(self, image_provider, template,
#                  metodo_de_busqueda=BusquedaAlrededor()):
#         self.img_provider = image_provider
#         self.metodo_de_busqueda = metodo_de_busqueda
#
#         # Object descriptors
#         self._obj_location = None  # Fila, columna
#         self._obj_frame_size = None
#         self._obj_descriptors = {
#             'template': template,
#         }
#
#     def train(self):
#         pass
#
#     def upgrade_detected_descriptors(self, img, ubicacion, tam_region):
#         frame = img[ubicacion[0]:ubicacion[0] + tam_region,
#                     ubicacion[1]:ubicacion[1] + tam_region]
#
#         # Actualizo los keypoints y descriptores
#         kps, desc = self.detect_features(frame)
#
#         obj_descriptors = {
#             'frame': frame,
#             'keypoints': kps,
#             'descriptors': desc,
#         }
#         self.set_object_descriptors(ubicacion, tam_region, obj_descriptors)
#
#     def upgrade_followed_descriptors(self, img, ubicacion, tam_region):
#         self.upgrade_detected_descriptors(img, ubicacion, tam_region)