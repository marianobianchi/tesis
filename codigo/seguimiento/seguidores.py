#!/usr/bin/python
# -*- coding: utf-8 -*-
#Con esto, todos los strings literales son unicode (no hace falta poner u'algo')
from __future__ import unicode_literals


import numpy as np
import cv2

from esquemas_seguimiento import FollowingSchema
from observar_seguimiento import (MuestraSeguimientoEnVivo, MuestraBusquedaEnVivo,
                                  GrabaSeguimientoEnArchivo)
from proveedores_de_imagenes import FramesAsVideo
from metodos_comunes import *
from metodos_de_busqueda import *



class ObjectDetectorAndFollower(object):
    """
    Es la clase base para los primeros ejemplos sencillos.
    Este caso base sirve para el video creado via opencv, el de la pelotita
    negra (moving_circle)
    """

    def __init__(self, image_provider, metodo_de_busqueda=BusquedaEnEspiral()):
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

            nueva_comparacion = self.object_comparisson(roi)

            # Si hubo coincidencia
            if self.is_best_match(nueva_comparacion, valor_comparativo):
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
        self.upgrade_detected_descriptors(img, self.object_location(), self.object_frame_size())
        return self.object_frame_size(), self.object_location()

    ##########################
    # Funciones de cada clase
    ##########################
    def calculate_mask(self, img):
        # Da vuelta los valores (0->255 y 255->0)
        return cv2.bitwise_not(img)


class CalculaMascaraPorColorNaranjaMixin(object):
    def calculate_mask(self, img):
        # Convert BGR to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # define range of orange color in HSV
        # HSV from OpenCV valid values: (H:0-180, S:0-255, V:0-255)
        # HSV from GIMP valid values: (H:0-360, S:0-100, V:0-100)
        H = 35 # Del GIMP
        S = 87 # Del GIMP
        V = 100 # Del GIMP
        lower_orange = np.array([int((H/2)-9), int(S*2.55/2), int(V*2.55/2)])
        upper_orange = np.array([int((H/2)+9), min(int(S*2.55)*2, 255), min(int(V*2.55)*2, 255)])


        # Threshold the HSV image to get only orange colors
        mask = cv2.inRange(hsv, lower_orange, upper_orange)

        # Uso transformaciones morfologicas para limpiar un poco la mascara
        # IMPORTANTE EL KERNEL. CON EL PRIMERO TERMINA MUY RAPIDO. CON LOS
        # OTROS DOS FUNCIONA MAS TIEMPO
        #kernel = np.ones((7,7), np.uint8)
        #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(6,6))
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(10,10))

        opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

        return closing


class ComparacionPorDiferenciaCuadraticaMixin(object):
    #########################
    # Metodos de comparacion
    #########################
    def object_comparisson_base(self, img):
        return 2000

    def object_comparisson(self, roi):
        # Comparación simple: distancia euclideana
        return cv2.norm(roi, self.object_roi(), mask=self.object_mask())


class ComparacionPorCantidadDePixelesIgualesMixin(object):
    def object_comparisson_base(self, img):
        """
        Idea: que a lo sumo encuentre 1/2 del objeto
        """
        return cv2.countNonZero(self.object_mask()) / 2

    def object_comparisson(self, roi):
        """
        Asumiendo que queda en blanco la parte del objeto que estamos
        buscando, comparo la cantidad de blancos entre el objeto guardado y
        el objeto que se esta observando.
        """
        # Calculo la máscara del pedazo de imagen que estoy mirando
        roi_mask = self.calculate_mask(roi)

        # Hago una comparacion bit a bit de la imagen original
        # Compara solo en la zona de la máscara y deja 0's en donde hay
        # coincidencias y 255's en donde no coinciden
        xor = cv2.bitwise_xor(self.object_mask(), roi_mask, mask=self.object_mask())

        # Cuento la cantidad de nros distintos a 0 y me quedo con la mejor comparacion
        return cv2.countNonZero(xor)


class DeteccionDePelotaNaranjaPorContornosMixin(object):
    #######################
    # Funcion de deteccion
    #######################
    def detect(self, img):
        # Filtro los objetos de color naranja de toda la imagen
        cleaned_image_mask = self.calculate_mask(img)

        # Calculo los contornos
        contours, hierarchy = cv2.findContours(
            cleaned_image_mask,
            cv2.RETR_LIST, # Forma en que se devuelven los contornos (jerarquia)
            cv2.CHAIN_APPROX_SIMPLE # Cantidad de puntos del contorno
        )

        max_perimeter = 0
        best_contour = None
        for contour in contours:
            # Busco el perímetro del contorno asumiendo que la forma es cerrada,
            # es decir, que es algo solido y no una curva
            perimeter = cv2.arcLength(contour, True)

            if perimeter > max_perimeter:
                max_perimeter = perimeter
                best_contour = contour

        if len(best_contour) > 0:

            # Punta izquierda de arriba del rectangulo y alto y ancho
            y, x, w, h = cv2.boundingRect(best_contour)

            # Tamaño de la región de imagen que se usará para detectar y seguir
            tam_region = min(w, h) # Uso el minimo ya que el maximo se puede ir de rango

            # Detectar objeto: Cuadrado donde esta el objeto
            ubicacion = (x, y) # Fila, columna

            # Este es el objeto a seguir
            img_objeto = img[ubicacion[0]:ubicacion[0]+tam_region,
                             ubicacion[1]:ubicacion[1]+tam_region]

            # Mascara del objeto
            mask_objeto = self.calculate_mask(img_objeto)

        else:
            # TODO: Ver que hacer.... Creo que conviene levantar una excepcion
            tam_region = 0
            ubicacion = (0,0)

        self.upgrade_detected_descriptors(img, ubicacion, tam_region)

        return self.object_frame_size(), self.object_location()


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

        return self.object_frame_size(), self.object_location()


class CalculaHistogramaMixin(object):
    def calculate_histogram(self, roi):
        # Paso la imagen de BGR a HSV
        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        ###################################
        # TODO: Ver por que falla al usar los 3 canales

        # Aplico equalizacion de histograma en V (http://en.wikipedia.org/wiki/Histogram_equalization#Histogram_equalization_of_color_images)
        #roi_hsv[:,:,2] = cv2.equalizeHist(roi_hsv[:,:,2])

        #hist = cv2.calcHist(
        #    [roi_hsv], # Imagen
        #    [0,1,2], # Canales
        #    None, # Mascara
        #    [180, 256, 256], # Numero de bins para cada canal
        #    [0,180,0,256,0,256], # Rangos válidos para los pixeles de cada canal
        #)
        ########################################

        # Calculo el histograma del roi (para H y S)
        hist = cv2.calcHist(
            [roi_hsv], # Imagen
            [0,1], # Canales
            None, # Mascara
            [180, 256], # Numero de bins para cada canal
            [0,180,0,256], # Rangos válidos para los pixeles de cada canal
        )

        # Normalizo el histograma para evitar errores por distinta escala
        hist = cv2.normalize(hist)

        return hist


class ComparacionDeHistogramasPorBhattacharyya(object):
    def saved_object_comparisson(self):
        if 'hist' not in self._obj_descriptors:
            obj = self.object_roi()
            hist = self.calculate_histogram(obj)
            self._obj_descriptors['hist'] = hist

        return self._obj_descriptors['hist']

    def object_comparisson_base(self, img):
        # TODO: ver que valor conviene poner. Esto es el umbral para la
        # deteccion en el seguimiento
        return 0.6

    def object_comparisson(self, roi):
        roi_hist = self.calculate_histogram(roi)

        # Tomo el histograma del objeto para comparar
        obj_hist = self.saved_object_comparisson()

        return cv2.compareHist(roi_hist, obj_hist, cv2.cv.CV_COMP_BHATTACHARYYA)


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
        """

        """
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


class OrangeBallDetectorAndFollower(CalculaMascaraPorColorNaranjaMixin,
                                    ComparacionPorDiferenciaCuadraticaMixin,
                                    ObjectDetectorAndFollower):
    def __init__(self, image_provider, metodo_de_busqueda=BusquedaEnEspiral()):
        self.img_provider = image_provider
        self.metodo_de_busqueda = metodo_de_busqueda

        # Object descriptors
        self._obj_location = (128, 492) # Fila, columna
        self._obj_frame_size = 76
        self._obj_descriptors = {}


class OrangeBallDetectorAndFollowerVersion2(CalculaMascaraPorColorNaranjaMixin,
                                            ComparacionPorCantidadDePixelesIgualesMixin,
                                            DeteccionDePelotaNaranjaPorContornosMixin,
                                            ObjectDetectorAndFollower):
    def __init__(self, image_provider, metodo_de_busqueda=BusquedaEnEspiral()):
        self.img_provider = image_provider
        self.metodo_de_busqueda = metodo_de_busqueda

        # Object descriptors
        self._obj_location = None # Fila, columna
        self._obj_frame_size = None
        self._obj_descriptors = {}


class OrangeBallDetectorAndFollowerVersion3(CalculaHistogramaMixin,
                                            ComparacionDeHistogramasPorBhattacharyya,
                                            CalculaMascaraPorColorNaranjaMixin,
                                            DeteccionDePelotaNaranjaPorContornosMixin,
                                            ObjectDetectorAndFollower):
    """
    Comparacion por histograma usando Bhattacharyya
    """
    def __init__(self, image_provider, metodo_de_busqueda=BusquedaEnEspiralCambiandoFrameSize()):
        self.img_provider = image_provider
        self.metodo_de_busqueda = metodo_de_busqueda

        # Object descriptors
        self._obj_location = None # Fila, columna
        self._obj_frame_size = None
        self._obj_descriptors = {}

    ##########################
    # Actualizar descriptores
    ##########################
    def upgrade_detected_descriptors(self, img, ubicacion, tam_region):
        (super(OrangeBallDetectorAndFollowerVersion3, self)
         .upgrade_detected_descriptors(img, ubicacion, tam_region))

        # Actualizo el histograma
        roi = self.object_roi()
        roi = cv2.equalizeHist(roi)
        hist = self.calculate_histogram(roi)
        self._obj_descriptors['hist'] = hist

    def upgrade_followed_descriptors(self, img, ubicacion, tam_region):
        (super(OrangeBallDetectorAndFollowerVersion3, self)
         .upgrade_followed_descriptors(img, ubicacion, tam_region))

        # Actualizo el histograma
        roi = self.object_roi()
        roi = cv2.equalizeHist(roi)
        hist = self.calculate_histogram(roi)
        self._obj_descriptors['hist'] = hist


class ALittleGeneralObjectDetectorAndFollower(CalculaHistogramaMixin,
                                              ComparacionDeHistogramasPorBhattacharyya,
                                              MatchingTemplateDetectionMixin,
                                              ObjectDetectorAndFollower):
    def __init__(self, image_provider, template, metodo_de_busqueda=BusquedaEnEspiral()):
        self.img_provider = image_provider
        self.metodo_de_busqueda = metodo_de_busqueda

        # Object descriptors
        self._obj_location = None # Fila, columna
        self._obj_frame_size = None
        self._obj_descriptors = {
            'template': template,
        }

    def upgrade_detected_descriptors(self, img, ubicacion, tam_region):

        print "UBICACION:", ubicacion, tam_region

        frame = img[ubicacion[0]:ubicacion[0]+tam_region,
                    ubicacion[1]:ubicacion[1]+tam_region]

        # Actualizo el histograma
        hist = self.calculate_histogram(frame)

        obj_descriptors = {'frame': frame, 'hist': hist}
        self.set_object_descriptors(ubicacion, tam_region, obj_descriptors)

    def upgrade_followed_descriptors(self, img, ubicacion, tam_region):
        frame = img[ubicacion[0]:ubicacion[0]+tam_region,
                    ubicacion[1]:ubicacion[1]+tam_region]

        # Actualizo el histograma
        hist = self.calculate_histogram(frame)

        # TODO: actualizo el template?
        obj_descriptors = {'frame': frame, 'hist': hist, 'template': frame}
        self.set_object_descriptors(ubicacion, tam_region, obj_descriptors)


class TemplateMatchingAndSURFFollowing(CalculaSurfMixin,
                                       ComparacionDeSurfMixin,
                                       MatchingTemplateDetectionMixin,
                                       ObjectDetectorAndFollower):
    def __init__(self, image_provider, template, metodo_de_busqueda=BusquedaEnEspiral()):
        self.img_provider = image_provider
        self.metodo_de_busqueda = metodo_de_busqueda

        # Object descriptors
        self._obj_location = None # Fila, columna
        self._obj_frame_size = None
        self._obj_descriptors = {
            'template': template,
        }

    def upgrade_detected_descriptors(self, img, ubicacion, tam_region):

        frame = img[ubicacion[0]:ubicacion[0]+tam_region,
                    ubicacion[1]:ubicacion[1]+tam_region]

        # Actualizo los keypoints y descriptores
        kps, desc = self.detect_features(frame)

        obj_descriptors = {'frame': frame, 'keypoints': kps, 'descriptors': desc}
        self.set_object_descriptors(ubicacion, tam_region, obj_descriptors)

    def upgrade_followed_descriptors(self, img, ubicacion, tam_region):
        self.upgrade_detected_descriptors(img, ubicacion, tam_region)


def seguir_pelota_monocromo():
    img_provider = FramesAsVideo('videos/moving_circle')
    follower = ObjectDetectorAndFollower(img_provider)
    muestra_seguimiento = MuestraSeguimientoEnVivo('Seguimiento')
    FollowingSchema(img_provider, follower, muestra_seguimiento).run()


def seguir_pelota_naranja():
    """
    """
    img_provider = cv2.VideoCapture('../videos/pelotita_naranja_webcam/output.avi')
    follower = OrangeBallDetectorAndFollower(img_provider)
    muestra_seguimiento = MuestraSeguimientoEnVivo('Seguimiento')
    FollowingSchema(img_provider, follower, muestra_seguimiento).run()


def seguir_pelota_naranja_version2():
    """
    """
    img_provider = cv2.VideoCapture('../videos/pelotita_naranja_webcam/output.avi')
    follower = OrangeBallDetectorAndFollowerVersion2(img_provider)
    muestra_seguimiento = MuestraSeguimientoEnVivo(nombre='Seguimiento')
    FollowingSchema(img_provider, follower, muestra_seguimiento).run()

def seguir_pelota_naranja_version3():
    """
    """
    img_provider = cv2.VideoCapture('../videos/pelotita_naranja_webcam/output.avi')
    follower = OrangeBallDetectorAndFollowerVersion3(img_provider)
    muestra_seguimiento = MuestraSeguimientoEnVivo(nombre='Seguimiento')
    FollowingSchema(img_provider, follower, muestra_seguimiento).run()

def seguir_pelota_naranja_version4():
    """
    Template matching y comparacion de histogramas
    """
    img_provider = cv2.VideoCapture('../videos/pelotita_naranja_webcam/output.avi')
    template = cv2.imread('../videos/pelotita_naranja_webcam/template_pelota.jpg')
    follower = ALittleGeneralObjectDetectorAndFollower(img_provider, template, metodo_de_busqueda=BusquedaEnEspiralCambiandoFrameSize())
    muestra_seguimiento = MuestraSeguimientoEnVivo(nombre='Seguimiento')
    FollowingSchema(img_provider, follower, muestra_seguimiento).run()



if __name__ == '__main__':
    #keypoint_matching_entre_dos_imagenes()


    img_provider = cv2.VideoCapture('../videos/pelotita_naranja_webcam/output.avi')
    template = cv2.imread('../videos/pelotita_naranja_webcam/template_pelota.jpg')
    follower = TemplateMatchingAndSURFFollowing(img_provider, template, metodo_de_busqueda=BusquedaEnEspiralCambiandoFrameSize())
    muestra_seguimiento = MuestraSeguimientoEnVivo(nombre='Seguimiento')
    FollowingSchema(img_provider, follower, muestra_seguimiento).run()
