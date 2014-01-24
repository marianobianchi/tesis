#!/usr/bin/python
# -*- coding: utf-8 -*-
#Con esto, todos los strings literales son unicode (no hace falta poner u'algo')
from __future__ import unicode_literals


import numpy as np
import cv2

from esquemas_seguimiento import FollowingSchema
from observar_seguimiento import MuestraSeguimientoEnVivo
from proveedores_de_imagenes import FramesAsVideo


def espiral_desde((x, y), tam_region, filas, columnas):
    sum_x = 2
    sum_y = 2
    for j in range(50):
        # Hago 2 busquedas por cada nuevo X
        x += sum_x/2
        if (0 <= x <= (x+tam_region) <= filas) and (0 <= y <= (y+tam_region) <= columnas):
            yield (x, y)

        x += sum_x/2
        if (0 <= x <= (x+tam_region) <= filas) and (0 <= y <= (y+tam_region) <= columnas):
            yield (x, y)

        sum_x *= -2

        # Hago 2 busquedas por cada nuevo Y
        y += sum_y/2
        if (0 <= x <= (x+tam_region) <= filas) and (0 <= y <= (y+tam_region) <= columnas):
            yield (x, y)

        y += sum_y/2
        if (0 <= x <= (x+tam_region) <= filas) and (0 <= y <= (y+tam_region) <= columnas):
            yield (x, y)

        sum_y *= -2


class ObjectDetectorAndFollower(object):
    """
    Es la clase base para los primeros ejemplos sencillos.
    Este caso base sirve para el video creado via opencv, el de la pelotita
    negra (moving_circle)
    """

    def __init__(self, image_provider):
        self.img_provider = image_provider

        # Object descriptors
        self._obj_location = (40, 40)
        self._obj_frame = None
        self._obj_frame_mask = None
        self._obj_frame_size = 80

    def object_roi(self):
        return self._obj_frame

    def object_mask(self):
        return self._obj_frame_mask

    def object_frame_size(self):
        """
        Devuelve el tamaño de un lado del cuadrado que contiene al objeto
        """
        return self._obj_frame_size

    def object_location(self):
        return self._obj_location

    def set_object_location(self, location):
        self._obj_location = location

    def object_comparisson_base(self, img):
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

    def follow(self, img):

        vieja_ubicacion = self.object_location()

        filas, columnas = len(img), len(img[0])

        # Cantidad de pixeles distintos
        valor_comparativo = self.object_comparisson_base(img)

        # Seguimiento (busqueda/deteccion acotada)
        for x, y in espiral_desde(self.object_location(), self.object_frame_size(), filas, columnas):
            col_izq = y
            col_der = col_izq + self.object_frame_size()
            fil_arr = x
            fil_aba = fil_arr + self.object_frame_size()

            # Tomo una region de la imagen donde se busca el objeto
            roi = img[fil_arr:fil_aba,col_izq:col_der]

            # Si se quiere ver como va buscando, descomentar la siguiente linea
            # ver_seguimiento(img, 'Buscando el objeto', (x,y), tam_region, (x,y)==vieja_ubicacion)

            nueva_comparacion = self.object_comparisson(roi)

            # Si hubo coincidencia
            if self.is_best_match(nueva_comparacion, valor_comparativo):
                # Nueva ubicacion del objeto (esquina superior izquierda del cuadrado)
                self.set_object_location((x, y))

                # Actualizo el valor de la comparacion
                valor_comparativo = nueva_comparacion

        fue_exitoso = (vieja_ubicacion == self.object_location())
        nueva_ubicacion = self.object_location if fue_exitoso else None

        return fue_exitoso, nueva_ubicacion

    def calculate_mask(self, img):
        # Da vuelta los valores (0->255 y 255->0)
        return cv2.bitwise_not(img)

    def detect(self, img):
        # Tamaño de la región de imagen que se usará para detectar y seguir
        tam_region = self.object_frame_size() # Pixeles cada lado

        # Detectar objeto: Cuadrado donde esta el objeto
        ubicacion = self.object_location() # Fila, columna

        # Este es el objeto a seguir
        self._obj_frame = img[ubicacion[0]:ubicacion[0]+tam_region,
                              ubicacion[1]:ubicacion[1]+tam_region]

        # Mascara del objeto
        self._obj_frame_mask = self.calculate_mask(self.object_roi())

        return tam_region, ubicacion, self.object_roi()


class OrangeBallDetectorAndFollower(ObjectDetectorAndFollower):

    def __init__(self, image_provider):
        self.img_provider = image_provider

        # Object descriptors
        self._obj_location = (128, 492) # Fila, columna
        self._obj_frame = None
        self._obj_frame_mask = None
        self._obj_frame_size = 76

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

    def object_comparisson_base(self, img):
        return cv2.norm(img)

    def object_comparisson(self, roi):
        # Comparación simple: distancia euclideana
        return cv2.norm(roi, self.object_roi(), mask=self.object_mask())


class OrangeBallDetectorAndFollowerVersion2(OrangeBallDetectorAndFollower):

    def __init__(self, image_provider):
        self.img_provider = image_provider

        # Object descriptors
        self._obj_location = None # Fila, columna
        self._obj_frame = None
        self._obj_frame_mask = None
        self._obj_frame_size = None

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
            tam_region = self._obj_frame_size = min(w, h) # Uso el minimo ya que el maximo se puede ir de rango

            # Detectar objeto: Cuadrado donde esta el objeto
            ubicacion = self._obj_location = (x, y) # Fila, columna

            # Este es el objeto a seguir
            self._obj_frame = img[ubicacion[0]:ubicacion[0]+tam_region,
                                  ubicacion[1]:ubicacion[1]+tam_region]

            # Mascara del objeto
            self._obj_frame_mask = self.calculate_mask(self.object_roi())

            return tam_region, ubicacion, self.object_roi()

        else:
            # TODO: Ver que hacer.... Creo que conviene levantar una excepcion
            return 0, (0,0), []

    def follow(self, img):
        """
        IDEA (EN CASO QUE LAS IDEAS DE COMPARACION NO FUNCIONEN):
        Acomodar todo para que la busqueda sea similar a la deteccion pero en
        una superficie apenas mayor que la del "object_frame_size". Acualizar
        en cada paso esta variable y la máscara del objeto, ademas de la
        ubicacion que ya se hacia.
        """
        vieja_ubicacion = self.object_location()

        filas, columnas = len(img), len(img[0])

        # Cantidad de pixeles distintos
        valor_comparativo = self.object_comparisson_base(img)

        # Seguimiento (busqueda/deteccion acotada)
        for x, y in espiral_desde(self.object_location(), self.object_frame_size(), filas, columnas):
            col_izq = y
            col_der = col_izq + self.object_frame_size()
            fil_arr = x
            fil_aba = fil_arr + self.object_frame_size()

            # Tomo una region de la imagen donde se busca el objeto
            roi = img[fil_arr:fil_aba,col_izq:col_der]

            # Si se quiere ver como va buscando, descomentar la siguiente linea
            # ver_seguimiento(img, 'Buscando el objeto', (x,y), tam_region, (x,y)==vieja_ubicacion)

            nueva_comparacion = self.object_comparisson(roi)

            # Si hubo coincidencia
            if self.is_best_match(nueva_comparacion, valor_comparativo):
                # Nueva ubicacion del objeto (esquina superior izquierda del cuadrado)
                self.set_object_location((x, y))

                # Actualizo el valor de la comparacion
                valor_comparativo = nueva_comparacion

        fue_exitoso = (vieja_ubicacion == self.object_location())
        nueva_ubicacion = self.object_location if fue_exitoso else None

        return fue_exitoso, nueva_ubicacion

    def object_comparisson(self, roi):
        """
        IDEA: asumiendo que queda en blanco la parte del objeto que estamos
        buscando, comparar la cantidad de blancos entre el objeto guardado y
        el objeto que se esta observando, permitiendo una cierta variacion.

        NO se puede comparar roi con roi a lo bestia ya que son de tamaño
        variable

        IMPORTANTE: hay que ir actualizando el "object_roi" y el "object_mask"
        en cada seguimiento/deteccion exitoso
        """
        # Calculo la máscara del pedazo de imagen que estoy mirando
        roi_mask = self.calculate_mask(roi)

        # Calculo la máscara del objeto guardado
        obj_mask = self.calculate_mask(self.object_roi())

        # Comparación simple: distancia euclideana
        return cv2.norm(roi_mask, obj_mask, mask=self.object_mask())



class GeneralObjectDetectorAndFollower(ObjectDetectorAndFollower):
    def object_comparisson(self, roi):
        pass




def seguir_pelota_monocromo():
    img_provider = FramesAsVideo('videos/moving_circle')
    follower = ObjectDetectorAndFollower(img_provider)
    muestra_seguimiento = MuestraSeguimientoEnVivo('Seguimiento')
    FollowingSchema(img_provider, follower, muestra_seguimiento).run()


def seguir_pelota_naranja():
    img_provider = cv2.VideoCapture('../videos/pelotita_naranja_webcam/output.avi')
    follower = OrangeBallDetectorAndFollower(img_provider)
    muestra_seguimiento = MuestraSeguimientoEnVivo('Seguimiento')
    FollowingSchema(img_provider, follower, muestra_seguimiento).run()



if __name__ == '__main__':
    img_provider = cv2.VideoCapture('../videos/pelotita_naranja_webcam/output.avi')
    follower = OrangeBallDetectorAndFollowerVersion2(img_provider)
    muestra_seguimiento = MuestraSeguimientoEnVivo(nombre='Seguimiento')
    FollowingSchema(img_provider, follower, muestra_seguimiento).run()