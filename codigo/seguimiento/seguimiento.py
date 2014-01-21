#!/usr/bin/python
# -*- coding: utf-8 -*-
#Con esto, todos los strings literales son unicode (no hace falta poner u'algo')
from __future__ import unicode_literals


import os

import numpy as np
import cv2


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


def dibujar_cuadrado(img, (fila_borde_sup_izq, col_borde_sup_izq), tam_region, color=(0,0,0)):
    cv2.rectangle(
        img,
        (col_borde_sup_izq, fila_borde_sup_izq),
        (col_borde_sup_izq+tam_region, fila_borde_sup_izq+tam_region),
        color,
        3
    )
    return img


def ver_seguimiento(img,
                    frame_title,
                    nueva_ubicacion,
                    tam_region,
                    vieja_ubicacion,
                    frenar=False):

    if len(img.shape) == 2:
        # Convierto a imagen a color para dibujar un cuadrado
        color_img = np.zeros((filas, columnas, 3), dtype=np.uint8)
        color_img[:,:,0] = img[:,:]
        color_img[:,:,1] = img[:,:]
        color_img[:,:,2] = img[:,:]
    else:
        color_img = img

    # Cuadrado verde si hubo una coincidencia/cambio de lugar
    # Rojo si no hubo coincidencia alguna
    if nueva_ubicacion == vieja_ubicacion:
        color_img = dibujar_cuadrado(color_img, nueva_ubicacion, tam_region, color=(0,0,255))
    else:
        color_img = dibujar_cuadrado(color_img, nueva_ubicacion, tam_region, color=(0,255,0))

    # Muestro el resultado y espero que se apriete la tecla q
    cv2.imshow(frame_title, color_img)
    if frenar:
        while cv2.waitKey(1) & 0xFF != ord('q'):
            pass



class ImageProvider(object):
    """
    Esta clase se va a encargar de proveer imágenes, ya sea provenientes
    de un video o tiras de frames guardadas en el disco.

    Idea: copiar parte de la API de cv2.VideoCapture
    """
    def read(self, gray=False):
        """
        Cada subclase implementa este método. Debe devolver una imagen
        distinta cada vez, simulando los frames de un video.
        Si gray == True, debe devolver la imagen en escala de grises
        """
        pass


class FramesAsVideo(ImageProvider):

    def __init__(self, path):
        """
        La carpeta apuntada por 'path' debe contener solo imagenes que seran
        devueltas por el metodo 'read'.
        Se devolveran ordenadas alfabéticamente.
        """
        self.path = path
        self.img_filenames = os.listdir(path)
        self.img_filenames.sort()
        self.img_filenames = [os.path.join(path, fn) for fn in self.img_filenames]

    def read(self):
        have_images = len(self.img_filenames) > 0
        img = None
        if have_images:
            # guardo proxima imagen
            img = cv2.imread(self.img_filenames[0])

            self.img_filenames = self.img_filenames[1:] # quito la imagen de la lista

        return (have_images, img)


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
            # ver_seguimiento(img, 'Buscando el objeto', (x,y), tam_region, vieja_ubicacion)

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
        H = 21
        S = 63
        V = 100
        lower_orange = np.array([(H/2)-15,int(S*2.55/2),int(V*2.55/2)])
        upper_orange = np.array([(H/2)+15,max(int(S*2.55)*2, 255),max(int(V*2.55)*2, 255)])


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


class FollowingSchema(object):

    def __init__(self, img_provider, obj_follower):
        self.img_provider = img_provider
        self.obj_follower = obj_follower

    def run(self):

        #########################
        # Etapa de entrenamiento
        #########################


        ######################
        # Etapa de detección
        ######################

        have_images, img = self.img_provider.read()

        tam_region, ultima_ubicacion, img_objeto = self.obj_follower.detect(img)

        #######################
        # Etapa de seguimiento
        #######################

        have_images, img = self.img_provider.read()

        while have_images:

            fue_exitoso, nueva_ubicacion = self.obj_follower.follow(img)

            if not fue_exitoso:
                tam_region, nueva_ubicacion, img_objeto = self.obj_follower.detect(img)

            # Muestro el seguimiento para hacer pruebas
            ver_seguimiento(img,
                            'Seguimiento',
                            nueva_ubicacion,
                            tam_region,
                            ultima_ubicacion,
                            True)

            # Guardo la ultima deteccion para dibujar el seguimiento
            ultima_ubicacion = nueva_ubicacion

            # Tomo una nueva imagen en escala de grises
            have_images, img = self.img_provider.read()

        cv2.destroyAllWindows()


def seguir_pelota_monocromo():
    img_provider = FramesAsVideo('videos/moving_circle')
    follower = ObjectDetectorAndFollower(img_provider)
    FollowingSchema(img_provider, follower).run()


if __name__ == '__main__':
    img_provider = cv2.VideoCapture('../videos/pelotita_naranja_webcam/output.avi')
    follower = OrangeBallDetectorAndFollower(img_provider)
    FollowingSchema(img_provider, follower).run()