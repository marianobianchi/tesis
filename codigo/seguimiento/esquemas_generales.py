#!/usr/bin/python
# -*- coding: utf-8 -*-
#Con esto, todos los strings literales son unicode (no hace falta poner u'algo')
from __future__ import unicode_literals

import cv2

class GeneralSchema(object):

    def __init__(self, img_provider, obj_follower, show_following):
        self.img_provider = img_provider
        self.obj_follower = obj_follower
        self.show_following = show_following

    def run(self):

        #########################
        # Etapa de entrenamiento
        #########################


        ######################
        # Etapa de detección
        ######################

        have_images, img = self.img_provider.read()

        tam_region, ubicacion_inicial = self.obj_follower.detect(img)

        # Muestro el seguimiento para hacer pruebas
        self.show_following.run(img, ubicacion_inicial, tam_region, False, frenar=True)


        #######################
        # Etapa de seguimiento
        #######################

        have_images, img = self.img_provider.read()

        while have_images:

            fue_exitoso, tam_region, nueva_ubicacion = self.obj_follower.follow(img)

            if not fue_exitoso:
                tam_region, nueva_ubicacion = self.obj_follower.detect(img)

            # Muestro el seguimiento para hacer pruebas
            self.show_following.run(img, nueva_ubicacion, tam_region, fue_exitoso, frenar=True)

            # Tomo una nueva imagen en escala de grises
            have_images, img = self.img_provider.read()

        cv2.destroyAllWindows()
        self.show_following.close()


class FollowingSchema(object):

    def __init__(self, detector_and_follower):
        self.detector_and_follower = detector_and_follower
        self.seguidor = self.detector_and_follower.clase_seguidora(self.detector_and_follower.descriptores())

    def simple_follow(self, img, ubicacion, valor_comparativo, tam_region_inicial):
        """
        Esta funcion es el esquema de seguimiento del objeto
        """
        filas, columnas = len(img), len(img[0])

        nueva_ubicacion = ubicacion
        nueva_comparacion = None
        tam_region_final = tam_region_inicial

        # Seguimiento (busqueda/deteccion acotada)
        for x, y, tam_region in (self.detector_and_follower
                                     .metodo_de_busqueda
                                     .get_positions_and_framesizes(ubicacion,
                                                                   tam_region_inicial,
                                                                   filas,
                                                                   columnas)):
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

            nueva_comparacion = self.seguidor.object_comparisson(roi)

            # Si hubo coincidencia
            if self.seguidor.is_best_match(nueva_comparacion, valor_comparativo):
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
        y trata de hacer una busqueda más exhaustiva
        """
        # Descomentar si se quiere ver la busqueda
        #img_copy = img.copy()

        vieja_ubicacion = self.detector_and_follower.object_location()
        nueva_ubicacion = vieja_ubicacion

        tam_region = self.detector_and_follower.object_frame_size()
        tam_region_final = tam_region

        # Valor comparativo base
        valor_comparativo = self.seguidor.object_comparisson_base(img)

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

        return fue_exitoso, tam_region_final, nueva_ubicacion