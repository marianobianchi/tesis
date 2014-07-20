#!/usr/bin/python
# -*- coding: utf-8 -*-
#Con esto, todos los strings literales son unicode (no hace falta poner u'algo')
from __future__ import unicode_literals

import cv2



class NameBasedFollowingScheme(object):

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

        fue_exitoso, tam_region, ubicacion_inicial = (
            self.obj_follower.detect()
        )
        # TODO: Hacer algo cuando la deteccion no es exitosa

        # Muestro el seguimiento para hacer pruebas
        self.show_following.run(
            img_provider=self.img_provider,
            ubicacion=ubicacion_inicial,
            tam_region=tam_region,
            fue_exitoso=fue_exitoso,
            es_deteccion=True,
            frenar=True,
        )

        #######################
        # Etapa de seguimiento
        #######################

        while self.img_provider.have_images():
            # Adelanto un frame
            self.img_provider.next()

            es_deteccion = False
            if fue_exitoso:
                fue_exitoso, tam_region, nueva_ubicacion = (
                self.obj_follower.follow()
                )

            else:
                es_deteccion = True
                fue_exitoso, tam_region, nueva_ubicacion = (
                    self.obj_follower.detect()
                )

            # Muestro el seguimiento
            self.show_following.run(
                img_provider=self.img_provider,
                ubicacion=nueva_ubicacion,
                tam_region=tam_region,
                fue_exitoso=fue_exitoso,
                es_deteccion=es_deteccion,
                frenar=True,
            )

        self.show_following.close()