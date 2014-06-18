#!/usr/bin/python
# -*- coding: utf-8 -*-
#Con esto, todos los strings literales son unicode (no hace falta poner u'algo')
from __future__ import unicode_literals

import cv2



class NameBasedFollowingScheme(object):

    def __init__(self, img_name_provider, obj_follower, show_following=None):
        self.img_name_provider = img_name_provider
        self.obj_follower = obj_follower
        self.show_following = show_following

    def run(self):
        #########################
        # Etapa de entrenamiento
        #########################

        ######################
        # Etapa de detección
        ######################

        name_dict = self.img_name_provider.next()

        fue_exitoso, tam_region, ubicacion_inicial = (
            self.obj_follower.detect(name_dict)
        )
        # TODO: Hacer algo cuando la deteccion no es exitosa

        # Muestro el seguimiento para hacer pruebas
        #self.show_following.run(
        #    name_dict,
        #    ubicacion_inicial,
        #    tam_region,
        #    False,  # fue_exitoso
        #    frenar=True,
        #)

        #######################
        # Etapa de seguimiento
        #######################

        while self.img_name_provider.have_images():
            # Tomo el siguiente elemento
            name_dict = self.img_name_provider.next()

            fue_exitoso, tam_region, nueva_ubicacion = (
                self.obj_follower.follow(name_dict)
            )

            if not fue_exitoso:
                fue_exitoso, tam_region, nueva_ubicacion = (
                    self.obj_follower.detect(name_dict)
                )

            # Muestro el seguimiento para hacer pruebas
            #self.show_following.run(
            #    name_dict,
            #    nueva_ubicacion,
            #    tam_region,
            #    fue_exitoso,
            #    frenar=True
            #)

        #self.show_following.close()