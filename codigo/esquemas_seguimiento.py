#!/usr/bin/python
# -*- coding: utf-8 -*-
#Con esto, todos los strings literales son unicode (no hace falta poner u'algo')
from __future__ import unicode_literals

import cv2



class NameBasedFollowingScheme(object):

    def __init__(self, img_name_provider, obj_follower, show_following):
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
        self.show_following.run(
            img_list=self.img_name_provider.source_img(),
            ubicacion=ubicacion_inicial,
            tam_region=tam_region,
            fue_exitoso=fue_exitoso,
            es_deteccion=True,
            frenar=True,
        )

        #######################
        # Etapa de seguimiento
        #######################

        while self.img_name_provider.have_images():
            # Tomo el siguiente elemento
            name_dict = self.img_name_provider.next()

            es_deteccion = False
            if fue_exitoso:
                fue_exitoso, tam_region, nueva_ubicacion = (
                self.obj_follower.follow(name_dict)
                )

            else:
                es_deteccion = True
                fue_exitoso, tam_region, nueva_ubicacion = (
                    self.obj_follower.detect(name_dict)
                )

            # Muestro el seguimiento
            self.show_following.run(
                img_list=self.img_name_provider.source_img(),
                ubicacion=nueva_ubicacion,
                tam_region=tam_region,
                fue_exitoso=fue_exitoso,
                es_deteccion=es_deteccion,
                frenar=True,
            )

        self.show_following.close()