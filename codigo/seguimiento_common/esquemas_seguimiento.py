#!/usr/bin/python
# -*- coding: utf-8 -*-
#Con esto, todos los strings literales son unicode (no hace falta poner u'algo')
from __future__ import unicode_literals

import cv2

class FollowingSchema(object):

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

        fue_exitoso, tam_region, ubicacion_inicial = self.obj_follower.detect(img)
        # TODO: Hacer algo cuando la deteccion no es exitosa

        # Muestro el seguimiento para hacer pruebas
        self.show_following.run(
            img,
            ubicacion_inicial,
            tam_region,
            False, # fue_exitoso
            frenar=True,
        )

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


class FollowingSchemaCountingFrames(FollowingSchema):

    def run(self):

        #########################
        # Etapa de entrenamiento
        #########################


        ######################
        # Etapa de detección
        ######################

        have_images, img = self.img_provider.read()

        fue_exitoso, tam_region, ubicacion_inicial = self.obj_follower.detect(img, nframe=self.img_provider.nframe())
        # TODO: Hacer algo cuando la deteccion no es exitosa

        # Muestro el seguimiento para hacer pruebas
        self.show_following.run(
            img,
            ubicacion_inicial,
            tam_region,
            False, # fue_exitoso
            frenar=True,
        )

        #######################
        # Etapa de seguimiento
        #######################

        have_images, img = self.img_provider.read()

        while have_images:

            fue_exitoso, tam_region, nueva_ubicacion = self.obj_follower.follow(img)

            if not fue_exitoso:
                tam_region, nueva_ubicacion = self.obj_follower.detect(img, nframe=self.img_provider.nframe())

            # Muestro el seguimiento para hacer pruebas
            self.show_following.run(img, nueva_ubicacion, tam_region, fue_exitoso, frenar=True)

            # Tomo una nueva imagen en escala de grises
            have_images, img = self.img_provider.read()

        cv2.destroyAllWindows()
        self.show_following.close()