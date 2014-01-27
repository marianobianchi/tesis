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

        tam_region, ubicacion_inicial, img_objeto, mask_objeto = self.obj_follower.detect(img)
        self.obj_follower.set_object_descriptors(
            ubicacion_inicial,
            img_objeto,
            mask_objeto,
            tam_region
        )

        # Muestro el seguimiento para hacer pruebas
        self.show_following.run(img, ubicacion_inicial, tam_region, False, frenar=True)


        #######################
        # Etapa de seguimiento
        #######################

        have_images, img = self.img_provider.read()

        while have_images:

            fue_exitoso, tam_region, nueva_ubicacion, img_objeto, mask_objeto = self.obj_follower.follow(img)

            if not fue_exitoso:
                tam_region, nueva_ubicacion, img_objeto, mask_objeto = self.obj_follower.detect(img)

            self.obj_follower.set_object_descriptors(
                nueva_ubicacion,
                img_objeto,
                mask_objeto,
                tam_region
            )

            # Muestro el seguimiento para hacer pruebas
            self.show_following.run(img, nueva_ubicacion, tam_region, fue_exitoso, frenar=True)

            # Tomo una nueva imagen en escala de grises
            have_images, img = self.img_provider.read()

        cv2.destroyAllWindows()