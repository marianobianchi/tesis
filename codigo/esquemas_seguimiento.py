#coding=utf-8

from __future__ import unicode_literals


class FollowingScheme(object):

    def __init__(self, img_provider, obj_follower, show_following):
        self.img_provider = img_provider
        self.obj_follower = obj_follower
        self.show_following = show_following

    def run(self):
        #########################
        # Etapa de entrenamiento
        #########################
        self.obj_follower.train()

        ######################
        # Etapa de detección
        ######################
        fue_exitoso, tam_region, ubicacion_inicial = (
            self.obj_follower.detect()
        )

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

        # Adelanto un frame
        self.img_provider.next()

        while self.img_provider.have_images():
            es_deteccion = False
            if fue_exitoso:
                fue_exitoso, tam_region, nueva_ubicacion = (
                    self.obj_follower.follow()
                )

            if not fue_exitoso:
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

            # Adelanto un frame
            self.img_provider.next()

        self.show_following.close()