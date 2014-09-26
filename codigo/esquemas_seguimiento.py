#coding=utf-8

from __future__ import unicode_literals

import os

from cpp.common import save_pcd

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


class FollowingSchemeSavingData(FollowingScheme):
    def __init__(self, img_provider, obj_follower, path):
        super(FollowingSchemeSavingData, self).__init__(
            img_provider,
            obj_follower,
            None,
        )

        self.results_path = os.path.join(path, '{s}_{sn}/{o}_{on}/')
        self.results_path = self.results_path.format(
            s=self.img_provider.scene,
            sn=self.img_provider.scene_number,
            o=self.img_provider.obj,
            on=self.img_provider.obj_number,
        )
        if not os.path.isdir(self.results_path):
            os.makedirs(self.results_path)
        self.file = open(self.results_path + 'results.txt', 'w')

    def __del__(self):
        self.file.close()

    def run(self):
        # ########################
        # Etapa de entrenamiento
        #########################
        self.obj_follower.train()

        ######################
        # Etapa de detección
        ######################
        print "Buscando/Detectando en imagen {i}".format(
            i=self.img_provider.next_frame_number
        )

        fue_exitoso, tam_region, ubicacion_inicial = (
            self.obj_follower.detect()
        )

        self.save_result(0, fue_exitoso, ubicacion_inicial, tam_region)

        #######################
        # Etapa de seguimiento
        #######################

        # Adelanto un frame
        self.img_provider.next()

        while self.img_provider.have_images():

            print "Buscando/Detectando en imagen {i}".format(
                i=self.img_provider.next_frame_number
            )

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

            self.save_result(
                0 if es_deteccion else 1,
                fue_exitoso,
                nueva_ubicacion,
                tam_region
            )

            # Adelanto un frame
            self.img_provider.next()

    def save_result(self, method, fue_exitoso, ubicacion_inicial,
                    tam_region):
        """
        Formato para guardar:

        frame_number;exito;metodo;fila_sup;col_izq;fila_inf;col_der

        donde:
        metodo = 0 si deteccion, 1 si seguimiento
        exito = 0 si fallo, 1 si funciono
        """
        nframe = self.img_provider.next_frame_number
        exito = 1 if fue_exitoso else 0
        fila_sup = ubicacion_inicial[0]
        col_izq = ubicacion_inicial[1]
        fila_inf = fila_sup + tam_region
        col_der = col_izq + tam_region

        values = [nframe, exito, method, fila_sup, col_izq, fila_inf, col_der]

        self.file.write(b';'.join([str(o) for o in values]))
        self.file.write(b'\n')
        self.file.flush()

        if fue_exitoso and 'object_cloud' in self.obj_follower.descriptors():
            pcd = self.obj_follower.descriptors()['object_cloud']
            pcd_filename = 'obj_found_frame_{i:03}.pcd'.format(i=nframe)
            filename = os.path.join(self.results_path, pcd_filename)
            save_pcd(pcd, str(filename))