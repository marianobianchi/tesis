#!/usr/bin/python
#coding=utf-8

from __future__ import (unicode_literals, division)

import scipy.io

from seguidores_rgbd import (Follower, Detector, Finder)
from proveedores_de_imagenes import FrameNamesAndImageProvider
from observar_seguimiento import MuestraSeguimientoEnVivo
from esquemas_seguimiento import FollowingScheme



class FollowerWithStaticDetection(Follower):
    def descriptors(self):
        desc = super(FollowerWithStaticDetection, self).descriptors()
        desc.update({
            'nframe': self.img_provider.current_frame_number(),
        })
        return desc


class StaticDetector(Detector):
    """
    Esta clase se encarga de definir la ubicación del objeto buscado en la
    imagen valiendose de los datos provistos por la base de datos RGBD.
    Los datos se encuentran almacenados en un archivo ".mat".
    """
    def __init__(self, matfile_path, obj_rgbd_name):
        super(StaticDetector, self).__init__()
        self._matfile = scipy.io.loadmat(matfile_path)['bboxes']
        self._obj_rgbd_name = obj_rgbd_name

    def detect(self):
        nframe = self._descriptors['nframe']

        objs = self._matfile[0][nframe][0]

        fue_exitoso = False
        tam_region = 0
        location = (0, 0)

        for obj in objs:
            if obj[0][0] == self._obj_rgbd_name:
                fue_exitoso = True
                location = (int(obj[2][0][0]), int(obj[4][0][0]))
                tam_region = max(int(obj[3][0][0]) - int(obj[2][0][0]),
                                 int(obj[5][0][0]) - int(obj[4][0][0]))
                break

        detected_descriptors = {
            'size': tam_region,
            'location': location, #location=(fila, columna)
        }

        return fue_exitoso, detected_descriptors


def prueba_de_deteccion_estatica():
    img_provider = FrameNamesAndImageProvider(
        'videos/rgbd/scenes/', 'desk', '1', 'videos/rgbd/objs/', 'coffee_mug', '5',
    )  # path, objname, number

    detector = StaticDetector(
        'videos/rgbd/scenes/desk/desk_1.mat',
        'coffee_mug'
    )

    finder = Finder()

    follower = FollowerWithStaticDetection(img_provider, detector, finder)

    show_following = MuestraSeguimientoEnVivo('Deteccion estatica - Sin seguidor')

    FollowingScheme(
        img_provider,
        follower,
        show_following,
    ).run()


if __name__ == '__main__':
    prueba_de_deteccion_estatica()