#!/usr/bin/python
#coding=utf-8

from __future__ import (unicode_literals, division)

from cpp.my_pcl import (icp, filter_cloud, points, get_point)
from detector_estatico_sin_deteccion import StaticDetector
from esquemas_seguimiento import FollowingScheme
from metodos_comunes import (from_flat_to_cloud_limits, from_cloud_to_flat)
from observar_seguimiento import MuestraSeguimientoEnVivo
from proveedores_de_imagenes import FrameNamesAndImageProvider
from seguidores_rgbd import (Follower, Finder)


class FollowerWithStaticDetectionAndPCD(Follower):
    def descriptors(self):
        desc = super(FollowerWithStaticDetectionAndPCD, self).descriptors()
        desc.update({
            'nframe': self.img_provider.current_frame_number(),
            'depth_img': self.img_provider.depth_img(),
            'pcd': self.img_provider.pcd(),
        })
        return desc


class StaticDetectorWithPCDFiltering(StaticDetector):
    def calculate_descriptors(self, detected_descriptors):
        """
        Obtengo la nube de puntos correspondiente a la ubicacion y region
        pasadas por parametro.
        """
        ubicacion = detected_descriptors['location']
        tam_region = detected_descriptors['size']

        depth_img = self._descriptors['depth_img']

        rows_cols_limits = from_flat_to_cloud_limits(
            ubicacion,
            (ubicacion[0] + tam_region, ubicacion[1] + tam_region),
            depth_img,
        )

        r_top_limit = rows_cols_limits[0][0]
        r_bottom_limit = rows_cols_limits[0][1]
        c_left_limit = rows_cols_limits[1][0]
        c_right_limit = rows_cols_limits[1][1]

        cloud = self._descriptors['pcd']

        filter_cloud(cloud, str("y"), float(r_top_limit), float(r_bottom_limit))
        filter_cloud(cloud, str("x"), float(c_left_limit), float(c_right_limit))

        detected_descriptors.update({'object_cloud': cloud})

        return detected_descriptors


class ICPFinder(Finder):

    def find(self):
        object_cloud = self._descriptors['object_cloud']
        target_cloud = self._descriptors['pcd']
        depth_img = self._descriptors['depth_img']

        #TODO: filter target_cloud (por ejemplo, una zona 4 veces mayor)
        size = self._descriptors['size']
        im_c_left = self._descriptors['location'][1]
        im_c_right = im_c_left + size
        im_r_top = self._descriptors['location'][0]
        im_r_bottom = im_r_top + size

        topleft = (im_r_top, im_c_left)
        bottomright = (im_r_bottom, im_c_right)

        rows_cols_limits = from_flat_to_cloud_limits(topleft, bottomright, depth_img)

        r_top_limit = rows_cols_limits[0][0]
        r_bottom_limit = rows_cols_limits[0][1]
        c_left_limit = rows_cols_limits[1][0]
        c_right_limit = rows_cols_limits[1][1]

        # TODO: se puede hacer una busqueda mejor, en espiral o algo asi
        #       tomando como valor de comparacion el score que devuelve ICP

        # Define row and column limits for the zone to search the object
        # In this case, we look on a box N times the size of the original
        n = 4
        r_top_limit = r_top_limit - ( (r_bottom_limit - r_top_limit) * n)
        r_bottom_limit = r_bottom_limit + ( (r_bottom_limit - r_top_limit) * n)
        c_left_limit = c_left_limit - ( (c_right_limit - c_left_limit) * n)
        c_right_limit = c_right_limit + ( (c_right_limit - c_left_limit) * n)

        # Filter points corresponding to the zone where the object being
        # followed is supposed to be
        filter_cloud(target_cloud, str("y"), float(r_top_limit), float(r_bottom_limit))
        filter_cloud(target_cloud, str("x"), float(c_left_limit), float(c_right_limit))

        # Calculate ICP
        icp_result = icp(object_cloud, target_cloud)

        fue_exitoso = icp_result.has_converged
        descriptors = {}

        if fue_exitoso:
            filas = len(depth_img)
            columnas = len(depth_img[0])

            # Busco los limites en el dominio de las filas y columnas del RGB
            col_left_limit = columnas - 1
            col_right_limit = 0
            row_top_limit = filas - 1
            row_bottom_limit = 0

            import rpdb2
            rpdb2.start_embedded_debugger("pass")


            for i in range(points(icp_result.cloud)):
                point_xyz = get_point(icp_result.cloud, i)
                c = point_xyz.x
                r = point_xyz.y
                d = point_xyz.z

                flat_rc = from_cloud_to_flat(r, c, d)

                if flat_rc[0] < row_top_limit:
                    row_top_limit = flat_rc[0]

                if flat_rc[0] > row_bottom_limit:
                    row_bottom_limit = flat_rc[0]

                if flat_rc[1] < col_left_limit:
                    col_left_limit = flat_rc[1]

                if flat_rc[1] > col_right_limit:
                    col_right_limit = flat_rc[1]


            width = col_right_limit - col_left_limit
            height = row_bottom_limit - row_top_limit

            descriptors.update({
                'size': width if width > height else height,
                'location': (row_top_limit, col_left_limit),
                'object_cloud': icp_result.cloud,
            })

        return fue_exitoso, descriptors


def prueba_seguimiento_ICP():
    img_provider = FrameNamesAndImageProvider(
        'videos/rgbd/scenes/', 'desk', '1'
    )  # path, objname, number

    detector = StaticDetectorWithPCDFiltering(
        'videos/rgbd/scenes/desk/desk_1.mat',
        'coffee_mug'
    )

    finder = ICPFinder()

    follower = FollowerWithStaticDetectionAndPCD(img_provider, detector, finder)

    show_following = MuestraSeguimientoEnVivo('Seguidor ICP')

    FollowingScheme(
        img_provider,
        follower,
        show_following,
    ).run()


# HACER mi propia clase "cloud" en C++ y exportarla a python con los
# siguientes metodos:
# 1- el constructor recibe el nombre del archivo y levanta el pcd
# 2- "filter_cloud" que filtra la nube
# CAMBIAR el nombre del metodo "follow" por "icp"
# Ademas hacer que "icp" se calcule recibiendo 2 de estos objetos "cloud"


if __name__ == '__main__':
    prueba_seguimiento_ICP()