#coding=utf-8

from __future__ import (unicode_literals, division)


from cpp.my_pcl import icp, ICPDefaults, filter_cloud, points, get_point, \
    save_pcd, get_min_max, show_clouds, ICPResult, filter_object_from_scene_cloud
from cpp.alignment_prerejective import align, APDefaults
from metodos_comunes import measure_time, from_flat_to_cloud_limits, \
    from_cloud_to_flat


class Finder(object):
    """
    Es la clase que se encarga de buscar el objeto
    """
    def __init__(self):
        self._descriptors = {}

    def update(self, descriptors):
        self._descriptors.update(descriptors)

    def calculate_descriptors(self, desc):
        """
        Calcula los descriptores en base al objeto encontrado para que
        los almacene el Follower
        """
        if 'detected_cloud' in desc:
            desc['object_cloud'] = desc['detected_cloud']

        return desc

    def base_comparisson(self):
        """
        Comparacion base: sirve como umbral para las comparaciones que se
        realizan durante el seguimiento
        """
        return 0

    def comparisson(self, roi):
        return 0

    def is_best_match(self, new_value, old_value):
        return False

    #####################################
    # Esquema de seguimiento del objeto
    #####################################
    def find(self):
        return False, {}


class ICPFinder(Finder):

    def simple_follow(self, object_cloud, target_cloud):
        """
        Tomando como centro el centro del cuadrado que contiene al objeto
        en el frame anterior, busco el mismo objeto en una zona 4 veces mayor
        a la original.
        """
        # TODO: se puede hacer una busqueda mejor, en espiral o algo asi
        # tomando como valor de comparacion el score que devuelve ICP

        r_top_limit = self._descriptors['min_y_cloud']
        r_bottom_limit = self._descriptors['max_y_cloud']
        c_left_limit = self._descriptors['min_x_cloud']
        c_right_limit = self._descriptors['max_x_cloud']

        # Define row and column limits for the zone to search the object
        # In this case, we look on a box N times the size of the original
        n = 2

        factor = 0.5 * (n - 1)
        height = r_bottom_limit - r_top_limit
        width = c_right_limit - c_left_limit

        r_top_limit -= height * factor
        r_bottom_limit += height * factor
        c_left_limit -= width * factor
        c_right_limit += width * factor

        # Filter points corresponding to the zone where the object being
        # followed is supposed to be
        filter_cloud(
            target_cloud,
            str("y"),
            float(r_top_limit),
            float(r_bottom_limit)
        )
        filter_cloud(
            target_cloud,
            str("x"),
            float(c_left_limit),
            float(c_right_limit)
        )

        # Calculate ICP
        icp_defaults = ICPDefaults()
        icp_result = icp(object_cloud, target_cloud, icp_defaults)

        return icp_result

    def find(self):
        # Obtengo pcd's y depth
        object_cloud = self._descriptors['object_cloud']
        target_cloud = self._descriptors['pcd']

        icp_result = self.simple_follow(
            object_cloud,
            target_cloud,
        )

        fue_exitoso = icp_result.score < 5e-5
        descriptors = {}

        if fue_exitoso:
            # filas = len(depth_img)
            # columnas = len(depth_img[0])

            # Busco los limites en el dominio de las filas y columnas del RGB
            min_max = get_min_max(icp_result.cloud)

            med_z = (min_max.max_z - min_max.min_z) / 2 + min_max.min_z

            min_flat_rc = from_cloud_to_flat(min_max.min_y, min_max.min_x, med_z)
            row_top_limit = min_flat_rc[0]
            col_left_limit = min_flat_rc[1]

            max_flat_rc = from_cloud_to_flat(min_max.max_y, min_max.max_x, med_z)
            row_bottom_limit = max_flat_rc[0]
            col_right_limit = max_flat_rc[1]

            width = col_right_limit - col_left_limit
            height = row_bottom_limit - row_top_limit

            descriptors.update({
                'size': max(width, height),
                'location': (row_top_limit, col_left_limit),
                'detected_cloud': icp_result.cloud,
            })

        return fue_exitoso, descriptors


class ICPFinderWithModel(ICPFinder):
    def simple_follow(self, object_cloud, target_cloud):
        """
        Tomando como centro el centro del cuadrado que contiene al objeto
        en el frame anterior, busco el mismo objeto en una zona N veces mayor
        a la original.
        """
        # Define row and column limits for the zone to search the object
        # In this case, we look on a box N times the size of the original
        # i.e: if height is 1 and i want a box 2 times bigger and centered
        # on the center of the original box, i have to substract 0.5 times the
        # height to the top of the box and add the same amount to the bottom
        # TODO: ver http://docs.pointclouds.org/1.7.0/crop__box_8h_source.html
        # TODO: ver http://www.pcl-users.org/How-to-use-Crop-Box-td3888183.html

        r_top_limit = self._descriptors['min_y_cloud']
        r_bottom_limit = self._descriptors['max_y_cloud']
        c_left_limit = self._descriptors['min_x_cloud']
        c_right_limit = self._descriptors['max_x_cloud']
        d_front_limit = self._descriptors['min_z_cloud']
        d_back_limit = self._descriptors['max_z_cloud']

        # Define row and column limits for the zone to search the object
        # In this case, we look on a box N times the size of the original
        n = 2

        factor = 0.5 * (n - 1)
        height = r_bottom_limit - r_top_limit
        width = c_right_limit - c_left_limit
        depth = d_back_limit - d_front_limit

        r_top_limit -= height * factor
        r_bottom_limit += height * factor
        c_left_limit -= width * factor
        c_right_limit += width * factor
        d_front_limit -= depth * factor
        d_back_limit += depth * factor

        # Filter points corresponding to the zone where the object being
        # followed is supposed to be
        filter_cloud(
            target_cloud,
            str("y"),
            float(r_top_limit),
            float(r_bottom_limit)
        )
        filter_cloud(
            target_cloud,
            str("x"),
            float(c_left_limit),
            float(c_right_limit)
        )
        filter_cloud(
            target_cloud,
            str("z"),
            float(d_front_limit),
            float(d_back_limit)
        )

        # Calculate ICP
        icp_defaults = ICPDefaults()
        icp_defaults.euc_fit = 1e-15
        icp_defaults.max_corr_dist = 0.3
        icp_defaults.max_iter = 50
        icp_defaults.transf_epsilon = 1e-15
        # icp_defaults.ran_iter
        # icp_defaults.ran_out_rej
        # icp_defaults.show_values = True

        path = b'pruebas_guardadas/detector_con_modelo/'
        nframe = self._descriptors['nframe']
        save_pcd(object_cloud, path + b'object_{n}.pcd'.format(n=nframe))
        save_pcd(target_cloud, path + b'target_{n}.pcd'.format(n=nframe))


        icp_result = self._icp(
            object_cloud,
            target_cloud,
            icp_defaults,
        )

        obj_scene_cloud = filter_object_from_scene_cloud(
            icp_result.cloud,  # object
            target_cloud,  # scene
            0.001,  # radius
            False,  # show values
        )
        icp_result.cloud = obj_scene_cloud

        # save_pcd(icp_result.cloud, path + b'result_{n}.pcd'.format(n=nframe))

        msg = b'score = {s}'
        show_clouds(
            msg.format(s=icp_result.score),
            target_cloud,
            icp_result.cloud,
        )

        return icp_result

    @staticmethod
    def _icp(object_cloud, target_cloud, icp_defaults):
        return icp(object_cloud, target_cloud, icp_defaults)

    def _align_and_icp(self,
                       object_cloud,
                       target_cloud,
                       ap_defaults,
                       icp_defaults):

        # ap_defaults.show_values = True
        aligned_prerejective_result = align(
            object_cloud,
            target_cloud,
            ap_defaults,
        )

        if aligned_prerejective_result.has_converged:
            icp_result = self._icp(
                aligned_prerejective_result.cloud,
                target_cloud,
                icp_defaults,
            )
        else:
            icp_result = ICPResult()
            icp_result.has_converged = False
            icp_result.score = 1e20

        return icp_result

    def calculate_descriptors(self, detected_descriptors):
        detected_descriptors = (super(ICPFinderWithModel, self)
                                .calculate_descriptors(detected_descriptors))
        detected_descriptors['obj_model'] = detected_descriptors['object_cloud']

        minmax = get_min_max(detected_descriptors['object_cloud'])

        detected_descriptors.update({
            'min_x_cloud': minmax.min_x,
            'max_x_cloud': minmax.max_x,
            'min_y_cloud': minmax.min_y,
            'max_y_cloud': minmax.max_y,
            'min_z_cloud': minmax.min_z,
            'max_z_cloud': minmax.max_z,
        })

        return detected_descriptors