#coding=utf-8

from __future__ import (unicode_literals, division)


from cpp.my_pcl import icp, ICPDefaults, filter_cloud, points, get_point, \
    save_pcd
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
    # TODO: se puede hacer una busqueda mejor, en espiral o algo asi
    #       tomando como valor de comparacion el score que devuelve ICP
    @measure_time
    def simple_follow(self,
                      r_top_limit,
                      r_bottom_limit,
                      c_left_limit,
                      c_right_limit,
                      object_cloud,
                      target_cloud):
        """
        Tomando como centro el centro del cuadrado que contiene al objeto
        en el frame anterior, busco el mismo objeto en una zona 4 veces mayor
        a la original.
        """
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
        filter_cloud(target_cloud, str("y"), float(r_top_limit), float(r_bottom_limit))
        filter_cloud(target_cloud, str("x"), float(c_left_limit), float(c_right_limit))

        # Calculate ICP
        icp_defaults = ICPDefaults()
        icp_result = icp(object_cloud, target_cloud, icp_defaults)

        return icp_result

    def find(self):

        # Obtengo pcd's y depth
        object_cloud = self._descriptors['object_cloud']
        target_cloud = self._descriptors['pcd']
        depth_img = self._descriptors['depth_img']

        # Get frame size and location (in RGB image)
        size = self._descriptors['size']
        im_c_left = self._descriptors['location'][1]
        im_c_right = min(im_c_left + size, len(depth_img[0]) - 1)
        im_r_top = self._descriptors['location'][0]
        im_r_bottom = min(im_r_top + size, len(depth_img) - 1)

        topleft = (im_r_top, im_c_left)
        bottomright = (im_r_bottom, im_c_right)

        # Get location in point cloud
        rows_cols_limits = from_flat_to_cloud_limits(topleft,
                                                     bottomright,
                                                     depth_img)
        r_top_limit = rows_cols_limits[0][0]
        r_bottom_limit = rows_cols_limits[0][1]
        c_left_limit = rows_cols_limits[1][0]
        c_right_limit = rows_cols_limits[1][1]

        icp_result = self.simple_follow(
            r_top_limit,
            r_bottom_limit,
            c_left_limit,
            c_right_limit,
            object_cloud,
            target_cloud,
        )

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


class ICPFinderWithModel(ICPFinder):
    def simple_follow(self,
                      r_top_limit,
                      r_bottom_limit,
                      c_left_limit,
                      c_right_limit,
                      object_cloud,
                      target_cloud):
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
        filter_cloud(target_cloud, str("y"), float(r_top_limit), float(r_bottom_limit))
        filter_cloud(target_cloud, str("x"), float(c_left_limit), float(c_right_limit))

        model_cloud = self._descriptors['obj_model']

        nframe = self._descriptors['nframe']
        path = 'pruebas_guardadas/detector_con_modelo/'
        save_pcd(target_cloud, str(path + 'target_{i}.pcd'.format(i=nframe)))

        # Calculate ICP
        icp_defaults = ICPDefaults()
        icp_result = icp(object_cloud, target_cloud, icp_defaults)
        save_pcd(icp_result.cloud, str(path + 'icp_{i}.pcd'.format(i=nframe)))

        #if icp_result.has_converged:
        #    ap_defaults = APDefaults()
        #    ap_defaults.simil_threshold = 0.1
        #    ap_defaults.inlier_fraction = 0.7
        #    #ap_defaults.show_values = True
        #    aligned_prerejective_result = align(model_cloud, icp_result.cloud, ap_defaults)
        #
        #    if aligned_prerejective_result.has_converged:
        #        return aligned_prerejective_result

        return icp_result

    def calculate_descriptors(self, detected_descriptors):
        detected_descriptors = (super(ICPFinderWithModel, self)
                                .calculate_descriptors(detected_descriptors))
        detected_descriptors['obj_model'] = detected_descriptors['object_cloud']
        return detected_descriptors