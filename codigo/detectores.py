#coding=utf-8

from __future__ import (unicode_literals, division)

import scipy.io

from cpp.my_pcl import filter_cloud, icp, ICPDefaults, save_pcd
from cpp.alignment_prerejective import align, APDefaults

from metodos_comunes import from_flat_to_cloud_limits


class Detector(object):
    """
    Es la clase que se encarga de detectar el objeto buscado
    """
    def __init__(self):
        self._descriptors = {}

    def update(self, desc):
        self._descriptors.update(desc)

    def calculate_descriptors(self, desc):
        """
        Calcula los descriptores en base al objeto encontrado para que
        los almacene el Follower
        """
        return desc

    def detect(self):
        pass


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

        # Comienza contando desde 0 por eso hago nframe - 1
        objs = self._matfile[0][nframe - 1][0]

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
            'location': location,  # location=(fila, columna)
        }

        return fue_exitoso, detected_descriptors


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

        detected_descriptors.update(
            {
                'object_cloud': cloud,
                'min_x_cloud': c_left_limit,
                'max_x_cloud': c_right_limit,
                'min_y_cloud': r_top_limit,
                'max_y_cloud': r_bottom_limit,
            }
        )

        return detected_descriptors


class StaticDetectorWithModelAlignment(StaticDetectorWithPCDFiltering):
    def calculate_descriptors(self, detected_descriptors):
        """
        Obtengo la nube de puntos correspondiente a la ubicacion y region
        pasadas por parametro.
        """
        detected_descriptors = (
            super(StaticDetectorWithModelAlignment, self)
            .calculate_descriptors(detected_descriptors)
        )

        model_cloud = self._descriptors['obj_model']
        detected_cloud = detected_descriptors['object_cloud']


        # Calculate alignment prerejective
        ap_defaults = APDefaults()
        ap_defaults.max_ransac_iters = 10000
        ap_defaults.nearest_features_used = 3
        ap_defaults.simil_threshold = 0.1
        ap_defaults.inlier_threshold = 1.5
        ap_defaults.inlier_fraction = 0.7
        #ap_defaults.show_values = True
        ap_result = align(model_cloud, detected_cloud, ap_defaults)

        if ap_result.has_converged:
            # Calculate ICP
            icp_defaults = ICPDefaults()
            icp_defaults.euc_fit = 1e-15
            icp_defaults.max_corr_dist = 3
            icp_defaults.max_iter = 50
            icp_defaults.transf_epsilon = 1e-15
            # icp_defaults.show_values = True
            icp_result = icp(ap_result.cloud, detected_cloud, icp_defaults)

            if icp_result.has_converged:
                detected_descriptors['object_cloud'] = icp_result.cloud
                detected_descriptors['obj_model'] = icp_result.cloud

        return detected_descriptors