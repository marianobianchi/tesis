#coding=utf-8

from __future__ import (unicode_literals, division)

import scipy.io

from cpp.my_pcl import filter_cloud, icp, ICPDefaults, save_pcd, get_min_max,\
    show_clouds, filter_object_from_scene_cloud
from cpp.alignment_prerejective import align, APDefaults

from metodos_comunes import from_flat_to_cloud_limits
from metodos_de_busqueda import BusquedaPorFramesSolapados


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

        filas = len(depth_img)
        columnas = len(depth_img[0])

        ubicacion_punto_diagonal = (
            min(ubicacion[0] + tam_region, filas - 1),
            min(ubicacion[1] + tam_region, columnas - 1)
        )

        rows_cols_limits = from_flat_to_cloud_limits(
            ubicacion,
            ubicacion_punto_diagonal,
            depth_img,
        )

        r_top_limit = rows_cols_limits[0][0]
        r_bottom_limit = rows_cols_limits[0][1]
        c_left_limit = rows_cols_limits[1][0]
        c_right_limit = rows_cols_limits[1][1]

        cloud = self._descriptors['pcd']

        cloud = filter_cloud(cloud, str("y"), float(r_top_limit), float(r_bottom_limit))
        cloud = filter_cloud(cloud, str("x"), float(c_left_limit), float(c_right_limit))

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
        ap_defaults.leaf = 0.004
        ap_defaults.max_ransac_iters = 1000
        ap_defaults.points_to_sample = 5
        ap_defaults.nearest_features_used = 3
        ap_defaults.simil_threshold = 0.1
        ap_defaults.inlier_threshold = 1.5
        ap_defaults.inlier_fraction = 0.7
        #ap_defaults.show_values = True

        ap_result = align(model_cloud, detected_cloud, ap_defaults)

        # show_clouds(b"alineacion en zona de deteccion", detected_cloud, ap_result.cloud)

        if ap_result.has_converged:
            # Calculate ICP
            icp_defaults = ICPDefaults()
            icp_defaults.euc_fit = 1e-15
            icp_defaults.max_corr_dist = 3
            icp_defaults.max_iter = 50
            icp_defaults.transf_epsilon = 1e-15
            # icp_defaults.show_values = True
            icp_result = icp(ap_result.cloud, detected_cloud, icp_defaults)

            # show_clouds(b"icp de alineacion en zona de deteccion", detected_cloud, icp_result.cloud)

            if icp_result.has_converged:
                # Filtro los puntos de la escena que se corresponden con el
                # objeto que estoy buscando
                obj_scene_cloud = filter_object_from_scene_cloud(
                    icp_result.cloud,  # object
                    detected_cloud,  # scene
                    0.001,  # radius
                    False,  # show values
                )

                # show_clouds(b"kdtree en deteccion", detected_cloud, obj_scene_cloud)

                minmax = get_min_max(obj_scene_cloud)
                detected_descriptors.update({
                    'min_z_cloud': minmax.min_z,
                    'max_z_cloud': minmax.max_z,
                    'object_cloud': obj_scene_cloud,
                    'obj_model': obj_scene_cloud,
                })

        return detected_descriptors


class AutomaticDetection(Detector):
    def __init__(self, obj_rgbd_name):
        super(AutomaticDetection, self).__init__()
        self.obj_rgbd_name = obj_rgbd_name

    def detect(self):
        model_cloud = self._descriptors['obj_model']
        scene_cloud = self._descriptors['pcd']

        # obtengo tamaño del modelo del objeto a detectar y lo duplico
        min_max = get_min_max(model_cloud)
        obj_width = (min_max.max_x - min_max.min_x) * 2
        obj_height = (min_max.max_y - min_max.min_y) * 2

        # obtengo limites de la escena
        min_max = get_min_max(scene_cloud)
        scene_min_col = min_max.min_x
        scene_max_col = min_max.max_x
        scene_min_row = min_max.min_y
        scene_max_row = min_max.max_y

        detected_descriptors = {
            'size': 0,
            'location': (0, 0),  # location=(fila, columna)
        }

        # alignment prerejective parameters
        ap_defaults = APDefaults()
        ap_defaults.leaf = 0.004
        ap_defaults.max_ransac_iters = 1500
        ap_defaults.points_to_sample = 5
        ap_defaults.nearest_features_used = 4
        ap_defaults.simil_threshold = 0.3
        ap_defaults.inlier_threshold = 2
        ap_defaults.inlier_fraction = 0.7
        ap_defaults.show_values = True

        #icp parameters
        icp_defaults = ICPDefaults()
        icp_defaults.euc_fit = 1e-15
        icp_defaults.max_corr_dist = 3
        icp_defaults.max_iter = 50
        icp_defaults.transf_epsilon = 1e-15
        # icp_defaults.show_values = True

        best_aligned_scene = None
        best_alignment_score = 0.1  # lesser is better
        best_icp_aligned_scene = None
        best_icp_score = 0.1  # lesser is better

        for limits in (BusquedaPorFramesSolapados()
                       .iterate_frame_boxes(scene_min_col,
                                            scene_max_col,
                                            scene_min_row,
                                            scene_max_row,
                                            obj_width,
                                            obj_height)):
            cloud = filter_cloud(pcd, b'x', limits['min_x'], limits['max_x'])
            cloud = filter_cloud(cloud, b'y', limits['min_y'], limits['max_y'])

            # Calculate alignment
            ap_result = align(model_cloud, cloud, ap_defaults)
            if (ap_result.has_converged and
                        ap_result.score < best_alignment_score):
                best_alignment_score = ap_result.score
                best_aligned_scene = ap_result.cloud

            if best_aligned_scene is not None:
                # Calculate ICP
                icp_result = icp(best_aligned_scene, cloud, icp_defaults)

                if (icp_result.has_converged and
                            icp_result.score < best_icp_score):
                    best_icp_score = icp_result.score
                    best_icp_aligned_scene = icp_result.cloud


        if best_icp_aligned_scene is not None:
            # Filtro los puntos de la escena que se corresponden con el
            # objeto que estoy buscando
            obj_scene_cloud = filter_object_from_scene_cloud(
                best_icp_aligned_scene,  # object
                cloud,  # scene
                0.001,  # radius
                False,  # show values
            )

            minmax = get_min_max(obj_scene_cloud)
            detected_descriptors.update({
                'min_x_cloud': minax.min_x,
                'max_x_cloud': minax.max_x,
                'min_y_cloud': minax.min_y,
                'max_y_cloud': minax.max_y,
                'min_z_cloud': minmax.min_z,
                'max_z_cloud': minmax.max_z,
                'object_cloud': obj_scene_cloud,
                'obj_model': obj_scene_cloud,
                # TODO: calcular tamaño y posicion
                'size': 0,
                'location': (0, 0),  # location=(fila, columna)
            })







        # # show_clouds(b"alineacion en zona de deteccion", detected_cloud, ap_result.cloud)
        # path = b'pruebas_guardadas/deteccion_automatica/'
        # nframe = self._descriptors['nframe']
        # save_pcd(model_cloud, path + b'model_cloud_{i}.pcd'.format(i=nframe))
        # save_pcd(scene_cloud, path + b'scene_cloud_{i}.pcd'.format(i=nframe))
        #
        # if ap_result.has_converged:
        #
        #     save_pcd(
        #         ap_result.cloud,
        #         path + b'aligned_cloud_{i}.pcd'.format(i=nframe)
        #     )
        #
        #     # Calculate ICP
        #     icp_result = icp(ap_result.cloud, scene_cloud, icp_defaults)
        #
        #     # show_clouds(b"icp de alineacion en zona de deteccion", detected_cloud, icp_result.cloud)
        #
        #     if icp_result.has_converged:
        #         save_pcd(
        #             icp_result.cloud,
        #             path + b'icp_cloud_{i}.pcd'.format(i=nframe)
        #         )
        #
        #         # Filtro los puntos de la escena que se corresponden con el
        #         # objeto que estoy buscando
        #         obj_scene_cloud = filter_object_from_scene_cloud(
        #             icp_result.cloud,  # object
        #             scene_cloud,  # scene
        #             0.001,  # radius
        #             False,  # show values
        #         )
        #
        #         # show_clouds(b"kdtree en deteccion", detected_cloud, obj_scene_cloud)
        #
        #         minmax = get_min_max(obj_scene_cloud)
        #         detected_descriptors.update({
        #             'min_z_cloud': minmax.min_z,
        #             'max_z_cloud': minmax.max_z,
        #             'object_cloud': obj_scene_cloud,
        #             'obj_model': obj_scene_cloud,
        #         })

        return detected_descriptors