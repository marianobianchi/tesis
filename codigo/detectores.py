#coding=utf-8

from __future__ import (unicode_literals, division)

import cv2
import scipy.io

from cpp.icp import icp, ICPDefaults
from cpp.common import filter_cloud, save_pcd, get_min_max, show_clouds, \
    filter_object_from_scene_cloud, points
from cpp.alignment_prerejective import align, APDefaults

from metodos_comunes import from_flat_to_cloud_limits, \
    from_cloud_to_flat_limits, AdaptLeafRatio
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
        bottomright = (0, 0)

        for obj in objs:
            if obj[0][0] == self._obj_rgbd_name:
                fue_exitoso = True
                location = (int(obj[2][0][0]), int(obj[4][0][0]))
                bottomright = (int(obj[3][0][0]), int(obj[5][0][0]))
                tam_region = max(int(obj[3][0][0]) - int(obj[2][0][0]),
                                 int(obj[5][0][0]) - int(obj[4][0][0]))
                break

        detected_descriptors = {
            'size': tam_region,
            'location': location,  # location=(fila, columna)
            'topleft': location,
            'bottomright': bottomright,
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

        cloud = filter_cloud(
            cloud,
            str("y"),
            float(r_top_limit),
            float(r_bottom_limit)
        )
        cloud = filter_cloud(
            cloud,
            str("x"),
            float(c_left_limit),
            float(c_right_limit)
        )

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

        # show_clouds(
        #     b"alineacion en zona de deteccion",
        #     detected_cloud,
        #     ap_result.cloud
        # )

        if ap_result.has_converged:
            # Calculate ICP
            icp_defaults = ICPDefaults()
            icp_defaults.euc_fit = 1e-15
            icp_defaults.max_corr_dist = 3
            icp_defaults.max_iter = 50
            icp_defaults.transf_epsilon = 1e-15
            # icp_defaults.show_values = True
            icp_result = icp(ap_result.cloud, detected_cloud, icp_defaults)

            # show_clouds(
            #     b"icp de alineacion en zona de deteccion",
            #     detected_cloud,
            #     icp_result.cloud
            # )

            if icp_result.has_converged:
                # Filtro los puntos de la escena que se corresponden con el
                # objeto que estoy buscando
                obj_scene_cloud = filter_object_from_scene_cloud(
                    icp_result.cloud,  # object
                    detected_cloud,  # scene
                    0.001,  # radius
                    False,  # show values
                )

                # show_clouds(
                #     b"kdtree en deteccion",
                #     detected_cloud,
                #     obj_scene_cloud
                # )

                minmax = get_min_max(obj_scene_cloud)
                detected_descriptors.update({
                    'min_z_cloud': minmax.min_z,
                    'max_z_cloud': minmax.max_z,
                    'object_cloud': obj_scene_cloud,
                    'obj_model': obj_scene_cloud,
                })

        return detected_descriptors


class AutomaticDetection(Detector):

    def __init__(self, ap_defaults=None, icp_defaults=None,
                 umbral_score=1e-3, **kwargs):
        super(AutomaticDetection, self).__init__()
        self.umbral_score = umbral_score
        if ap_defaults is None:
            # alignment prerejective parameters
            ap_defaults = APDefaults()
            ap_defaults.leaf = 0.005
            ap_defaults.max_ransac_iters = 100
            ap_defaults.points_to_sample = 3
            ap_defaults.nearest_features_used = 4
            ap_defaults.simil_threshold = 0.1
            ap_defaults.inlier_threshold = 3
            ap_defaults.inlier_fraction = 0.8
            # ap_defaults.show_values = True

        self._ap_defaults = ap_defaults

        if icp_defaults is None:
            # icp parameters
            icp_defaults = ICPDefaults()
            icp_defaults.euc_fit = 1e-5
            icp_defaults.max_corr_dist = 3
            icp_defaults.max_iter = 50
            icp_defaults.transf_epsilon = 1e-5
            # icp_defaults.show_values = True

        self._icp_defaults = icp_defaults

        # Seteo el tamaño de las esferas usadas para filtrar de la escena
        # los puntos del objeto encontrado
        self.adapt_leaf = kwargs.get('adapt_leaf', AdaptLeafRatio())
        self.first_leaf_size = kwargs.get('first_leaf_size', 0.005)

        # Seteo el porcentaje de puntos que permito conservar del modelo del
        # objeto antes de considerar que lo que se encontró no es el objeto
        self.perc_obj_model_points = kwargs.get('perc_obj_model_points', 0.5)

        # Seteo el tamaño del frame de busqueda. Este valor se va a multiplicar
        # por la altura y el ancho del objeto. Ej: si se multiplica por 2, el
        # frame de busqueda tiene un area 4 (2*2) veces mayor que la del objeto
        self.obj_mult = kwargs.get('obj_mult', 2)

    def detect(self):
        model_cloud = self._descriptors['obj_model']
        model_cloud_points = points(model_cloud)

        accepted_points = model_cloud_points * self.perc_obj_model_points

        if not self.adapt_leaf.was_started():
            self.adapt_leaf.set_first_values(model_cloud_points)

        scene_cloud = self._descriptors['pcd']

        # obtengo tamaño del modelo del objeto a detectar y tomo una region
        # X veces mas grande
        obj_limits = get_min_max(model_cloud)

        # obtengo limites de la escena
        scene_limits = get_min_max(scene_cloud)

        detected_descriptors = {
            'topleft': (0, 0),  # (fila, columna)
            'bottomright': 0,
        }
        fue_exitoso = False

        best_aligned_scene = None
        best_alignment_score = self.umbral_score  # lesser is better
        best_limits = {}

        # Busco la mejor alineacion del objeto segmentando la escena
        for limits in (BusquedaPorFramesSolapados()
                       .iterate_frame_boxes(obj_limits, scene_limits,
                                            obj_mult=self.obj_mult)):

            cloud = filter_cloud(
                scene_cloud,
                b'x',
                limits['min_x'],
                limits['max_x']
            )
            cloud = filter_cloud(
                cloud,
                b'y',
                limits['min_y'],
                limits['max_y']
            )

            if points(cloud) > model_cloud_points:
                # Calculate alignment
                ap_result = align(model_cloud, cloud, self._ap_defaults)
                if (ap_result.has_converged and
                        ap_result.score < best_alignment_score):
                    best_alignment_score = ap_result.score
                    best_aligned_scene = ap_result.cloud
                    best_limits.update(limits)

        # Su hubo una buena alineacion
        if best_aligned_scene is not None:
            cloud = filter_cloud(
                scene_cloud,
                b'x',
                best_limits['min_x'],
                best_limits['max_x']
            )
            cloud = filter_cloud(
                cloud,
                b'y',
                best_limits['min_y'],
                best_limits['max_y']
            )

            # Calculate ICP
            icp_result = icp(best_aligned_scene, cloud, self._icp_defaults)

            if icp_result.has_converged and icp_result.score < self.umbral_score:
                # Filtro los puntos de la escena que se corresponden con el
                # objeto que estoy buscando
                obj_scene_cloud = filter_object_from_scene_cloud(
                    icp_result.cloud,  # object
                    scene_cloud,  # complete scene
                    self.adapt_leaf.leaf_ratio(),  # radius
                    False,  # show values
                )

                obj_scene_points = points(obj_scene_cloud)

                fue_exitoso = obj_scene_points > accepted_points

                if fue_exitoso:
                    self.adapt_leaf.set_found_points(obj_scene_points)
                else:
                    self.adapt_leaf.reset()

                minmax = get_min_max(obj_scene_cloud)

                topleft, bottomright = from_cloud_to_flat_limits(
                    obj_scene_cloud
                )

                detected_descriptors.update({
                    'min_x_cloud': minmax.min_x,
                    'max_x_cloud': minmax.max_x,
                    'min_y_cloud': minmax.min_y,
                    'max_y_cloud': minmax.max_y,
                    'min_z_cloud': minmax.min_z,
                    'max_z_cloud': minmax.max_z,
                    'object_cloud': obj_scene_cloud,
                    'obj_model': icp_result.cloud,  # original model transformed
                    'detected_cloud': icp_result.cloud,  # lo guardo solo para la estadistica
                    'topleft': topleft,  # (fila, columna)
                    'bottomright': bottomright,
                })

                # show_clouds(
                #   b'Modelo detectado vs escena',
                #   icp_result.cloud,
                #   scene_cloud
                # )

        return fue_exitoso, detected_descriptors


#################
# Detectores RGB
#################

class RGBTemplateDetector(Detector):
    def detect(self):
        img = self._descriptors['scene_rgb']
        template = self._descriptors['obj_rgb_template']
        template_filas, template_columnas = len(template), len(template[0])

        # Leer: http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_template_matching/py_template_matching.html#theory
        # Aplico el template Matching
        res = cv2.matchTemplate(img, template, cv2.TM_SQDIFF_NORMED)

        # Busco la posición
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        desc = {}
        # TODO: revisar este umbral
        fue_exitoso = min_val < 0.16

        if fue_exitoso:
            # min_loc y max_loc tienen primero las columnas y despues las filas,
            # entonces lo doy vuelta
            topleft = (min_loc[1], min_loc[0])
            bottomright = (min_loc[1] + template_filas, min_loc[0] + template_columnas)

            desc = {
                'topleft': topleft,
                'bottomright': bottomright,
                'object_frame': img[topleft[0]:bottomright[0],
                                    topleft[1]:bottomright[1]],
            }

        return fue_exitoso, desc
