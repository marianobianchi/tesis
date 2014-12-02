#coding=utf-8

from __future__ import (unicode_literals, division)

import cv2

from cpp.icp import icp, ICPDefaults, ICPResult
from cpp.common import filter_cloud, points, get_min_max, transform_cloud, \
    filter_object_from_scene_cloud, show_clouds

from metodos_comunes import from_cloud_to_flat_limits, AdaptSearchArea, \
    AdaptLeafRatio
from metodos_de_busqueda import BusquedaEnEspiral
from observar_seguimiento import MuestraBusquedaEnVivo


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

    def __init__(self, icp_defaults=None, umbral_score=1e-4):
        super(ICPFinder, self).__init__()
        if icp_defaults is None:
            icp_defaults = ICPDefaults()
            icp_defaults.euc_fit = 1e-15
            icp_defaults.max_corr_dist = 0.3
            icp_defaults.max_iter = 50
            icp_defaults.transf_epsilon = 1e-15

        self._icp_defaults = icp_defaults
        self.umbral_score = umbral_score

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
        target_cloud = filter_cloud(
            target_cloud,
            str("y"),
            float(r_top_limit),
            float(r_bottom_limit)
        )
        target_cloud = filter_cloud(
            target_cloud,
            str("x"),
            float(c_left_limit),
            float(c_right_limit)
        )

        # Calculate ICP
        icp_result = icp(object_cloud, target_cloud, self._icp_defaults)

        return icp_result

    def find(self):
        # Obtengo pcd's y depth
        object_cloud = self._descriptors['object_cloud']
        target_cloud = self._descriptors['pcd']

        icp_result = self.simple_follow(
            object_cloud,
            target_cloud,
        )

        fue_exitoso = icp_result.score < self.umbral_score
        descriptors = {}

        if fue_exitoso:
            # filas = len(depth_img)
            # columnas = len(depth_img[0])

            # Busco los limites en el dominio de las filas y columnas del RGB
            topleft, bottomright = from_cloud_to_flat_limits(
                icp_result.cloud
            )

            descriptors.update({
                'topleft': topleft,
                'bottomright': bottomright,
                'detected_cloud': icp_result.cloud,
                'detected_transformation': icp_result.transformation,
            })

        return fue_exitoso, descriptors


class ICPFinderWithModel(ICPFinder):

    def __init__(self, icp_defaults=None, umbral_score=1e-4, **kwargs):
        """
        valid kwargs:
        obj_scene_leaf (default: 0.002)
        perc_obj_model_points (default: 0.5)
        """
        super(ICPFinderWithModel, self).__init__(icp_defaults, umbral_score)

        # Seteo el tamaño de las esferas usadas para filtrar de la escena
        # los puntos del objeto encontrado
        self.adapt_leaf = kwargs.get('adapt_leaf', AdaptLeafRatio())
        self.first_leaf_size = kwargs.get('first_leaf_size', 0.002)

        # Seteo el porcentaje de puntos que permito conservar del modelo del
        # objeto antes de considerar que lo que se encontró no es el objeto
        self.perc_obj_model_points = kwargs.get('perc_obj_model_points', 0.5)

        # Agrego un objeto que adapta la zona de busqueda segun la velocidad
        # del objeto que estoy buscando
        self.adapt_area = kwargs.get('adapt_area', AdaptSearchArea())

    def get_object_points_from_scene(self, found_obj, scene):
        filtered_scene = self._filter_target_cloud(scene)
        return filter_object_from_scene_cloud(
            found_obj,
            filtered_scene,
            self.adapt_leaf.leaf_ratio(),
            False,
        )

    def _filter_target_cloud(self, target_cloud):
        # TODO: ver http://docs.pointclouds.org/1.7.0/crop__box_8h_source.html
        # TODO: ver http://www.pcl-users.org/How-to-use-Crop-Box-td3888183.html

        r_top_limit = self._descriptors['min_y_cloud']
        r_bottom_limit = self._descriptors['max_y_cloud']
        c_left_limit = self._descriptors['min_x_cloud']
        c_right_limit = self._descriptors['max_x_cloud']
        d_front_limit = self._descriptors['min_z_cloud']
        d_back_limit = self._descriptors['max_z_cloud']

        # Define row and column limits for the zone to search the object
        x_move = self.adapt_area.estimate_distance('x')
        y_move = self.adapt_area.estimate_distance('y')
        z_move = self.adapt_area.estimate_distance('z')

        if x_move < 0 or y_move < 0 or z_move < 0:
            raise Exception(('Ojo que esta mal el area de busqueda. '
                             'x,y,z={x},{y},{z}'.format(x=x_move, y=y_move,
                                                        z=z_move)))

        r_top_limit -= y_move
        r_bottom_limit += y_move
        c_left_limit -= x_move
        c_right_limit += x_move
        d_front_limit -= z_move
        d_back_limit += z_move

        # Filter points corresponding to the zone where the object being
        # followed is supposed to be
        target_cloud = filter_cloud(
            target_cloud,
            str("y"),
            float(r_top_limit),
            float(r_bottom_limit)
        )
        target_cloud = filter_cloud(
            target_cloud,
            str("x"),
            float(c_left_limit),
            float(c_right_limit)
        )
        target_cloud = filter_cloud(
            target_cloud,
            str("z"),
            float(d_front_limit),
            float(d_back_limit)
        )
        return target_cloud

    def find(self):
        # Obtengo pcd's y depth
        object_cloud = self._descriptors['object_cloud']
        target_cloud = self._descriptors['pcd']

        obj_model = self._descriptors['obj_model']
        model_points = points(obj_model)
        self.adapt_area.set_default_distances(obj_model)
        if not self.adapt_leaf.was_started():
            self.adapt_leaf.set_first_values(model_points)

        accepted_points = model_points * self.perc_obj_model_points

        icp_result = self.simple_follow(
            object_cloud,
            target_cloud,
        )

        points_from_scene = 0
        if icp_result.has_converged:
            obj_from_scene_points = self.get_object_points_from_scene(
                icp_result.cloud,
                target_cloud,
            )
            points_from_scene = points(obj_from_scene_points)

        fue_exitoso = icp_result.score < self.umbral_score
        fue_exitoso = (
            fue_exitoso and
            points_from_scene >= accepted_points
        )

        descriptors = {}

        if fue_exitoso:
            self.adapt_leaf.set_found_points(points_from_scene)

            # filas = len(depth_img)
            # columnas = len(depth_img[0])

            # Busco los limites en el dominio de las filas y columnas del RGB
            topleft, bottomright = from_cloud_to_flat_limits(
                obj_from_scene_points
            )

            descriptors.update({
                'topleft': topleft,
                'bottomright': bottomright,
                'detected_cloud': obj_from_scene_points,
                'detected_transformation': icp_result.transformation,
            })
        else:
            self.adapt_leaf.reset()

        return fue_exitoso, descriptors

    def simple_follow(self, object_cloud, target_cloud):
        """
        Tomando como centro el centro del cuadrado que contiene al objeto
        en el frame anterior, busco el mismo objeto en una zona N veces mayor
        a la original.
        """
        target_cloud = self._filter_target_cloud(target_cloud)

        if points(target_cloud) > 0:
            # Calculate ICP
            icp_result = icp(object_cloud, target_cloud, self._icp_defaults)
        else:
            icp_result = ICPResult()
            icp_result.has_converged = False
            icp_result.score = 100

        return icp_result

    def calculate_descriptors(self, detected_descriptors):
        detected_descriptors = (super(ICPFinderWithModel, self)
                                .calculate_descriptors(detected_descriptors))

        transformation = detected_descriptors['detected_transformation']
        old_obj_model = self._descriptors['obj_model']
        new_obj_model = transform_cloud(old_obj_model, transformation)

        #################################################
        # show_clouds(
        #     b'Modelo transformado vs objeto de la escena',
        #     new_obj_model,
        #     obj_scene_cloud
        # )
        # show_clouds(
        #     b'Modelo transformado vs escena',
        #     new_obj_model,
        #     target_cloud
        # )
        #################################################

        detected_descriptors['obj_model'] = new_obj_model

        minmax = get_min_max(detected_descriptors['object_cloud'])

        self.adapt_area.save_centroid(
            minmax.min_x,
            minmax.min_y,
            minmax.min_z,
            minmax.max_x,
            minmax.max_y,
            minmax.max_z,
        )

        detected_descriptors.update({
            'min_x_cloud': minmax.min_x,
            'max_x_cloud': minmax.max_x,
            'min_y_cloud': minmax.min_y,
            'max_y_cloud': minmax.max_y,
            'min_z_cloud': minmax.min_z,
            'max_z_cloud': minmax.max_z,
        })

        return detected_descriptors


#############
# RGB Finder
#############
class TemplateAndFrameHistogramFinder(Finder):
    def __init__(self, template_comp_method=cv2.cv.CV_COMP_BHATTACHARYYA,
                 frame_comp_method=cv2.cv.CV_COMP_BHATTACHARYYA):
        super(TemplateAndFrameHistogramFinder, self).__init__()
        self.metodo_de_busqueda = BusquedaEnEspiral()
        self.template_comp_method = template_comp_method
        self.frame_comp_method = frame_comp_method

    @staticmethod
    def calculate_rgb_histogram(roi, mask=None):
        # Paso la imagen de BGR a RGB
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

        # Calculo el histograma del roi (para H y S)
        hist = cv2.calcHist(
            [roi_rgb],  # Imagen
            [0, 1, 2],  # Canales
            mask,  # Mascara
            [8, 8, 8],  # Numero de bins para cada canal
            [0, 256, 0, 256, 0, 256],  # Rangos válidos para cada canal
        )

        # Normalizo el histograma para evitar errores por distinta escala
        hist = cv2.normalize(hist).flatten()

        return hist

    @staticmethod
    def calculate_hsv_histogram(roi, mask=None):
        # Paso la imagen de BGR a RGB
        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Calculo el histograma del roi (para H y S)
        hist = cv2.calcHist(
            [roi_hsv],  # Imagen
            [1, 2],  # Canales
            mask,  # Mascara
            [8, 16],  # Numero de bins para cada canal
            [0, 256, 0, 256],  # Rangos válidos para cada canal
        )

        # Normalizo el histograma para evitar errores por distinta escala
        hist = cv2.normalize(hist).flatten()

        return hist

    def saved_object_comparisson(self):
        if 'object_frame_hsv_hist' not in self._descriptors:
            obj = self._descriptors['object_frame']
            hist = self.calculate_hsv_histogram(obj)
            self._descriptors['object_frame_hsv_hist'] = hist

        if 'object_template_rgb_hist' not in self._descriptors:
            obj = self._descriptors['object_template']
            mask = self._descriptors['object_mask']
            hist = self.calculate_rgb_histogram(obj, mask)
            self._descriptors['object_template_rgb_hist'] = hist

        return (self._descriptors['object_template_rgb_hist'],
                self._descriptors['object_frame_hsv_hist'])

    @staticmethod
    def object_comparisson_base(img):
        return {
            'object_template_comp': 0.6,
            'object_frame_comp': 0.4,
        }

    def object_comparisson(self, roi):
        roi_rgb_hist = self.calculate_rgb_histogram(roi)
        roi_hsv_hist = self.calculate_hsv_histogram(roi)

        # Tomo el histograma del objeto para comparar
        obj_template_rgb_hist, object_frame_hsv_hist = (
            self.saved_object_comparisson()
        )

        template_comp = cv2.compareHist(
            obj_template_rgb_hist,
            roi_rgb_hist,
            self.template_comp_method,
        )
        frame_comp = cv2.compareHist(
            object_frame_hsv_hist,
            roi_hsv_hist,
            self.frame_comp_method,
        )

        return {
            'object_template_comp': template_comp,
            'object_frame_comp': frame_comp,
        }

    def is_best_match(self, new_value, old_value):
        templ_better = new_value['object_template_comp'] < old_value['object_template_comp']
        obj_better = new_value['object_frame_comp'] < old_value['object_frame_comp']
        return templ_better and obj_better


    def simple_follow(self, img, topleft, bottomright, valor_comparativo):
        """
        Esta funcion es el esquema de seguimiento del objeto.
        """
        filas, columnas = len(img), len(img[0])

        new_topleft = topleft
        new_bottomright = bottomright

        # Seguimiento (busqueda/deteccion acotada)
        generador_de_ubicaciones = (
            self.metodo_de_busqueda.get_positions_and_framesizes(
                topleft,
                bottomright,
                filas,
                columnas
            )
        )

        for explored_topleft, explored_bottomright in generador_de_ubicaciones:
            col_izq = explored_topleft[1]
            col_der = explored_bottomright[1]
            fil_arr = explored_topleft[0]
            fil_aba = explored_bottomright[0]

            # Tomo una region de la imagen donde se busca el objeto
            roi = img[fil_arr:fil_aba, col_izq:col_der]

            # Comparo
            nueva_comparacion = self.object_comparisson(roi)

            # Si se quiere ver como va buscando, descomentar la siguiente linea
            # MuestraBusquedaEnVivo('Buscando el objeto').run(
            #     img,
            #     (fil_arr, col_izq),
            #     (fil_aba, col_der),
            #     frenar=True,
            # )

            # Si hubo coincidencia
            if self.is_best_match(nueva_comparacion, valor_comparativo):
                # Nueva ubicacion del objeto (esquina superior izquierda del
                # cuadrado)
                new_topleft = explored_topleft

                # Esq. inferior derecha
                new_bottomright = explored_bottomright

                # Actualizo el valor de la comparacion
                valor_comparativo = nueva_comparacion

        return new_topleft, new_bottomright, valor_comparativo

    def find(self):
        img = self._descriptors['scene_rgb']

        topleft = self._descriptors['topleft']
        new_topleft = topleft

        bottomright = self._descriptors['bottomright']
        new_bottomright = bottomright

        valor_comparativo_base = self.object_comparisson_base(img)
        valor_comparativo = valor_comparativo_base

        # Repito 3 veces (cantidad arbitraria) una busqueda, partiendo siempre
        # de la ultima mejor ubicacion del objeto encontrada
        for i in range(3):
            new_topleft, new_bottomright, new_valor_comparativo = self.simple_follow(
                img,
                new_topleft,
                new_bottomright,
                valor_comparativo,
            )
            # Si no cambio de posicion, no sigo buscando
            if not self.is_best_match(new_valor_comparativo, valor_comparativo):
                break

            valor_comparativo = new_valor_comparativo

        fue_exitoso = self.is_best_match(
            valor_comparativo,
            valor_comparativo_base,
        )
        desc = {}

        if fue_exitoso:
            desc = {
                'topleft': new_topleft,
                'bottomright': new_bottomright,
            }
        else:
            self._descriptors = {}

        return fue_exitoso, desc

    def calculate_descriptors(self, desc):
        img = self._descriptors['scene_rgb']
        topleft = desc['topleft']
        bottomright = desc['bottomright']

        frame = img[topleft[0]:bottomright[0],
                    topleft[1]:bottomright[1]]

        # Actualizo el histograma
        hist = self.calculate_hsv_histogram(frame)

        desc.update(
            {
                'object_frame': frame,
                'object_frame_hsv_hist': hist,
            }
        )
        return desc