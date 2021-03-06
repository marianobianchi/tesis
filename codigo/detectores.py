# coding=utf-8

from __future__ import (unicode_literals, division)

import os
import cv2
import scipy.io

from cpp.icp import icp, ICPDefaults
from cpp.common import filter_cloud, save_pcd, get_min_max, show_clouds, \
    filter_object_from_scene_cloud, points, VectorMat, FloatVector, \
    transform_cloud
from cpp.alignment_prerejective import align, APDefaults

from metodos_comunes import from_flat_to_cloud_limits, \
    from_cloud_to_flat_limits, AdaptLeafRatio
from metodos_de_busqueda import BusquedaPorFramesSolapados
from observar_seguimiento import MuestraBusquedaEnVivo


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
    def __init__(self, matfile_path, obj_rgbd_name, obj_rgbd_num):
        super(StaticDetector, self).__init__()
        self._matfile = scipy.io.loadmat(matfile_path)['bboxes']
        self._obj_rgbd_name = obj_rgbd_name
        self._obj_rgbd_num = obj_rgbd_num

    def detect(self):
        nframe = self._descriptors['nframe']

        # Comienza contando desde 0 por eso hago nframe - 1
        objs = self._matfile[0][nframe - 1][0]

        fue_exitoso = False
        tam_region = 0
        topleft = (0, 0)
        bottomright = (0, 0)

        for obj in objs:
            if (obj[0][0] == self._obj_rgbd_name and
                    unicode(obj[1][0][0]) == self._obj_rgbd_num):
                fue_exitoso = True
                topleft = (int(obj[2][0][0]) - 1, int(obj[4][0][0]) - 1)
                bottomright = (int(obj[3][0][0]) - 1, int(obj[5][0][0]) - 1)
                tam_region = max(bottomright[0] - topleft[0],
                                 bottomright[1] - topleft[1])
                break

        if 'scene_rgb' in self._descriptors:
            img = self._descriptors['scene_rgb']
        elif 'depth_img' in self._descriptors:
            img = self._descriptors['depth_img']
        else:
            img = [[i for i in range(640)]] + [[i for i in range(479)]]
        ult_fila = len(img) - 1
        ult_columna = len(img[0]) - 1

        is_on_top_or_left = topleft[0] == 0 or topleft[1] == 0
        is_on_bottom_or_right = (
            bottomright[0] == ult_fila or bottomright[1] == ult_columna
        )

        detected_descriptors = {
            'size': tam_region,
            'location': topleft,  # topleft=(fila, columna)
            'topleft': topleft,
            'bottomright': bottomright,
        }

        # Solo doy por deteccion a aquellos cuadrantes del ground truth que no
        # esten pegados a uno de los bordes de la imagen
        if is_on_top_or_left or is_on_bottom_or_right:
            fue_exitoso = False

        return fue_exitoso, detected_descriptors


class DepthStaticDetectorWithPCDFiltering(StaticDetector):
    def detect(self):
        fue_exitoso, detected_descriptors = (
            super(DepthStaticDetectorWithPCDFiltering, self).detect()
        )

        # Si lo que se detecto en RGB no posee la cantidad de puntos minimos
        # necesarios en depth, se considera no detectado aunque la BD diga lo
        # contrario
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

        accepted_points = (
            self._descriptors['obj_model_points'] * self.perc_obj_model_pts
        )
        if points(cloud) < accepted_points:
            detected_descriptors = {
                'size': 0,
                'location': (0, 0),  # location=(fila, columna)
                'topleft': (0, 0),
                'bottomright': (0, 0),
            }
            fue_exitoso = False

        return fue_exitoso, detected_descriptors


class StaticDetectorWithModelAlignment(DepthStaticDetectorWithPCDFiltering):
    def __init__(self, matfile_path, obj_rgbd_name, obj_rgbd_num, ap_defaults=APDefaults(),
                 icp_defaults=ICPDefaults(), leaf_size=0.002,
                 icp_threshold=1e-3, perc_obj_model_pts=0.5):
        (super(StaticDetectorWithModelAlignment, self)
         .__init__(matfile_path, obj_rgbd_name, obj_rgbd_num))
        self._ap_defaults = ap_defaults
        self._icp_defaults = icp_defaults

        # El tamaño del radio de la esfera para buscar puntos en la escena
        self.leaf_size = leaf_size

        # Umbral de aceptacion para ICP
        self.icp_threshold = icp_threshold

        # Porcentaje minimo de puntos del modelo que deben detectarse
        self.perc_obj_model_pts = perc_obj_model_pts

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
        # scene_cloud = self._descriptors['pcd']

        # Esta es la nube de puntos proveniente de filtrar el cuadrado marcado
        # por la base de datos en la imagen RGB
        detected_cloud = detected_descriptors['object_cloud']

        accepted_points = (
            self._descriptors['obj_model_points'] * self.perc_obj_model_pts
        )

        # Defino los valores minimos que debería tener el resultado. Si se
        # pueden mejorar en el "for" de más adelante, genial!
        minmax = get_min_max(detected_cloud)
        detected_descriptors.update({
            'min_x_cloud': minmax.min_x,
            'max_x_cloud': minmax.max_x,
            'min_y_cloud': minmax.min_y,
            'max_y_cloud': minmax.max_y,
            'min_z_cloud': minmax.min_z,
            'max_z_cloud': minmax.max_z,

            # original model transformed
            'obj_model': model_cloud,

            # lo guardo solo para que se guarde la nube de puntos
            # cuando hago las corridas
            'detected_cloud': detected_cloud,
        })

        # Corro varias veces alignment e icp tratando de hacer bien la
        # alineacion
        transformed_model_cloud = model_cloud
        # show_clouds(
        #     b'Objeto detectado cuadrado vs escena',
        #     scene_cloud,
        #     detected_cloud,
        # )
        # show_clouds(
        #     b'Modelo vs objeto detectado cuadrado',
        #     model_cloud,
        #     detected_cloud,
        # )
        best_result_cloud = None
        best_threshold = self.icp_threshold

        for i in range(3):
            # Calculate alignment
            ap_result = align(
                transformed_model_cloud,
                detected_cloud,
                self._ap_defaults
            )

            # Calculate ICP
            if ap_result.has_converged:
                transformed_model_cloud = ap_result.cloud

            icp_res = icp(
                transformed_model_cloud,
                detected_cloud,
                self._icp_defaults
            )

            if icp_res.has_converged:
                transformed_model_cloud = icp_res.cloud

            if icp_res.has_converged and icp_res.score < best_threshold:
                best_threshold = icp_res.score
                best_result_cloud = icp_res.cloud

                # show_clouds(
                #     b'Modelo detectado vs escena',
                #     scene_cloud,
                #     transformed_model_cloud,
                # )

        if best_result_cloud is not None:
            # Filtro los puntos de la escena que se corresponden con el
            # objeto que estoy buscando
            obj_scene_cloud = filter_object_from_scene_cloud(
                best_result_cloud,  # object
                detected_cloud,  # partial scene
                self.leaf_size,  # radius
                False,  # show values
            )
            # show_clouds(
            #     b'Modelo detectado y filtrado vs escena',
            #     scene_cloud,
            #     obj_scene_cloud,
            # )

            obj_scene_points = points(obj_scene_cloud)

            extraccion_exitosa = obj_scene_points > accepted_points

            if extraccion_exitosa:
                minmax = get_min_max(obj_scene_cloud)

                detected_descriptors.update({
                    'min_x_cloud': minmax.min_x,
                    'max_x_cloud': minmax.max_x,
                    'min_y_cloud': minmax.min_y,
                    'max_y_cloud': minmax.max_y,
                    'min_z_cloud': minmax.min_z,
                    'max_z_cloud': minmax.max_z,
                    'object_cloud': obj_scene_cloud,

                    # original model transformed
                    'obj_model': best_result_cloud,

                    # lo guardo solo para que se guarde la nube de puntos
                    # cuando hago las corridas
                    'detected_cloud': best_result_cloud,
                })

                # show_clouds(
                #     b'Extraccion EXITOSA. Viendo objeto transformado',
                #     scene_cloud,
                #     transformed_model_cloud,
                # )

        return detected_descriptors


class DepthDetection(Detector):

    def __init__(self, ap_defaults=None, icp_defaults=None,
                 umbral_score=1e-3, **kwargs):
        super(DepthDetection, self).__init__()
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
        model_cloud_points = self._descriptors['obj_model_points']

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
    def __init__(self, template_threshold=0.02):
        super(RGBTemplateDetector, self).__init__()
        self.template_threshold = template_threshold

    def detect(self):
        img = self._descriptors['scene_rgb']
        templates = self._descriptors['object_templates']
        masks = self._descriptors['object_masks']


        fue_exitoso = False
        best_threshold = self.template_threshold
        topleft = (0, 0)
        bottomright = (0, 0)
        best_template_index = None

        for i, (template, mask) in enumerate(zip(templates, masks)):
            height, width = len(template), len(template[0])

            # Aplico la mascara al template
            # template_with_mask = cv2.bitwise_and(template, template, mask=mask)

            # Leer: http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_template_matching/py_template_matching.html#theory
            # Aplico el template Matching
            res = cv2.matchTemplate(img, template, cv2.TM_SQDIFF_NORMED)

            # Busco la posición
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            # print ('        El template numero {i:02d} tiene un umbral de '
            #        'comparacion de {u}'.format(i=i, u=min_val))

            # cv2.imshow('Template usado', template)
            # print "Valor de la comparacion:", min_val
            # cv2.waitKey()

            if min_val < best_threshold:
                # min_loc y max_loc tienen primero las columnas y despues las filas,
                # entonces lo doy vuelta
                fue_exitoso = True
                topleft = (min_loc[1], min_loc[0])
                bottomright = (min_loc[1] + height, min_loc[0] + width)
                best_template_index = i

        desc = {
            'topleft': topleft,
            'bottomright': bottomright,
            'object_frame': img[topleft[0]:bottomright[0],
                                topleft[1]:bottomright[1]],

        }

        if best_template_index is not None:
            desc.update({'best_template_index': best_template_index})

        return fue_exitoso, desc

    def calculate_descriptors(self, desc):
        if 'best_template_index' in desc:
            idx = desc['best_template_index']
            template = self._descriptors['object_templates'][idx]
            mask = self._descriptors['object_masks'][idx]

            desc['object_template'] = template
            desc['object_mask'] = mask

        return desc


class StaticDetectorForRGBFinder(StaticDetector):
    def calculate_descriptors(self, desc):
        img = self._descriptors['scene_rgb']
        img = img[desc['topleft'][0]:desc['bottomright'][0],
                  desc['topleft'][1]: desc['bottomright'][1]]
        desc.update({'object_frame': img})
        # cv2.imwrite('gorra_encontrada.png', img)

        desc['object_template'] = self._descriptors['object_templates'][0]
        desc['object_mask'] = self._descriptors['object_masks'][0]
        return desc


###################
# Detectores RGB-D
###################

class StaticDetectorForRGBD(StaticDetectorWithModelAlignment,
                            StaticDetectorForRGBFinder):
    def calculate_descriptors(self, detected_descriptors):
        desc = StaticDetectorWithModelAlignment.calculate_descriptors(
            self,
            detected_descriptors
        )
        desc = StaticDetectorForRGBFinder.calculate_descriptors(
            self,
            desc
        )

        return desc


class RGBDDetector(RGBTemplateDetector, DepthDetection):
    def __init__(self, template_threshold=0.02, ap_defaults=None,
                 icp_defaults=None, umbral_score=1e-3, depth_area_extra=0.33,
                 **kwargs):
        RGBTemplateDetector.__init__(self, template_threshold)
        DepthDetection.__init__(self, ap_defaults, icp_defaults, umbral_score, **kwargs)
        # Esto sirve para extender el area de busqueda para algoritmo de depth
        # tomando como referencia el area reportada por la deteccion RGB
        self.depth_area_extra = depth_area_extra

    def detect(self):
        # from analisis import Rectangle
        # Deteccion RGB
        rgb_fue_exitoso, rgb_desc = super(RGBDDetector, self).detect()

        depth_desc = {
            'topleft': (0, 0),  # (fila, columna)
            'bottomright': 0,
        }
        depth_fue_exitoso = False

        if rgb_fue_exitoso:
            # Tomo los datos del objeto a encontrar
            model_cloud = self._descriptors['obj_model']
            model_cloud_points = self._descriptors['obj_model_points']

            # Parametro de aceptacion del resultado nro 1
            accepted_points = model_cloud_points * self.perc_obj_model_points

            # Esto es para adaptar el rango de busqueda de puntos de la escena
            if not self.adapt_leaf.was_started():
                self.adapt_leaf.set_first_values(model_cloud_points)

            # Nube de puntos de la escena
            scene_cloud = self._descriptors['pcd']

            # Tamaño de la escena RGB
            rgb_height = self._descriptors['scene_rgb'].shape[0]
            rgb_width = self._descriptors['scene_rgb'].shape[1]

            topleft = rgb_desc['topleft']
            bottomright = rgb_desc['bottomright']

            # MuestraBusquedaEnVivo('Deteccion RGB').run(
            #     self._descriptors['scene_rgb'],
            #     topleft,
            #     bottomright,
            # )

            # Obtengo un bounding rectangulo cuya longitud en x e y es el doble
            # del encontrado por RGB pero con mismo centro de masa
            width = abs(bottomright[1] - topleft[1])
            height = abs(bottomright[0] - topleft[0])
            extra_width = max(int(width * self.depth_area_extra), 1)
            extra_height = max(int(height * self.depth_area_extra), 1)

            new_topleft = (
                max(topleft[0] - extra_height, 0),
                max(topleft[1] - extra_width, 0)
            )
            new_bottomright = (
                min(bottomright[0] + extra_height, rgb_height - 1),
                min(bottomright[1] + extra_width, rgb_width - 1)
            )
            # Rectangulo RGB y rectangulo nuevo
            # found_rgb = Rectangle(topleft, bottomright)
            # new_rgb = Rectangle(new_topleft, new_bottomright)
            # print "Rectangulo RGB detectado:", found_rgb.area()
            # print "Rectangulo RGB para Depth:", new_rgb.area()
            #
            # MuestraBusquedaEnVivo('Deteccion RGB').run(
            #     self._descriptors['scene_rgb'],
            #     new_topleft,
            #     new_bottomright,
            # )

            # Paso los valores del nuevo bounding box de RGB a DEPTH
            depth_img = self._descriptors['depth_img']
            top_bottom, left_right = from_flat_to_cloud_limits(
                new_topleft,
                new_bottomright,
                depth_img
            )
            # print "Depth topleft: (", top_bottom[0], ",", left_right[0], ")"
            # print "Depth bottomright: (", top_bottom[1], ",", left_right[1], ")"

            # Filtro la escena y obtengo solo los puntos señalados por la
            # deteccion RGB
            first_cloud = filter_cloud(
                scene_cloud,
                b'x',  # X son las columnas
                left_right[0],
                left_right[1],
            )
            cloud = filter_cloud(
                first_cloud,
                b'y',  # Y son las filas
                top_bottom[0],
                top_bottom[1],
            )

            best_aligned_scene = None
            best_alignment_score = self.umbral_score  # lesser is better

            # Corro la deteccion en depth varias veces, solo si la cantidad de
            # puntos es adecuada. Para mejorar un poco la condicion inicial de
            # cada corrida, el modelo de objeto que uso es el que va quedando
            # despues de cada alineacion, haya sido fallida o no
            if points(cloud) > model_cloud_points:
                for i in range(3):
                    # Calculate alignment
                    ap_result = align(model_cloud, cloud, self._ap_defaults)
                    if ap_result.has_converged:
                        model_cloud = ap_result.cloud
                        if ap_result.score < best_alignment_score:
                            best_alignment_score = ap_result.score
                            best_aligned_scene = ap_result.cloud


            # Su hubo una buena alineacion
            if best_aligned_scene is not None:
                # print "        HUBO UNA ALINEACION CORRECTA!!!!"
                # Calculate ICP
                icp_result = icp(best_aligned_scene, cloud, self._icp_defaults)

                if (icp_result.has_converged and
                        icp_result.score < self.umbral_score):
                    # print "        HUBO UN ICP CORRECTO!!!!"
                    # Filtro los puntos de la escena que se corresponden con el
                    # objeto que estoy buscando
                    # show_clouds(
                    #     b'Modelo alineado por AP e ICP vs escena parcial segun RGB',
                    #     icp_result.cloud,
                    #     cloud,
                    # )
                    # show_clouds(
                    #     b'Modelo alineado por AP e ICP vs escena completa',
                    #     icp_result.cloud,
                    #     scene_cloud,
                    # )
                    obj_scene_cloud = filter_object_from_scene_cloud(
                        icp_result.cloud,  # object
                        cloud,  # partial scene
                        self.adapt_leaf.leaf_ratio(),  # radius
                        False,  # show values
                    )

                    obj_scene_points = points(obj_scene_cloud)

                    depth_fue_exitoso = obj_scene_points > accepted_points

                    if depth_fue_exitoso:
                        self.adapt_leaf.set_found_points(obj_scene_points)
                        # print "        HUBO UNA DETECCION!!!!"
                    else:
                        self.adapt_leaf.reset()

                    topleft, bottomright = from_cloud_to_flat_limits(
                        obj_scene_cloud
                    )

                    depth_desc.update({
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

        rgb_desc.update(depth_desc)

        return depth_fue_exitoso, rgb_desc

    def calculate_descriptors(self, desc):
        desc = super(RGBDDetector, self).calculate_descriptors(desc)

        # Obtengo la bounding box del objeto en la nube de puntos
        obj_scene_cloud = desc['object_cloud']
        minmax = get_min_max(obj_scene_cloud)
        desc.update({
            'min_x_cloud': minmax.min_x,
            'max_x_cloud': minmax.max_x,
            'min_y_cloud': minmax.min_y,
            'max_y_cloud': minmax.max_y,
            'min_z_cloud': minmax.min_z,
            'max_z_cloud': minmax.max_z,
        })

        # Actualizo el object_frame para RGB segun el resultado en DEPTH
        img = self._descriptors['scene_rgb']
        desc.update({
            'object_frame': img[desc['topleft'][0]:desc['bottomright'][0],
                                desc['topleft'][1]:desc['bottomright'][1]],
        })

        return desc


class StaticDepthTransformationDetection(Detector):
    def __init__(self, transf_file_path, sname, snum, objname, objnum):
        super(StaticDepthTransformationDetection, self).__init__()

        # Deteccion por spin_images/fpfh/cshot
        transf_file = ('{path}/{sname}/{sname}_{snum}/{objname}/'
                       '{objname}_{objnum}/{objname}_{objnum}_{nframe}.txt')
        self.transf_file = transf_file.format(
            path=transf_file_path,
            sname=sname,
            snum=snum,
            objname=objname,
            objnum=objnum,
            nframe='{nframe}'
        )

    def transformation_detect(self):
        nframe = self._descriptors['nframe']
        model = self._descriptors['static_obj_model']
        transf_file = self.transf_file.format(nframe=nframe)

        fue_exitoso = False
        tam_region = 0
        topleft = (0, 0)
        bottomright = (0, 0)

        detected_descriptors = {}

        if os.path.isfile(transf_file):
            fue_exitoso = True

            with open(transf_file, 'r') as file_:
                transf_matrix = VectorMat()
                for line in file_:
                    if line.strip():
                        values = line.strip().split()
                        floats = [float(v) for v in values]
                        float_vec = FloatVector()
                        float_vec.extend(floats)
                        transf_matrix.append(float_vec)

            # Deteccion por spin_images/fpfh/cshot
            detected_model = transform_cloud(model, transf_matrix)

            topleft, bottomright = from_cloud_to_flat_limits(detected_model)
            tam_region = max(bottomright[0] - topleft[0],
                             bottomright[1] - topleft[1])
            detected_descriptors.update({
                'detected_cloud': detected_model,
            })
            # scene_cloud = self._descriptors['pcd']
            # show_clouds(
            #     b'Modelo detectado (transformacion estatica) vs escena',
            #     scene_cloud,
            #     detected_model,
            # )

        detected_descriptors.update({
            'size': tam_region,
            'location': topleft,  # topleft=(fila, columna)
            'topleft': topleft,
            'bottomright': bottomright,
        })

        return fue_exitoso, detected_descriptors

    def detect(self):
        return self.transformation_detect()


class SDTWithPostAlignment(StaticDepthTransformationDetection):
    def __init__(self, transf_file_path, sname, snum, objname, objnum,
                 ap_defaults=None, icp_defaults=None,
                 umbral_score=1e-3, **kwargs):
        (super(SDTWithPostAlignment, self)
         .__init__(transf_file_path, sname, snum, objname, objnum))

        # alignment prerejective e ICP
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
        first_leaf_size = kwargs.get('first_leaf_size', 0.005)
        self.adapt_leaf = kwargs.get(
            'adapt_leaf',
            AdaptLeafRatio(first_leaf=first_leaf_size)
        )

        # Seteo el porcentaje de puntos que permito conservar del modelo del
        # objeto antes de considerar que lo que se encontró no es el objeto
        self.perc_obj_model_points = kwargs.get('perc_obj_model_points', 0.5)

        # Seteo el tamaño del frame de busqueda. Este valor se va a multiplicar
        # por la altura y el ancho del objeto. Ej: si se multiplica por 2, el
        # frame de busqueda tiene un area 4 (2*2) veces mayor que la del objeto
        self.obj_mult = kwargs.get('obj_mult', 2)

    def _best_alignment_prerejective(self, model_cloud, scene_cloud, times=6):

        best_result = align(model_cloud, scene_cloud, self._ap_defaults)
        best_alignment_score = min(self.umbral_score, best_result.score)

        for i in range(times - 1):
            # Puede que fallen los primeros align, por eso hago esto de abajo
            if not best_result.has_converged:
                ap_result = align(model_cloud, scene_cloud, self._ap_defaults)
            else:
                ap_result = align(best_result.cloud, scene_cloud, self._ap_defaults)

            if (ap_result.has_converged and
                    ap_result.score < best_alignment_score):
                best_alignment_score = ap_result.score
                best_result = ap_result

        return best_result

    def alignment_prerejective(self, descriptors):
        # obtengo tamaño del objeto detetado y me quedo con uno X veces mas grande
        model_cloud = descriptors['detected_cloud']
        if not self.adapt_leaf.was_started():
            model_points = self._descriptors['static_obj_model_points']
            self.adapt_leaf.set_first_values(model_points)

        obj_limits = get_min_max(model_cloud)

        length_func = lambda mul, l: l * mul / 2.0
        max_length = max(
            obj_limits.max_x - obj_limits.min_x,
            obj_limits.max_y - obj_limits.min_y,
            obj_limits.max_z - obj_limits.min_z,
        )
        x_center = obj_limits.max_x - (obj_limits.max_x - obj_limits.min_x)
        y_center = obj_limits.max_y - (obj_limits.max_y - obj_limits.min_y)
        z_center = obj_limits.max_z - (obj_limits.max_z - obj_limits.min_z)
        half_side_length = length_func(self.obj_mult, max_length)

        # obtengo limites de la escena
        scene_cloud = self._descriptors['pcd']

        # Filtro la escena y me quedo con la bounding-box de la deteccion por
        # transformaciones
        cloud = filter_cloud(
            scene_cloud,
            b'x',
            x_center - half_side_length,
            x_center + half_side_length
        )
        cloud = filter_cloud(
            cloud,
            b'y',
            y_center - half_side_length,
            y_center + half_side_length
        )
        cloud = filter_cloud(
            cloud,
            b'z',
            z_center - half_side_length,
            z_center + half_side_length
        )

        # show_clouds(
        #     b'Escena filtrando el bounding-box',
        #     cloud,
        #     model_cloud,
        # )

        detected_descriptors = {
            'topleft': (0, 0),  # (fila, columna)
            'bottomright': 0,
        }
        fue_exitoso = False

        accepted_points = (
            self._descriptors['static_obj_model_points'] *
            self.perc_obj_model_points
        )

        if points(cloud) > accepted_points:

            ap_result = self._best_alignment_prerejective(model_cloud, cloud)

            # print "Convergio AP:", ap_result.has_converged
            # print "Score AP:", ap_result.score, "(<", self.umbral_score, ")"

            # show_clouds(
            #     b'Escena filtrada vs alignment_prerejective',
            #     cloud,
            #     ap_result.cloud,
            # )

            if ap_result.has_converged and ap_result.score < self.umbral_score:
                # Calculate ICP
                icp_result = icp(ap_result.cloud, cloud, self._icp_defaults)
                # print "Convergio ICP:", icp_result.has_converged
                # print "Score ICP:", icp_result.score, "(<", self.umbral_score, ")"

                # show_clouds(
                #     b'Escena filtrada vs icp',
                #     cloud,
                #     icp_result.cloud,
                # )

                if (icp_result.has_converged and
                        icp_result.score < self.umbral_score):
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
                    tam_region = max(bottomright[0] - topleft[0],
                                     bottomright[1] - topleft[1])

                    detected_descriptors.update({
                        'min_x_cloud': minmax.min_x,
                        'max_x_cloud': minmax.max_x,
                        'min_y_cloud': minmax.min_y,
                        'max_y_cloud': minmax.max_y,
                        'min_z_cloud': minmax.min_z,
                        'max_z_cloud': minmax.max_z,
                        'object_cloud': obj_scene_cloud,
                        'detected_cloud': icp_result.cloud,  # lo guardo solo para la estadistica
                        'size': tam_region,
                        'location': topleft,
                        'topleft': topleft,  # (fila, columna)
                        'bottomright': bottomright,
                    })

                    # show_clouds(
                    #   b'Modelo detectado y filtrado vs escena',
                    #   scene_cloud,
                    #   obj_scene_cloud,
                    # )
                    # show_clouds(
                    #   b'Modelo detectado por TRANSF, AP e ICP vs escena',
                    #   scene_cloud,
                    #   icp_result.cloud,
                    # )

        return fue_exitoso, detected_descriptors

    def detect(self):
        fue_exitoso, detected_descriptors = self.transformation_detect()

        if fue_exitoso:
            print 'Hubo detección segun transformación de Nadia.'
            fue_exitoso, detected_descriptors = self.alignment_prerejective(
                detected_descriptors
            )
        else:
            print 'Falló la detección segun transformación de Nadia.'

        return fue_exitoso, detected_descriptors
