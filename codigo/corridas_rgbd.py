#coding=utf-8

from __future__ import (unicode_literals, division)

import cv2

from cpp.alignment_prerejective import APDefaults
from cpp.icp import ICPDefaults

from buscadores import ICPFinderWithModel, HistogramComparator, \
    ImproveObjectFoundWithHistogramFinder
from metodos_comunes import AdaptLeafRatio, FixedSearchArea
from metodos_de_busqueda import BusquedaCambiandoSizePeroMismoCentro
from detectores import StaticDetectorForRGBD
from esquemas_seguimiento import FollowingSchemeSavingDataRGBD, \
    FollowingSchemeExploringParameterRGBD
from observar_seguimiento import MuestraSeguimientoEnVivo
from proveedores_de_imagenes import FrameNamesAndImageProvider, \
    FrameNamesAndImageProviderPreChargedForPCD
from seguidores import RGBDPreferDFollowerWithStaticDetection



def correr_con_depth_como_principal(img_provider, scenename, scenenumber, objname):
    # Set alignment detection parameters values
    ap_defaults = APDefaults()
    ap_defaults.leaf = 0.005
    ap_defaults.max_ransac_iters = 120
    ap_defaults.points_to_sample = 3
    ap_defaults.nearest_features_used = 4
    ap_defaults.simil_threshold = 0.6
    ap_defaults.inlier_threshold = 4
    ap_defaults.inlier_fraction = 0.3

    icp_detection_defaults = ICPDefaults()
    icp_detection_defaults.euc_fit = 1e-10
    icp_detection_defaults.max_corr_dist = 3
    icp_detection_defaults.max_iter = 50
    icp_detection_defaults.transf_epsilon = 1e-6

    det_umbral_score = 1e-3
    det_obj_scene_leaf = 0.005
    det_perc_obj_model_points = 0.5

    # Set depth following parameters values
    icp_finder_defaults = ICPDefaults()
    icp_finder_defaults.euc_fit = 1e-10
    icp_finder_defaults.max_corr_dist = 0.1
    icp_finder_defaults.max_iter = 50
    icp_finder_defaults.transf_epsilon = 1e-6

    find_umbral_score = 1e-4
    find_adapt_area = FixedSearchArea(3)
    find_adapt_leaf = AdaptLeafRatio()
    find_obj_scene_leaf = 0.002
    find_perc_obj_model_points = 0.6

    # Set RGB following parameters values
    find_template_comp_method = cv2.cv.CV_COMP_BHATTACHARYYA
    template_perc = 0.6
    template_worst = 1
    find_template_reverse = False

    find_frame_comp_method = cv2.cv.CV_COMP_BHATTACHARYYA
    frame_perc = 0.3
    frame_worst = 1
    find_frame_reverse = False

    metodo_de_busqueda = BusquedaCambiandoSizePeroMismoCentro()

    # Repetir 3 veces para evitar detecciones fallidas por RANSAC
    for i in range(3):
        detector = StaticDetectorForRGBD(
            matfile_path=('videos/rgbd/scenes/{sname}/{sname}_{snum}.mat'
                          .format(sname=scenename, snum=scenenumber)),
            obj_rgbd_name=objname,
            ap_defaults=ap_defaults,
            icp_defaults=icp_detection_defaults,
            leaf_size=det_obj_scene_leaf,
            icp_threshold=det_umbral_score,
            perc_obj_model_pts=det_perc_obj_model_points
        )

        template_comparator = HistogramComparator(
            method=find_template_comp_method,
            perc=template_perc,
            worst_case=template_worst,
            reverse=find_template_reverse,
        )
        frame_comparator = HistogramComparator(
            method=find_frame_comp_method,
            perc=frame_perc,
            worst_case=frame_worst,
            reverse=find_frame_reverse,
        )

        rgb_finder = ImproveObjectFoundWithHistogramFinder(
            template_comparator,
            frame_comparator,
            metodo_de_busqueda,
        )

        depth_finder = ICPFinderWithModel(
            icp_defaults=icp_finder_defaults,
            umbral_score=find_umbral_score,
            adapt_area=find_adapt_area,
            adapt_leaf=find_adapt_leaf,
            first_leaf_size=find_obj_scene_leaf,
            perc_obj_model_points=find_perc_obj_model_points,
        )

        follower = RGBDPreferDFollowerWithStaticDetection(
            image_provider=img_provider,
            depth_static_detector=detector,
            rgb_finder=rgb_finder,
            depth_finder=depth_finder,
        )

        # mostrar_seguimiento = MuestraSeguimientoEnVivo('Seguimiento')

        # FollowingScheme(img_provider, follower, mostrar_seguimiento).run()
        FollowingSchemeExploringParameterRGBD(
            img_provider,
            follower,
            'pruebas_guardadas',
            'definitivo_RGBD',
            'DEFINITIVO',
        ).run()

        img_provider.restart()


if __name__ == '__main__':
    desk_1_img_provider = FrameNamesAndImageProviderPreChargedForPCD(
        'videos/rgbd/scenes/', 'desk', '1',
        'videos/rgbd/objs/', 'coffee_mug', '5',
    )  # path, objname, number

    # correr_con_depth_como_principal(desk_1_img_provider, 'desk', '1', 'coffee_mug')

    desk_1_img_provider.reinitialize_object('cap', '4')
    correr_con_depth_como_principal(desk_1_img_provider, 'desk', '1', 'cap')

    # desk_2_img_provider = FrameNamesAndImageProviderPreChargedForPCD(
    #     'videos/rgbd/scenes/', 'desk', '2',
    #     'videos/rgbd/objs/', 'bowl', '3',
    # )  # path, objname, number
    # correr_con_depth_como_principal(desk_2_img_provider, 'desk', '2', 'bowl')





