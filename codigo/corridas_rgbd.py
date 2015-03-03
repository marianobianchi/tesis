#coding=utf-8

from __future__ import (unicode_literals, division)

import cv2

from cpp.alignment_prerejective import APDefaults
from cpp.icp import ICPDefaults

from buscadores import ICPFinderWithModel, HistogramComparator, Finder, \
    ImproveObjectFoundWithHistogramFinder, TemplateAndFrameHistogramFinder, \
    ICPFinderWithModelToImproveRGB
from metodos_comunes import AdaptLeafRatio, FixedSearchArea
from metodos_de_busqueda import BusquedaCambiandoSizePeroMismoCentro, \
    BusquedaAlrededorCambiandoFrameSize
from detectores import StaticDetectorForRGBD, RGBDDetector
from esquemas_seguimiento import FollowingSchemeExploringParameterRGBD, \
    FollowingScheme
from proveedores_de_imagenes import FrameNamesAndImageProviderPreChargedForPCD
from seguidores import DetectionWithCombinedFollowers


def correr_con_depth_como_principal(img_provider, scenename, scenenumber,
                                    objname, objnum):
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
            obj_rgbd_num=objnum,
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

        follower = DetectionWithCombinedFollowers(
            image_provider=img_provider,
            detector=detector,
            main_finder=depth_finder,
            secondary_finder=rgb_finder,
        )

        # mostrar_seguimiento = MuestraSeguimientoEnVivo('Seguimiento')

        # FollowingScheme(img_provider, follower, mostrar_seguimiento).run()
        FollowingSchemeExploringParameterRGBD(
            img_provider,
            follower,
            'pruebas_guardadas',
            'definitivo_RGBD_preferD',
            'DEFINITIVO',
        ).run()

        img_provider.restart()


def correr_con_rgb_como_principal(img_provider, scenename, scenenumber,
                                  objname, objnum):
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

    metodo_de_busqueda = BusquedaAlrededorCambiandoFrameSize()

    # Repetir 3 veces para evitar detecciones fallidas por RANSAC
    for i in range(3):
        detector = StaticDetectorForRGBD(
            matfile_path=('videos/rgbd/scenes/{sname}/{sname}_{snum}.mat'
                          .format(sname=scenename, snum=scenenumber)),
            obj_rgbd_name=objname,
            obj_rgbd_num=objnum,
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

        rgb_finder = TemplateAndFrameHistogramFinder(
            template_comparator,
            frame_comparator,
            metodo_de_busqueda,
        )

        depth_finder = ICPFinderWithModelToImproveRGB(
            icp_defaults=icp_finder_defaults,
            umbral_score=find_umbral_score,
            adapt_area=find_adapt_area,
            adapt_leaf=find_adapt_leaf,
            first_leaf_size=find_obj_scene_leaf,
            perc_obj_model_points=find_perc_obj_model_points,
        )

        follower = DetectionWithCombinedFollowers(
            image_provider=img_provider,
            depth_static_detector=detector,
            main_finder=rgb_finder,
            secondary_finder=depth_finder,
        )

        # mostrar_seguimiento = MuestraSeguimientoEnVivo('Seguimiento RGBD RGB')

        FollowingSchemeExploringParameterRGBD(
            img_provider,
            follower,
            'pruebas_guardadas',
            'definitivo_RGBD_preferRGB',
            'DEFINITIVO',
            # show_following=mostrar_seguimiento,
        ).run()

        img_provider.restart()


def prueba_solo_deteccion_automatica(img_provider):

    # Set detection parameters values
    det_template_threshold = 0.2

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
    det_depth_area_extra = 0.333

    detector = RGBDDetector(
        template_threshold=det_template_threshold,
        ap_defaults=ap_defaults,
        icp_defaults=icp_detection_defaults,
        umbral_score=det_umbral_score,
        perc_obj_model_points=det_perc_obj_model_points,
        first_leaf_size=det_obj_scene_leaf,
        depth_area_extra=det_depth_area_extra,
    )

    finder = Finder()

    follower = DetectionWithCombinedFollowers(
        img_provider,
        detector,
        finder,
        finder
    )

    for i in range(3):
        FollowingSchemeExploringParameterRGBD(
            img_provider,
            follower,
            'pruebas_guardadas',
            'probando_deteccion_automatica_RGBD',
            'UNICO',
        ).run()

        img_provider.restart()


def correr_escena_1():
    desk_1_img_provider = FrameNamesAndImageProviderPreChargedForPCD(
        'videos/rgbd/scenes/', 'desk', '1',
        'videos/rgbd/objs/', 'coffee_mug', '5',
    )  # path, objname, number

    # DEPTH como principal
    desk_1_img_provider.reinitialize_object('coffee_mug', '5')
    correr_con_depth_como_principal(
        desk_1_img_provider,
        'desk', '1',
        'coffee_mug', '5'
    )

    desk_1_img_provider.reinitialize_object('cap', '4')
    correr_con_depth_como_principal(
        desk_1_img_provider,
        'desk', '1',
        'cap', '4'
    )

    # RGB como principal
    desk_1_img_provider.reinitialize_object('coffee_mug', '5')
    correr_con_rgb_como_principal(
        desk_1_img_provider,
        'desk', '1',
        'coffee_mug', '5'
    )

    desk_1_img_provider.reinitialize_object('cap', '4')
    correr_con_rgb_como_principal(
        desk_1_img_provider,
        'desk', '1',
        'cap', '4'
    )


def correr_escena_2():
    desk_2_img_provider = FrameNamesAndImageProviderPreChargedForPCD(
        'videos/rgbd/scenes/', 'desk', '2',
        'videos/rgbd/objs/', 'bowl', '3',
    )  # path, objname, number

    # DEPTH como principal
    correr_con_depth_como_principal(
        desk_2_img_provider,
        'desk', '2',
        'bowl', '3'
    )

    # RGB como principal
    desk_2_img_provider.reinitialize_object('bowl', '3')
    correr_con_rgb_como_principal(
        desk_2_img_provider,
        'desk', '2',
        'bowl', '3'
    )


def correr_escena_3():
    table_1_img_provider = FrameNamesAndImageProviderPreChargedForPCD(
        'videos/rgbd/scenes/', 'table', '1',
        'videos/rgbd/objs/', 'coffee_mug', '1',
    )  # path, objname, number

    # DEPTH como principal
    table_1_img_provider.reinitialize_object('coffee_mug', '1')
    correr_con_depth_como_principal(table_1_img_provider, 'table', '1', 'coffee_mug', '1')

    table_1_img_provider.reinitialize_object('soda_can', '4')
    correr_con_depth_como_principal(table_1_img_provider, 'table', '1', 'soda_can', '4')

    # RGB como principal
    table_1_img_provider.reinitialize_object('coffee_mug', '1')
    correr_con_rgb_como_principal(table_1_img_provider, 'table', '1', 'coffee_mug', '1')

    table_1_img_provider.reinitialize_object('soda_can', '4')
    correr_con_rgb_como_principal(table_1_img_provider, 'table', '1', 'soda_can', '4')


def correr_escena_4():
    table_small_2_img_provider = FrameNamesAndImageProviderPreChargedForPCD(
        'videos/rgbd/scenes/', 'table_small', '2',
        'videos/rgbd/objs/', 'cereal_box', '4',
    )  # path, objname, number

    # DEPTH como principal
    correr_con_depth_como_principal(table_small_2_img_provider, 'table_small', '2', 'cereal_box', '4')

    # RGB como principal
    table_small_2_img_provider.reinitialize_object('cereal_box', '4')
    correr_con_rgb_como_principal(table_small_2_img_provider, 'table_small', '2', 'cereal_box', '4')


def sistema_de_seguimiento_automatico_completo(img_provider):

    # Set detection parameters values
    det_template_threshold = 0.2

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
    det_depth_area_extra = 0.333

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
        detector = RGBDDetector(
            template_threshold=det_template_threshold,
            ap_defaults=ap_defaults,
            icp_defaults=icp_detection_defaults,
            umbral_score=det_umbral_score,
            perc_obj_model_points=det_perc_obj_model_points,
            first_leaf_size=det_obj_scene_leaf,
            depth_area_extra=det_depth_area_extra,
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

        follower = DetectionWithCombinedFollowers(
            image_provider=img_provider,
            detector=detector,
            main_finder=depth_finder,
            secondary_finder=rgb_finder,
        )

        # mostrar_seguimiento = MuestraSeguimientoEnVivo('Seguimiento')

        # FollowingScheme(img_provider, follower, mostrar_seguimiento).run()
        FollowingSchemeExploringParameterRGBD(
            img_provider,
            follower,
            'pruebas_guardadas',
            'definitivo_automatico_RGBD',
            'UNICO',
        ).run()

        img_provider.restart()


def sistema_de_seguimiento_automatico_completo_rgb_rgbd(img_provider):

    # Set detection parameters values
    det_template_threshold = 0.2

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
    det_depth_area_extra = 0.333

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
        detector = RGBDDetector(
            template_threshold=det_template_threshold,
            ap_defaults=ap_defaults,
            icp_defaults=icp_detection_defaults,
            umbral_score=det_umbral_score,
            perc_obj_model_points=det_perc_obj_model_points,
            first_leaf_size=det_obj_scene_leaf,
            depth_area_extra=det_depth_area_extra,
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

        rgb_finder = TemplateAndFrameHistogramFinder(
            template_comparator,
            frame_comparator,
            metodo_de_busqueda,
        )
        depth_finder = ICPFinderWithModelToImproveRGB(
            icp_defaults=icp_finder_defaults,
            umbral_score=find_umbral_score,
            adapt_area=find_adapt_area,
            adapt_leaf=find_adapt_leaf,
            first_leaf_size=find_obj_scene_leaf,
            perc_obj_model_points=find_perc_obj_model_points,
        )

        follower = DetectionWithCombinedFollowers(
            image_provider=img_provider,
            detector=detector,
            main_finder=depth_finder,
            secondary_finder=rgb_finder,
        )

        # mostrar_seguimiento = MuestraSeguimientoEnVivo('Seguimiento')

        # FollowingScheme(img_provider, follower, mostrar_seguimiento).run()
        FollowingSchemeExploringParameterRGBD(
            img_provider,
            follower,
            'pruebas_guardadas',
            'definitivo_automatico_RGB_RGBD',
            'UNICO',
        ).run()

        img_provider.restart()


def guardar_todos_los_movimientos_del_algoritmo():
    # DEPTH como principal

    img_provider = FrameNamesAndImageProviderPreChargedForPCD(
        'videos/rgbd/scenes/', 'desk', '1',
        'videos/rgbd/objs/', 'cap', '4',
    )  # path, objname, number
    scenename = 'desk'
    scenenumber = '1'
    objname = 'cap'
    objnum = '4'

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

    detector = StaticDetectorForRGBD(
        matfile_path=('videos/rgbd/scenes/{sname}/{sname}_{snum}.mat'
                      .format(sname=scenename, snum=scenenumber)),
        obj_rgbd_name=objname,
        obj_rgbd_num=objnum,
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

    follower = DetectionWithCombinedFollowers(
        image_provider=img_provider,
        detector=detector,
        main_finder=depth_finder,
        secondary_finder=rgb_finder,
    )

    # mostrar_seguimiento = MuestraSeguimientoEnVivo('Seguimiento')
    mostrar_seguimiento = None
    FollowingScheme(img_provider, follower, mostrar_seguimiento).run()

if __name__ == '__main__':

    # correr_escena_1()
    #
    # correr_escena_2()
    #
    # correr_escena_3()
    #
    # correr_escena_4()

    # Probando deteccion automatica sin seguimiento
    # desk_1_img_provider = FrameNamesAndImageProviderPreChargedForPCD(
    #    'videos/rgbd/scenes/', 'desk', '1',
    #    'videos/rgbd/objs/', 'coffee_mug', '5',
    # )  # path, objname, number
    #
    # desk_1_img_provider.reinitialize_object('coffee_mug', '5')
    # prueba_solo_deteccion_automatica(desk_1_img_provider)
    #
    # desk_1_img_provider.reinitialize_object('cap', '4')
    # prueba_solo_deteccion_automatica(desk_1_img_provider)
    #
    # desk_2_img_provider = FrameNamesAndImageProviderPreChargedForPCD(
    #     'videos/rgbd/scenes/', 'desk', '2',
    #     'videos/rgbd/objs/', 'bowl', '3',
    # )  # path, objname, number
    # prueba_solo_deteccion_automatica(desk_2_img_provider)

    # Sistema de seguimiento completamente automatico RGBD-RGB
    # desk_1_img_provider = FrameNamesAndImageProviderPreChargedForPCD(
    #    'videos/rgbd/scenes/', 'desk', '1',
    #    'videos/rgbd/objs/', 'coffee_mug', '5',
    # )  # path, objname, number
    #
    # desk_1_img_provider.reinitialize_object('coffee_mug', '5')
    # sistema_de_seguimiento_automatico_completo(desk_1_img_provider)
    #
    # desk_1_img_provider.reinitialize_object('cap', '4')
    # sistema_de_seguimiento_automatico_completo(desk_1_img_provider)
    #
    # desk_2_img_provider = FrameNamesAndImageProviderPreChargedForPCD(
    #     'videos/rgbd/scenes/', 'desk', '2',
    #     'videos/rgbd/objs/', 'bowl', '3',
    # )  # path, objname, number
    # sistema_de_seguimiento_automatico_completo(desk_2_img_provider)

    # table_1_img_provider = FrameNamesAndImageProviderPreChargedForPCD(
    #     'videos/rgbd/scenes/', 'table', '1',
    #     'videos/rgbd/objs/', 'coffee_mug', '1',
    # )  # path, objname, number

    # table_1_img_provider.reinitialize_object('coffee_mug', '1')
    # sistema_de_seguimiento_automatico_completo(table_1_img_provider)

    # table_1_img_provider.reinitialize_object('soda_can', '4')
    # sistema_de_seguimiento_automatico_completo(table_1_img_provider)

    # table_small_2_img_provider = FrameNamesAndImageProviderPreChargedForPCD(
    #     'videos/rgbd/scenes/', 'table_small', '2',
    #     'videos/rgbd/objs/', 'cereal_box', '4',
    # )  # path, objname, number
    #
    # sistema_de_seguimiento_automatico_completo(table_small_2_img_provider)

    # Sistema de seguimiento completamente automatico RGB-RGBD
    # desk_1_img_provider = FrameNamesAndImageProviderPreChargedForPCD(
    #     'videos/rgbd/scenes/', 'desk', '1',
    #     'videos/rgbd/objs/', 'coffee_mug', '5',
    # )  # path, objname, number
    #
    # desk_1_img_provider.reinitialize_object('coffee_mug', '5')
    # sistema_de_seguimiento_automatico_completo_rgb_rgbd(desk_1_img_provider)
    #
    # desk_1_img_provider.reinitialize_object('cap', '4')
    # sistema_de_seguimiento_automatico_completo_rgb_rgbd(desk_1_img_provider)
    #
    # desk_2_img_provider = FrameNamesAndImageProviderPreChargedForPCD(
    #     'videos/rgbd/scenes/', 'desk', '2',
    #     'videos/rgbd/objs/', 'bowl', '3',
    # )  # path, objname, number
    # sistema_de_seguimiento_automatico_completo_rgb_rgbd(desk_2_img_provider)
    #
    # table_1_img_provider = FrameNamesAndImageProviderPreChargedForPCD(
    #     'videos/rgbd/scenes/', 'table', '1',
    #     'videos/rgbd/objs/', 'coffee_mug', '1',
    # )  # path, objname, number
    #
    # table_1_img_provider.reinitialize_object('coffee_mug', '1')
    # sistema_de_seguimiento_automatico_completo_rgb_rgbd(table_1_img_provider)
    #
    # table_1_img_provider.reinitialize_object('soda_can', '4')
    # sistema_de_seguimiento_automatico_completo_rgb_rgbd(table_1_img_provider)

    # table_small_2_img_provider = FrameNamesAndImageProviderPreChargedForPCD(
    #     'videos/rgbd/scenes/', 'table_small', '2',
    #     'videos/rgbd/objs/', 'cereal_box', '4',
    # )  # path, objname, number
    #
    # sistema_de_seguimiento_automatico_completo_rgb_rgbd(
    #     table_small_2_img_provider
    # )

    guardar_todos_los_movimientos_del_algoritmo()
