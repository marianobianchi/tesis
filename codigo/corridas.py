#coding=utf-8

from __future__ import (unicode_literals, division)

from cpp.alignment_prerejective import APDefaults
from cpp.icp import ICPDefaults

from buscadores import Finder, ICPFinder, ICPFinderWithModel
from metodos_comunes import AdaptLeafRatio, AdaptSearchArea
from detectores import StaticDetector, StaticDetectorWithPCDFiltering, \
    StaticDetectorWithModelAlignment, AutomaticDetection
from esquemas_seguimiento import FollowingScheme, FollowingSchemeSavingData, \
    FollowingSquemaExploringParameter
from metodos_comunes import Timer
from observar_seguimiento import MuestraSeguimientoEnVivo
from proveedores_de_imagenes import FrameNamesAndImageProvider, \
    FrameNamesAndImageProviderPreCharged
from seguidores import FollowerWithStaticDetection, \
    FollowerWithStaticDetectionAndPCD, FollowerStaticICPAndObjectModel


def deteccion_estatica():
    img_provider = FrameNamesAndImageProvider(
        'videos/rgbd/scenes/',  # scene path
        'desk',  # scene
        '1',  # scene number
        'videos/rgbd/objs/',  # object path
        'coffee_mug',  # object
        '5',  # object number
    )

    detector = StaticDetector(
        'videos/rgbd/scenes/desk/desk_1.mat',
        'coffee_mug'
    )

    finder = Finder()

    follower = FollowerWithStaticDetection(img_provider, detector, finder)

    show_following = MuestraSeguimientoEnVivo(
        'Deteccion estatica - Sin seguidor'
    )

    FollowingScheme(
        img_provider,
        follower,
        show_following,
    ).run()


def icp_sin_modelo():
    img_provider = FrameNamesAndImageProviderPreCharged(
        'videos/rgbd/scenes/',  # scene path
        'desk',  # scene
        '1',  # scene number
        'videos/rgbd/objs/',  # object path
        'coffee_mug',  # object
        '5',  # object number
    )

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


def icp_con_modelo():
    img_provider = FrameNamesAndImageProviderPreCharged(
        'videos/rgbd/scenes/', 'desk', '1', 'videos/rgbd/objs/', 'coffee_mug', '5',
    )  # path, objname, number

    detector = StaticDetectorWithModelAlignment(
        'videos/rgbd/scenes/desk/desk_1.mat',
        'coffee_mug'
    )

    finder = ICPFinderWithModel()

    follower = FollowerStaticICPAndObjectModel(img_provider, detector, finder)

    show_following = MuestraSeguimientoEnVivo('Seguidor ICP')

    FollowingScheme(
        img_provider,
        follower,
        show_following,
    ).run()


def deteccion_automatica_icp_con_modelo():
    img_provider = FrameNamesAndImageProviderPreCharged(
        'videos/rgbd/scenes/', 'desk', '1',
        'videos/rgbd/objs/', 'coffee_mug', '5',
    )  # path, objname, number

    detector = AutomaticDetection()

    finder = ICPFinderWithModel()

    follower = FollowerStaticICPAndObjectModel(img_provider, detector, finder)

    show_following = MuestraSeguimientoEnVivo('Seguidor ICP')

    FollowingScheme(
        img_provider,
        follower,
        show_following,
    ).run()


def desk_1_coffee_mug_5():
    """
    algoritmo final guardando datos
    """
    img_provider = FrameNamesAndImageProvider(
        'videos/rgbd/scenes/', 'desk', '1',
        'videos/rgbd/objs/', 'coffee_mug', '5',
    )  # path, objname, number

    detector = AutomaticDetection()

    finder = ICPFinderWithModel()

    follower = FollowerStaticICPAndObjectModel(img_provider, detector, finder)

    FollowingSchemeSavingData(
        img_provider,
        follower,
        'pruebas_guardadas'
    ).run()


def desk_2_flashlight_1():
    """
    algoritmo final guardando datos
    """
    img_provider = FrameNamesAndImageProvider(
        'videos/rgbd/scenes/', 'desk', '2',
        'videos/rgbd/objs/', 'flashlight', '1',
    )  # path, objname, number

    detector = AutomaticDetection()

    finder = ICPFinderWithModel()

    follower = FollowerStaticICPAndObjectModel(img_provider, detector, finder)

    FollowingSchemeSavingData(
        img_provider,
        follower,
        'pruebas_guardadas'
    ).run()


def desk_1_cap_4():
    """
    algoritmo final guardando datos
    """
    img_provider = FrameNamesAndImageProvider(
        'videos/rgbd/scenes/', 'desk', '1',
        'videos/rgbd/objs/', 'cap', '4',
    )  # path, objname, number

    detector = AutomaticDetection()

    finder = ICPFinderWithModel()

    follower = FollowerStaticICPAndObjectModel(img_provider, detector, finder)

    FollowingSchemeSavingData(
        img_provider,
        follower,
        'pruebas_guardadas'
    ).run()


def desk_2_bowl_3():
    """
    algoritmo final guardando datos

    Aparece en los frames 26-67, 111-166
    """
    img_provider = FrameNamesAndImageProvider(
        'videos/rgbd/scenes/', 'desk', '2',
        'videos/rgbd/objs/', 'bowl', '3',
    )  # path, objname, number

    detector = AutomaticDetection()

    finder = ICPFinderWithModel()

    follower = FollowerStaticICPAndObjectModel(img_provider, detector, finder)

    FollowingSchemeSavingData(
        img_provider,
        follower,
        'pruebas_guardadas'
    ).run()


def correr_ejemplo(objname, objnumber, scenename, scenenumber):
    # Set parameters values
    ap_defaults = APDefaults()
    ap_defaults.leaf = 0.005
    ap_defaults.max_ransac_iters = 100
    ap_defaults.points_to_sample = 3
    ap_defaults.nearest_features_used = 4
    ap_defaults.simil_threshold = 0.1
    ap_defaults.inlier_threshold = 3
    ap_defaults.inlier_fraction = 0.8

    icp_detection_defaults = ICPDefaults()
    icp_detection_defaults.euc_fit = 1e-5
    icp_detection_defaults.max_corr_dist = 0.8
    icp_detection_defaults.max_iter = 50
    icp_detection_defaults.transf_epsilon = 1e-5

    icp_finder_defaults = ICPDefaults()
    icp_finder_defaults.euc_fit = 1e-5
    icp_finder_defaults.max_corr_dist = 0.5
    icp_finder_defaults.max_iter = 50
    icp_finder_defaults.transf_epsilon = 1e-5

    det_umbral_score = 1e-3
    det_obj_scene_leaf = 0.005
    det_perc_obj_model_points = 0.5

    find_umbral_score = 1e-4
    find_obj_scene_leaf = 0.002
    find_perc_obj_model_points = 0.3

    # Create objects
    img_provider = FrameNamesAndImageProvider(
        'videos/rgbd/scenes/', scenename, scenenumber,
        'videos/rgbd/objs/', objname, objnumber,
    )  # path, objname, number

    detector = AutomaticDetection(
        ap_defaults=ap_defaults,
        icp_defaults=icp_detection_defaults,
        umbral_score=det_umbral_score,
        obj_scene_leaf=det_obj_scene_leaf,
        perc_obj_model_points=det_perc_obj_model_points,
    )

    finder = ICPFinderWithModel(
        icp_defaults=icp_finder_defaults,
        umbral_score=find_umbral_score,
        obj_scene_leaf=find_obj_scene_leaf,
        perc_obj_model_points=find_perc_obj_model_points,
    )

    follower = FollowerStaticICPAndObjectModel(img_provider, detector, finder)

    FollowingSchemeSavingData(
        img_provider,
        follower,
        'pruebas_guardadas'
    ).run()


def barrer_find_percentage_object(objname, objnumber, scenename, scenenumber):
    # Set parameters values
    ap_defaults = APDefaults()
    ap_defaults.leaf = 0.005
    ap_defaults.max_ransac_iters = 100
    ap_defaults.points_to_sample = 3
    ap_defaults.nearest_features_used = 4
    ap_defaults.simil_threshold = 0.1
    ap_defaults.inlier_threshold = 3
    ap_defaults.inlier_fraction = 0.8

    icp_detection_defaults = ICPDefaults()
    icp_detection_defaults.euc_fit = 1e-5
    icp_detection_defaults.max_corr_dist = 0.8
    icp_detection_defaults.max_iter = 50
    icp_detection_defaults.transf_epsilon = 1e-5

    icp_finder_defaults = ICPDefaults()
    icp_finder_defaults.euc_fit = 1e-5
    icp_finder_defaults.max_corr_dist = 0.5
    icp_finder_defaults.max_iter = 50
    icp_finder_defaults.transf_epsilon = 1e-5

    det_umbral_score = 1e-3
    det_adapt_leaf = AdaptLeafRatio()
    det_obj_scene_leaf = 0.005
    det_perc_obj_model_points = 0.5
    det_obj_mult = 2

    find_umbral_score = 1e-4
    find_adapt_area = AdaptSearchArea()
    find_adapt_leaf = AdaptLeafRatio()
    find_obj_scene_leaf = 0.002
    find_perc_obj_model_points = 0.3

    # Create objects
    img_provider = FrameNamesAndImageProviderPreCharged(
        'videos/rgbd/scenes/', scenename, scenenumber,
        'videos/rgbd/objs/', objname, objnumber,
    )  # path, objname, number

    # Repetir 3 veces para evitar detecciones fallidas por RANSAC
    for i in range(3):
        for find_perc_obj_model_points in [0.2, 0.4, 0.5, 0.75]:
            detector = AutomaticDetection(
                ap_defaults=ap_defaults,
                icp_defaults=icp_detection_defaults,
                umbral_score=det_umbral_score,
                adapt_leaf=det_adapt_leaf,
                first_leaf_size=det_obj_scene_leaf,
                perc_obj_model_points=det_perc_obj_model_points,
                obj_mult=det_obj_mult,
            )

            finder = ICPFinderWithModel(
                icp_defaults=icp_finder_defaults,
                umbral_score=find_umbral_score,
                adapt_area=find_adapt_area,
                adapt_leaf=find_adapt_leaf,
                first_leaf_size=find_obj_scene_leaf,
                perc_obj_model_points=find_perc_obj_model_points,
            )

            follower = FollowerStaticICPAndObjectModel(
                img_provider,
                detector,
                finder
            )

            FollowingSquemaExploringParameter(
                img_provider,
                follower,
                'pruebas_guardadas',
                'find_perc_obj_model_points',
                find_perc_obj_model_points,
            ).run()

            img_provider.restart()


def barrer_detection_frame_size(objname, objnumber, scenename, scenenumber):
    # Set parameters values
    ap_defaults = APDefaults()
    ap_defaults.leaf = 0.005
    ap_defaults.max_ransac_iters = 100
    ap_defaults.points_to_sample = 3
    ap_defaults.nearest_features_used = 4
    ap_defaults.simil_threshold = 0.1
    ap_defaults.inlier_threshold = 3
    ap_defaults.inlier_fraction = 0.8

    icp_detection_defaults = ICPDefaults()
    icp_detection_defaults.euc_fit = 1e-5
    icp_detection_defaults.max_corr_dist = 0.8
    icp_detection_defaults.max_iter = 50
    icp_detection_defaults.transf_epsilon = 1e-5

    icp_finder_defaults = ICPDefaults()
    icp_finder_defaults.euc_fit = 1e-5
    icp_finder_defaults.max_corr_dist = 0.5
    icp_finder_defaults.max_iter = 50
    icp_finder_defaults.transf_epsilon = 1e-5

    det_umbral_score = 1e-3
    det_adapt_leaf = AdaptLeafRatio()
    det_obj_scene_leaf = 0.005
    det_perc_obj_model_points = 0.5
    det_obj_mult = 2

    find_umbral_score = 1e-4
    find_adapt_area = AdaptSearchArea()
    find_adapt_leaf = AdaptLeafRatio()
    find_obj_scene_leaf = 0.002
    find_perc_obj_model_points = 0.3

    # Create objects
    img_provider = FrameNamesAndImageProviderPreCharged(
        'videos/rgbd/scenes/', scenename, scenenumber,
        'videos/rgbd/objs/', objname, objnumber,
    )  # path, objname, number

    # Repetir 3 veces para evitar detecciones fallidas por RANSAC
    for i in range(3):
        for det_obj_mult in [2, 3, 5, 7]:
            detector = AutomaticDetection(
                ap_defaults=ap_defaults,
                icp_defaults=icp_detection_defaults,
                umbral_score=det_umbral_score,
                adapt_leaf=det_adapt_leaf,
                first_leaf_size=det_obj_scene_leaf,
                perc_obj_model_points=det_perc_obj_model_points,
                obj_mult=det_obj_mult,
            )

            finder = ICPFinderWithModel(
                icp_defaults=icp_finder_defaults,
                umbral_score=find_umbral_score,
                adapt_area=find_adapt_area,
                adapt_leaf=find_adapt_leaf,
                first_leaf_size=find_obj_scene_leaf,
                perc_obj_model_points=find_perc_obj_model_points,
            )

            follower = FollowerStaticICPAndObjectModel(
                img_provider,
                detector,
                finder
            )

            FollowingSquemaExploringParameter(
                img_provider,
                follower,
                'pruebas_guardadas',
                'detection_frame_size',
                det_obj_mult,
            ).run()

            img_provider.restart()


def barrer_inlier_fraction(objname, objnumber, scenename, scenenumber):
    # Set parameters values
    ap_defaults = APDefaults()
    ap_defaults.leaf = 0.005
    ap_defaults.max_ransac_iters = 100
    ap_defaults.points_to_sample = 3
    ap_defaults.nearest_features_used = 4
    ap_defaults.simil_threshold = 0.1
    ap_defaults.inlier_threshold = 3
    ap_defaults.inlier_fraction = 0.8

    icp_detection_defaults = ICPDefaults()
    icp_detection_defaults.euc_fit = 1e-5
    icp_detection_defaults.max_corr_dist = 0.8
    icp_detection_defaults.max_iter = 50
    icp_detection_defaults.transf_epsilon = 1e-5

    icp_finder_defaults = ICPDefaults()
    icp_finder_defaults.euc_fit = 1e-5
    icp_finder_defaults.max_corr_dist = 0.5
    icp_finder_defaults.max_iter = 50
    icp_finder_defaults.transf_epsilon = 1e-5

    det_umbral_score = 1e-3
    det_adapt_leaf = AdaptLeafRatio()
    det_obj_scene_leaf = 0.005
    det_perc_obj_model_points = 0.5
    det_obj_mult = 2

    find_umbral_score = 1e-4
    find_adapt_area = AdaptSearchArea()
    find_adapt_leaf = AdaptLeafRatio()
    find_obj_scene_leaf = 0.002
    find_perc_obj_model_points = 0.3

    # Create objects
    img_provider = FrameNamesAndImageProviderPreCharged(
        'videos/rgbd/scenes/', scenename, scenenumber,
        'videos/rgbd/objs/', objname, objnumber,
    )  # path, objname, number

    # Repetir 3 veces para evitar detecciones fallidas por RANSAC
    for i in range(3):
        for inlier_fraction in [0.2, 0.4, 0.6, 0.7, 0.9]:
            ap_defaults.inlier_fraction = inlier_fraction
            detector = AutomaticDetection(
                ap_defaults=ap_defaults,
                icp_defaults=icp_detection_defaults,
                umbral_score=det_umbral_score,
                adapt_leaf=det_adapt_leaf,
                first_leaf_size=det_obj_scene_leaf,
                perc_obj_model_points=det_perc_obj_model_points,
                obj_mult=det_obj_mult,
            )

            finder = ICPFinderWithModel(
                icp_defaults=icp_finder_defaults,
                umbral_score=find_umbral_score,
                adapt_area=find_adapt_area,
                adapt_leaf=find_adapt_leaf,
                first_leaf_size=find_obj_scene_leaf,
                perc_obj_model_points=find_perc_obj_model_points,
            )

            follower = FollowerStaticICPAndObjectModel(
                img_provider,
                detector,
                finder
            )

            FollowingSquemaExploringParameter(
                img_provider,
                follower,
                'pruebas_guardadas',
                'detection_inlier_fraction',
                inlier_fraction,
            ).run()

            img_provider.restart()


if __name__ == '__main__':
    # barrer_find_percentage_object('coffee_mug', '5', 'desk', '1')
    # barrer_find_percentage_object('cap', '4', 'desk', '1')
    # barrer_find_percentage_object('bowl', '3', 'desk', '2')

    # barrer_detection_frame_size('coffee_mug', '5', 'desk', '1')
    # barrer_detection_frame_size('cap', '4', 'desk', '1')
    # barrer_detection_frame_size('bowl', '3', 'desk', '2')

    #barrer_inlier_fraction('coffee_mug', '5', 'desk', '1')
    barrer_inlier_fraction('cap', '4', 'desk', '1')
    barrer_inlier_fraction('bowl', '3', 'desk', '2')


