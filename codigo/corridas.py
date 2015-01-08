#coding=utf-8

from __future__ import (unicode_literals, division)

from cpp.alignment_prerejective import APDefaults
from cpp.icp import ICPDefaults

from buscadores import Finder, ICPFinder, ICPFinderWithModel
from metodos_comunes import AdaptLeafRatio, AdaptSearchArea, FixedSearchArea
from detectores import StaticDetector, DepthStaticDetectorWithPCDFiltering, \
    StaticDetectorWithModelAlignment, AutomaticDetection
from esquemas_seguimiento import FollowingScheme, FollowingSchemeSavingDataPCD, \
    FollowingSquemaExploringParameterPCD
from observar_seguimiento import MuestraSeguimientoEnVivo
from proveedores_de_imagenes import FrameNamesAndImageProvider, \
    FrameNamesAndImageProviderPreChargedForPCD
from seguidores import Follower, DepthFollower


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

    follower = Follower(img_provider, detector, finder)

    show_following = MuestraSeguimientoEnVivo(
        'Deteccion estatica - Sin seguidor'
    )

    FollowingScheme(
        img_provider,
        follower,
        show_following,
    ).run()


def icp_sin_modelo():
    img_provider = FrameNamesAndImageProviderPreChargedForPCD(
        'videos/rgbd/scenes/',  # scene path
        'desk',  # scene
        '1',  # scene number
        'videos/rgbd/objs/',  # object path
        'coffee_mug',  # object
        '5',  # object number
    )

    detector = DepthStaticDetectorWithPCDFiltering(
        'videos/rgbd/scenes/desk/desk_1.mat',
        'coffee_mug'
    )

    finder = ICPFinder()

    follower = DepthFollower(img_provider, detector, finder)

    show_following = MuestraSeguimientoEnVivo('Seguidor ICP')

    FollowingScheme(
        img_provider,
        follower,
        show_following,
    ).run()


def icp_con_modelo():
    img_provider = FrameNamesAndImageProviderPreChargedForPCD(
        'videos/rgbd/scenes/', 'desk', '1', 'videos/rgbd/objs/', 'coffee_mug', '5',
    )  # path, objname, number

    detector = StaticDetectorWithModelAlignment(
        'videos/rgbd/scenes/desk/desk_1.mat',
        'coffee_mug'
    )

    finder = ICPFinderWithModel()

    follower = DepthFollower(img_provider, detector, finder)

    show_following = MuestraSeguimientoEnVivo('Seguidor ICP')

    FollowingScheme(
        img_provider,
        follower,
        show_following,
    ).run()


def deteccion_automatica_icp_con_modelo():
    img_provider = FrameNamesAndImageProviderPreChargedForPCD(
        'videos/rgbd/scenes/', 'desk', '1',
        'videos/rgbd/objs/', 'coffee_mug', '5',
    )  # path, objname, number

    detector = AutomaticDetection()

    finder = ICPFinderWithModel()

    follower = DepthFollower(img_provider, detector, finder)

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

    follower = DepthFollower(img_provider, detector, finder)

    FollowingSchemeSavingDataPCD(
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

    follower = DepthFollower(img_provider, detector, finder)

    FollowingSchemeSavingDataPCD(
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

    follower = DepthFollower(img_provider, detector, finder)

    FollowingSchemeSavingDataPCD(
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

    follower = DepthFollower(img_provider, detector, finder)

    FollowingSchemeSavingDataPCD(
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

    follower = DepthFollower(img_provider, detector, finder)

    FollowingSchemeSavingDataPCD(
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
    det_obj_scene_leaf = 0.005
    det_perc_obj_model_points = 0.5
    det_adapt_leaf = AdaptLeafRatio(first_leaf=det_obj_scene_leaf)
    det_obj_mult = 2

    find_umbral_score = 1e-4
    find_adapt_area = AdaptSearchArea()
    find_obj_scene_leaf = 0.004
    find_perc_obj_model_points = 0.3
    find_adapt_leaf = AdaptLeafRatio(first_leaf=find_obj_scene_leaf)

    # Create objects
    img_provider = FrameNamesAndImageProviderPreChargedForPCD(
        'videos/rgbd/scenes/', scenename, scenenumber,
        'videos/rgbd/objs/', objname, objnumber,
    )  # path, objname, number

    # Repetir 3 veces para evitar detecciones fallidas por RANSAC
    for i in range(3):
        for find_perc_obj_model_points in [0.1, 0.3, 0.6, 0.8, 0.9]:#[0.2, 0.4, 0.5, 0.75]:
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

            follower = DepthFollower(
                img_provider,
                detector,
                finder
            )

            FollowingSquemaExploringParameterPCD(
                img_provider,
                follower,
                'pruebas_guardadas',
                'find_perc_obj_model_points',
                find_perc_obj_model_points,
            ).run()

            img_provider.restart()


def barrer_detection_frame_size(objname, objnumber, scenename, scenenumber):
    # TODO: correr con más valores (1, 1.5, 4, 6, 8)
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
    det_adapt_leaf = AdaptLeafRatio(first_leaf=det_obj_scene_leaf)
    det_obj_mult = 2

    find_umbral_score = 1e-4
    find_adapt_area = AdaptSearchArea()
    find_obj_scene_leaf = 0.004
    find_perc_obj_model_points = 0.3
    find_adapt_leaf = AdaptLeafRatio(first_leaf=find_obj_scene_leaf)

    # Create objects
    img_provider = FrameNamesAndImageProviderPreChargedForPCD(
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

            follower = DepthFollower(
                img_provider,
                detector,
                finder
            )

            FollowingSquemaExploringParameterPCD(
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
    det_obj_scene_leaf = 0.005
    det_perc_obj_model_points = 0.5
    det_adapt_leaf = AdaptLeafRatio(first_leaf=det_obj_scene_leaf)
    det_obj_mult = 2

    find_umbral_score = 1e-4
    find_adapt_area = AdaptSearchArea()
    find_obj_scene_leaf = 0.004
    find_perc_obj_model_points = 0.3
    find_adapt_leaf = AdaptLeafRatio(first_leaf=find_obj_scene_leaf)

    # Create objects
    img_provider = FrameNamesAndImageProviderPreChargedForPCD(
        'videos/rgbd/scenes/', scenename, scenenumber,
        'videos/rgbd/objs/', objname, objnumber,
    )  # path, objname, number

    # Repetir 3 veces para evitar detecciones fallidas por RANSAC
    for i in range(1):
        for inlier_fraction in [0.7, 0.9, 0.2, 0.4, 0.6, 0.7, 0.9]:
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

            follower = DepthFollower(
                img_provider,
                detector,
                finder
            )

            FollowingSquemaExploringParameterPCD(
                img_provider,
                follower,
                'pruebas_guardadas',
                'detection_inlier_fraction',
                inlier_fraction,
            ).run()

            img_provider.restart()


def barrer_similarity_threshold(objname, objnumber, scenename, scenenumber):
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
    img_provider = FrameNamesAndImageProviderPreChargedForPCD(
        'videos/rgbd/scenes/', scenename, scenenumber,
        'videos/rgbd/objs/', objname, objnumber,
    )  # path, objname, number

    # Repetir 3 veces para evitar detecciones fallidas por RANSAC
    for i in range(3):
        for simil_threshold in [0.2, 0.4, 0.6, 0.7, 0.9]:
            ap_defaults.simil_threshold = simil_threshold
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

            follower = DepthFollower(
                img_provider,
                detector,
                finder
            )

            FollowingSquemaExploringParameterPCD(
                img_provider,
                follower,
                'pruebas_guardadas',
                'detection_similarity_threshold',
                simil_threshold,
            ).run()

            img_provider.restart()


def barrer_fixed_search_area(objname, objnumber, scenename, scenenumber):
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
    find_adapt_area = FixedSearchArea()
    find_adapt_leaf = AdaptLeafRatio()
    find_obj_scene_leaf = 0.002
    find_perc_obj_model_points = 0.3

    # Create objects
    img_provider = FrameNamesAndImageProviderPreChargedForPCD(
        'videos/rgbd/scenes/', scenename, scenenumber,
        'videos/rgbd/objs/', objname, objnumber,
    )  # path, objname, number

    # Repetir 3 veces para evitar detecciones fallidas por RANSAC
    for i in range(3):
        for obj_size_times in [1.5, 2, 3, 4]:
            find_adapt_area = FixedSearchArea(obj_size_times=obj_size_times)
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

            follower = DepthFollower(
                img_provider,
                detector,
                finder
            )

            FollowingSquemaExploringParameterPCD(
                img_provider,
                follower,
                'pruebas_guardadas',
                'find_fixed_search_area',
                find_adapt_area.obj_size_times + 1,
            ).run()

            img_provider.restart()


######################################
# Corridas con la deteccion estatica
######################################
def barrer_find_euclidean_fitness(img_provider, scenename, scenenumber, objname):
    # Set detection parameters values
    ap_defaults = APDefaults()
    ap_defaults.leaf = 0.005
    ap_defaults.max_ransac_iters = 100
    ap_defaults.points_to_sample = 3
    ap_defaults.nearest_features_used = 2
    ap_defaults.simil_threshold = 0.4
    ap_defaults.inlier_threshold = 1.5
    ap_defaults.inlier_fraction = 0.7

    icp_detection_defaults = ICPDefaults()
    icp_detection_defaults.euc_fit = 1e-15
    icp_detection_defaults.max_corr_dist = 3
    icp_detection_defaults.max_iter = 50
    icp_detection_defaults.transf_epsilon = 1e-15

    det_umbral_score = 1e-3
    det_obj_scene_leaf = 0.005
    det_perc_obj_model_points = 0.5

    # Set following parameters values
    icp_finder_defaults = ICPDefaults()
    icp_finder_defaults.euc_fit = 1e-5
    icp_finder_defaults.max_corr_dist = 0.5
    icp_finder_defaults.max_iter = 50
    icp_finder_defaults.transf_epsilon = 1e-5

    find_umbral_score = 1e-4
    find_adapt_area = FixedSearchArea(3)
    find_adapt_leaf = AdaptLeafRatio()
    find_obj_scene_leaf = 0.002
    find_perc_obj_model_points = 0.5

    # Repetir 3 veces para evitar detecciones fallidas por RANSAC
    for i in range(3):
        for find_euclidean_fitness in [1e-15, 1e-10, 1e-8, 1e-5, 1e-3, 1e-2, 1e-1, 1, 1.5]:
            icp_finder_defaults.euc_fit = find_euclidean_fitness
            detector = StaticDetectorWithModelAlignment(
                matfile_path=('videos/rgbd/scenes/{sname}/{sname}_{snum}.mat'
                          .format(sname=scenename, snum=scenenumber)),
                obj_rgbd_name=objname,
                ap_defaults=ap_defaults,
                icp_defaults=icp_detection_defaults,
                leaf_size=det_obj_scene_leaf,
                icp_threshold=det_umbral_score,
                perc_obj_model_pts=det_perc_obj_model_points
            )

            finder = ICPFinderWithModel(
                icp_defaults=icp_finder_defaults,
                umbral_score=find_umbral_score,
                adapt_area=find_adapt_area,
                adapt_leaf=find_adapt_leaf,
                first_leaf_size=find_obj_scene_leaf,
                perc_obj_model_points=find_perc_obj_model_points,
            )

            follower = DepthFollower(
                img_provider,
                detector,
                finder
            )

            FollowingSquemaExploringParameterPCD(
                img_provider,
                follower,
                'pruebas_guardadas',
                'DEPTH_find_euclidean_fitness',
                find_euclidean_fitness,
            ).run()

            img_provider.restart()


def barrer_find_transformation_epsilon(img_provider, scenename, scenenumber, objname):
    # Set detection parameters values
    ap_defaults = APDefaults()
    ap_defaults.leaf = 0.005
    ap_defaults.max_ransac_iters = 100
    ap_defaults.points_to_sample = 3
    ap_defaults.nearest_features_used = 2
    ap_defaults.simil_threshold = 0.4
    ap_defaults.inlier_threshold = 1.5
    ap_defaults.inlier_fraction = 0.7

    icp_detection_defaults = ICPDefaults()
    icp_detection_defaults.euc_fit = 1e-15
    icp_detection_defaults.max_corr_dist = 3
    icp_detection_defaults.max_iter = 50
    icp_detection_defaults.transf_epsilon = 1e-15

    det_umbral_score = 1e-3
    det_obj_scene_leaf = 0.005
    det_perc_obj_model_points = 0.5

    # Set following parameters values
    icp_finder_defaults = ICPDefaults()
    icp_finder_defaults.euc_fit = 1e-5
    icp_finder_defaults.max_corr_dist = 0.5
    icp_finder_defaults.max_iter = 50
    icp_finder_defaults.transf_epsilon = 1e-5

    find_umbral_score = 1e-4
    find_adapt_area = FixedSearchArea(3)
    find_adapt_leaf = AdaptLeafRatio()
    find_obj_scene_leaf = 0.002
    find_perc_obj_model_points = 0.5

    # Repetir 3 veces para evitar detecciones fallidas por RANSAC
    for i in range(3):
        for find_transformation_epsilon in [1e-15, 1e-12, 1e-10, 1e-8, 1e-6, 1e-5, 1e-3, 1e-2, 1e-1]:
            icp_finder_defaults.transf_epsilon = find_transformation_epsilon
            detector = StaticDetectorWithModelAlignment(
                matfile_path=('videos/rgbd/scenes/{sname}/{sname}_{snum}.mat'
                          .format(sname=scenename, snum=scenenumber)),
                obj_rgbd_name=objname,
                ap_defaults=ap_defaults,
                icp_defaults=icp_detection_defaults,
                leaf_size=det_obj_scene_leaf,
                icp_threshold=det_umbral_score,
                perc_obj_model_pts=det_perc_obj_model_points
            )

            finder = ICPFinderWithModel(
                icp_defaults=icp_finder_defaults,
                umbral_score=find_umbral_score,
                adapt_area=find_adapt_area,
                adapt_leaf=find_adapt_leaf,
                first_leaf_size=find_obj_scene_leaf,
                perc_obj_model_points=find_perc_obj_model_points,
            )

            follower = DepthFollower(
                img_provider,
                detector,
                finder
            )

            FollowingSquemaExploringParameterPCD(
                img_provider,
                follower,
                'pruebas_guardadas',
                'DEPTH_find_transformation_epsilon',
                find_transformation_epsilon,
            ).run()

            img_provider.restart()


def barrer_find_correspondence_distance(img_provider, scenename, scenenumber, objname):
    # Set detection parameters values
    ap_defaults = APDefaults()
    ap_defaults.leaf = 0.005
    ap_defaults.max_ransac_iters = 100
    ap_defaults.points_to_sample = 3
    ap_defaults.nearest_features_used = 2
    ap_defaults.simil_threshold = 0.4
    ap_defaults.inlier_threshold = 1.5
    ap_defaults.inlier_fraction = 0.7

    icp_detection_defaults = ICPDefaults()
    icp_detection_defaults.euc_fit = 1e-15
    icp_detection_defaults.max_corr_dist = 3
    icp_detection_defaults.max_iter = 50
    icp_detection_defaults.transf_epsilon = 1e-15

    det_umbral_score = 1e-3
    det_obj_scene_leaf = 0.005
    det_perc_obj_model_points = 0.5

    # Set following parameters values
    icp_finder_defaults = ICPDefaults()
    icp_finder_defaults.euc_fit = 1e-5
    icp_finder_defaults.max_corr_dist = 0.5
    icp_finder_defaults.max_iter = 50
    icp_finder_defaults.transf_epsilon = 1e-5

    find_umbral_score = 1e-4
    find_adapt_area = FixedSearchArea(3)
    find_adapt_leaf = AdaptLeafRatio()
    find_obj_scene_leaf = 0.002
    find_perc_obj_model_points = 0.5

    # Repetir 3 veces para evitar detecciones fallidas por RANSAC
    for i in range(3):
        for find_correspondence_distance in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9]:
            icp_finder_defaults.max_corr_dist = find_correspondence_distance
            detector = StaticDetectorWithModelAlignment(
                matfile_path=('videos/rgbd/scenes/{sname}/{sname}_{snum}.mat'
                          .format(sname=scenename, snum=scenenumber)),
                obj_rgbd_name=objname,
                ap_defaults=ap_defaults,
                icp_defaults=icp_detection_defaults,
                leaf_size=det_obj_scene_leaf,
                icp_threshold=det_umbral_score,
                perc_obj_model_pts=det_perc_obj_model_points
            )

            finder = ICPFinderWithModel(
                icp_defaults=icp_finder_defaults,
                umbral_score=find_umbral_score,
                adapt_area=find_adapt_area,
                adapt_leaf=find_adapt_leaf,
                first_leaf_size=find_obj_scene_leaf,
                perc_obj_model_points=find_perc_obj_model_points,
            )

            follower = DepthFollower(
                img_provider,
                detector,
                finder
            )

            FollowingSquemaExploringParameterPCD(
                img_provider,
                follower,
                'pruebas_guardadas',
                'DEPTH_find_correspondence_distance',
                find_correspondence_distance,
            ).run()

            img_provider.restart()


def barrer_find_perc_obj_model_points(img_provider, scenename, scenenumber, objname):
    # Set detection parameters values
    ap_defaults = APDefaults()
    ap_defaults.leaf = 0.005
    ap_defaults.max_ransac_iters = 100
    ap_defaults.points_to_sample = 3
    ap_defaults.nearest_features_used = 2
    ap_defaults.simil_threshold = 0.4
    ap_defaults.inlier_threshold = 1.5
    ap_defaults.inlier_fraction = 0.7

    icp_detection_defaults = ICPDefaults()
    icp_detection_defaults.euc_fit = 1e-15
    icp_detection_defaults.max_corr_dist = 3
    icp_detection_defaults.max_iter = 50
    icp_detection_defaults.transf_epsilon = 1e-15

    det_umbral_score = 1e-3
    det_obj_scene_leaf = 0.005
    det_perc_obj_model_points = 0.5

    # Set following parameters values
    icp_finder_defaults = ICPDefaults()
    icp_finder_defaults.euc_fit = 1e-5
    icp_finder_defaults.max_corr_dist = 0.5
    icp_finder_defaults.max_iter = 50
    icp_finder_defaults.transf_epsilon = 1e-5

    find_umbral_score = 1e-4
    find_adapt_area = FixedSearchArea(3)
    find_adapt_leaf = AdaptLeafRatio()
    find_obj_scene_leaf = 0.002
    find_perc_obj_model_points = 0.5

    # Repetir 3 veces para evitar detecciones fallidas por RANSAC
    for i in range(3):
        for find_perc_obj_model_points in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
            detector = StaticDetectorWithModelAlignment(
                matfile_path=('videos/rgbd/scenes/{sname}/{sname}_{snum}.mat'
                          .format(sname=scenename, snum=scenenumber)),
                obj_rgbd_name=objname,
                ap_defaults=ap_defaults,
                icp_defaults=icp_detection_defaults,
                leaf_size=det_obj_scene_leaf,
                icp_threshold=det_umbral_score,
                perc_obj_model_pts=det_perc_obj_model_points
            )

            finder = ICPFinderWithModel(
                icp_defaults=icp_finder_defaults,
                umbral_score=find_umbral_score,
                adapt_area=find_adapt_area,
                adapt_leaf=find_adapt_leaf,
                first_leaf_size=find_obj_scene_leaf,
                perc_obj_model_points=find_perc_obj_model_points,
            )

            follower = DepthFollower(
                img_provider,
                detector,
                finder
            )

            FollowingSquemaExploringParameterPCD(
                img_provider,
                follower,
                'pruebas_guardadas',
                'DEPTH_find_perc_obj_model_points',
                find_perc_obj_model_points,
            ).run()

            img_provider.restart()


def barrer_find_umbral_score(img_provider, scenename, scenenumber, objname):
    # Set detection parameters values
    ap_defaults = APDefaults()
    ap_defaults.leaf = 0.005
    ap_defaults.max_ransac_iters = 100
    ap_defaults.points_to_sample = 3
    ap_defaults.nearest_features_used = 2
    ap_defaults.simil_threshold = 0.4
    ap_defaults.inlier_threshold = 1.5
    ap_defaults.inlier_fraction = 0.7

    icp_detection_defaults = ICPDefaults()
    icp_detection_defaults.euc_fit = 1e-15
    icp_detection_defaults.max_corr_dist = 3
    icp_detection_defaults.max_iter = 50
    icp_detection_defaults.transf_epsilon = 1e-15

    det_umbral_score = 1e-3
    det_obj_scene_leaf = 0.005
    det_perc_obj_model_points = 0.5

    # Set following parameters values
    icp_finder_defaults = ICPDefaults()
    icp_finder_defaults.euc_fit = 1e-5
    icp_finder_defaults.max_corr_dist = 0.5
    icp_finder_defaults.max_iter = 50
    icp_finder_defaults.transf_epsilon = 1e-5

    find_umbral_score = 1e-4
    find_adapt_area = FixedSearchArea(3)
    find_adapt_leaf = AdaptLeafRatio()
    find_obj_scene_leaf = 0.002
    find_perc_obj_model_points = 0.5

    # Repetir 3 veces para evitar detecciones fallidas por RANSAC
    for i in range(1):
        for find_umbral_score in [0.04, 1e-3, 1e-4]:
            detector = StaticDetectorWithModelAlignment(
                matfile_path=('videos/rgbd/scenes/{sname}/{sname}_{snum}.mat'
                          .format(sname=scenename, snum=scenenumber)),
                obj_rgbd_name=objname,
                ap_defaults=ap_defaults,
                icp_defaults=icp_detection_defaults,
                leaf_size=det_obj_scene_leaf,
                icp_threshold=det_umbral_score,
                perc_obj_model_pts=det_perc_obj_model_points
            )

            finder = ICPFinderWithModel(
                icp_defaults=icp_finder_defaults,
                umbral_score=find_umbral_score,
                adapt_area=find_adapt_area,
                adapt_leaf=find_adapt_leaf,
                first_leaf_size=find_obj_scene_leaf,
                perc_obj_model_points=find_perc_obj_model_points,
            )

            follower = DepthFollower(
                img_provider,
                detector,
                finder
            )

            FollowingSquemaExploringParameterPCD(
                img_provider,
                follower,
                'pruebas_guardadas',
                'DEPTH_find_umbral_score',
                find_umbral_score,
            ).run()

            img_provider.restart()


def barrer_det_max_iter(img_provider, scenename, scenenumber, objname):
    # Set detection parameters values
    ap_defaults = APDefaults()
    ap_defaults.leaf = 0.005
    ap_defaults.max_ransac_iters = 100
    ap_defaults.points_to_sample = 3
    ap_defaults.nearest_features_used = 2
    ap_defaults.simil_threshold = 0.4
    ap_defaults.inlier_threshold = 1.5
    ap_defaults.inlier_fraction = 0.7

    icp_detection_defaults = ICPDefaults()
    icp_detection_defaults.euc_fit = 1e-15
    icp_detection_defaults.max_corr_dist = 3
    icp_detection_defaults.max_iter = 50
    icp_detection_defaults.transf_epsilon = 1e-15

    det_umbral_score = 1e-3
    det_obj_scene_leaf = 0.005
    det_perc_obj_model_points = 0.5

    # Set following parameters values
    icp_finder_defaults = ICPDefaults()
    icp_finder_defaults.euc_fit = 1e-5
    icp_finder_defaults.max_corr_dist = 0.5
    icp_finder_defaults.max_iter = 50
    icp_finder_defaults.transf_epsilon = 1e-5

    find_umbral_score = 1e-4
    find_adapt_area = FixedSearchArea(3)
    find_adapt_leaf = AdaptLeafRatio()
    find_obj_scene_leaf = 0.002
    find_perc_obj_model_points = 0.5

    # Repetir 3 veces para evitar detecciones fallidas por RANSAC
    for i in range(1):
        for det_max_iter in [30, 40, 50, 80, 100, 120, 150]:
            ap_defaults.max_ransac_iters = det_max_iter
            detector = StaticDetectorWithModelAlignment(
                matfile_path=('videos/rgbd/scenes/{sname}/{sname}_{snum}.mat'
                          .format(sname=scenename, snum=scenenumber)),
                obj_rgbd_name=objname,
                ap_defaults=ap_defaults,
                icp_defaults=icp_detection_defaults,
                leaf_size=det_obj_scene_leaf,
                icp_threshold=det_umbral_score,
                perc_obj_model_pts=det_perc_obj_model_points
            )

            finder = ICPFinderWithModel(
                icp_defaults=icp_finder_defaults,
                umbral_score=find_umbral_score,
                adapt_area=find_adapt_area,
                adapt_leaf=find_adapt_leaf,
                first_leaf_size=find_obj_scene_leaf,
                perc_obj_model_points=find_perc_obj_model_points,
            )

            follower = DepthFollower(
                img_provider,
                detector,
                finder
            )

            FollowingSquemaExploringParameterPCD(
                img_provider,
                follower,
                'pruebas_guardadas',
                'DEPTH_det_max_iter',
                det_max_iter,
            ).run()

            img_provider.restart()


def barrer_det_points_to_sample(img_provider, scenename, scenenumber, objname):
    # Set detection parameters values
    ap_defaults = APDefaults()
    ap_defaults.leaf = 0.005
    ap_defaults.max_ransac_iters = 100
    ap_defaults.points_to_sample = 3
    ap_defaults.nearest_features_used = 2
    ap_defaults.simil_threshold = 0.4
    ap_defaults.inlier_threshold = 1.5
    ap_defaults.inlier_fraction = 0.7

    icp_detection_defaults = ICPDefaults()
    icp_detection_defaults.euc_fit = 1e-15
    icp_detection_defaults.max_corr_dist = 3
    icp_detection_defaults.max_iter = 50
    icp_detection_defaults.transf_epsilon = 1e-15

    det_umbral_score = 1e-3
    det_obj_scene_leaf = 0.005
    det_perc_obj_model_points = 0.5

    # Set following parameters values
    icp_finder_defaults = ICPDefaults()
    icp_finder_defaults.euc_fit = 1e-5
    icp_finder_defaults.max_corr_dist = 0.5
    icp_finder_defaults.max_iter = 50
    icp_finder_defaults.transf_epsilon = 1e-5

    find_umbral_score = 1e-4
    find_adapt_area = FixedSearchArea(3)
    find_adapt_leaf = AdaptLeafRatio()
    find_obj_scene_leaf = 0.002
    find_perc_obj_model_points = 0.5

    # Repetir 3 veces para evitar detecciones fallidas por RANSAC
    for i in range(1):
        for det_points_to_sample in [3, 4, 5, 6, 10, 15, 20, 30]:
            ap_defaults.points_to_sample = det_points_to_sample
            detector = StaticDetectorWithModelAlignment(
                matfile_path=('videos/rgbd/scenes/{sname}/{sname}_{snum}.mat'
                          .format(sname=scenename, snum=scenenumber)),
                obj_rgbd_name=objname,
                ap_defaults=ap_defaults,
                icp_defaults=icp_detection_defaults,
                leaf_size=det_obj_scene_leaf,
                icp_threshold=det_umbral_score,
                perc_obj_model_pts=det_perc_obj_model_points
            )

            finder = ICPFinderWithModel(
                icp_defaults=icp_finder_defaults,
                umbral_score=find_umbral_score,
                adapt_area=find_adapt_area,
                adapt_leaf=find_adapt_leaf,
                first_leaf_size=find_obj_scene_leaf,
                perc_obj_model_points=find_perc_obj_model_points,
            )

            follower = DepthFollower(
                img_provider,
                detector,
                finder
            )

            FollowingSquemaExploringParameterPCD(
                img_provider,
                follower,
                'pruebas_guardadas',
                'DEPTH_det_points_to_sample',
                det_points_to_sample,
            ).run()

            img_provider.restart()


def barrer_det_nearest_features(img_provider, scenename, scenenumber, objname):
    # Set detection parameters values
    ap_defaults = APDefaults()
    ap_defaults.leaf = 0.005
    ap_defaults.max_ransac_iters = 100
    ap_defaults.points_to_sample = 3
    ap_defaults.nearest_features_used = 2
    ap_defaults.simil_threshold = 0.4
    ap_defaults.inlier_threshold = 1.5
    ap_defaults.inlier_fraction = 0.7

    icp_detection_defaults = ICPDefaults()
    icp_detection_defaults.euc_fit = 1e-15
    icp_detection_defaults.max_corr_dist = 3
    icp_detection_defaults.max_iter = 50
    icp_detection_defaults.transf_epsilon = 1e-15

    det_umbral_score = 1e-3
    det_obj_scene_leaf = 0.005
    det_perc_obj_model_points = 0.5

    # Set following parameters values
    icp_finder_defaults = ICPDefaults()
    icp_finder_defaults.euc_fit = 1e-5
    icp_finder_defaults.max_corr_dist = 0.5
    icp_finder_defaults.max_iter = 50
    icp_finder_defaults.transf_epsilon = 1e-5

    find_umbral_score = 1e-4
    find_adapt_area = FixedSearchArea(3)
    find_adapt_leaf = AdaptLeafRatio()
    find_obj_scene_leaf = 0.002
    find_perc_obj_model_points = 0.5

    # Repetir 3 veces para evitar detecciones fallidas por RANSAC
    for i in range(1):
        for det_nearest_features in [2, 3, 4, 5, 10, 15, 20]:
            ap_defaults.nearest_features_used = det_nearest_features
            detector = StaticDetectorWithModelAlignment(
                matfile_path=('videos/rgbd/scenes/{sname}/{sname}_{snum}.mat'
                          .format(sname=scenename, snum=scenenumber)),
                obj_rgbd_name=objname,
                ap_defaults=ap_defaults,
                icp_defaults=icp_detection_defaults,
                leaf_size=det_obj_scene_leaf,
                icp_threshold=det_umbral_score,
                perc_obj_model_pts=det_perc_obj_model_points
            )

            finder = ICPFinderWithModel(
                icp_defaults=icp_finder_defaults,
                umbral_score=find_umbral_score,
                adapt_area=find_adapt_area,
                adapt_leaf=find_adapt_leaf,
                first_leaf_size=find_obj_scene_leaf,
                perc_obj_model_points=find_perc_obj_model_points,
            )

            follower = DepthFollower(
                img_provider,
                detector,
                finder
            )

            FollowingSquemaExploringParameterPCD(
                img_provider,
                follower,
                'pruebas_guardadas',
                'DEPTH_det_nearest_features',
                det_nearest_features,
            ).run()

            img_provider.restart()


def barrer_det_simil_thresh(img_provider, scenename, scenenumber, objname):
    # Set detection parameters values
    ap_defaults = APDefaults()
    ap_defaults.leaf = 0.005
    ap_defaults.max_ransac_iters = 100
    ap_defaults.points_to_sample = 3
    ap_defaults.nearest_features_used = 2
    ap_defaults.simil_threshold = 0.4
    ap_defaults.inlier_threshold = 1.5
    ap_defaults.inlier_fraction = 0.7

    icp_detection_defaults = ICPDefaults()
    icp_detection_defaults.euc_fit = 1e-15
    icp_detection_defaults.max_corr_dist = 3
    icp_detection_defaults.max_iter = 50
    icp_detection_defaults.transf_epsilon = 1e-15

    det_umbral_score = 1e-3
    det_obj_scene_leaf = 0.005
    det_perc_obj_model_points = 0.5

    # Set following parameters values
    icp_finder_defaults = ICPDefaults()
    icp_finder_defaults.euc_fit = 1e-5
    icp_finder_defaults.max_corr_dist = 0.5
    icp_finder_defaults.max_iter = 50
    icp_finder_defaults.transf_epsilon = 1e-5

    find_umbral_score = 1e-4
    find_adapt_area = FixedSearchArea(3)
    find_adapt_leaf = AdaptLeafRatio()
    find_obj_scene_leaf = 0.002
    find_perc_obj_model_points = 0.5

    # Repetir 3 veces para evitar detecciones fallidas por RANSAC
    for i in range(1):
        for det_simil_thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            ap_defaults.simil_threshold = det_simil_thresh
            detector = StaticDetectorWithModelAlignment(
                matfile_path=('videos/rgbd/scenes/{sname}/{sname}_{snum}.mat'
                          .format(sname=scenename, snum=scenenumber)),
                obj_rgbd_name=objname,
                ap_defaults=ap_defaults,
                icp_defaults=icp_detection_defaults,
                leaf_size=det_obj_scene_leaf,
                icp_threshold=det_umbral_score,
                perc_obj_model_pts=det_perc_obj_model_points
            )

            finder = ICPFinderWithModel(
                icp_defaults=icp_finder_defaults,
                umbral_score=find_umbral_score,
                adapt_area=find_adapt_area,
                adapt_leaf=find_adapt_leaf,
                first_leaf_size=find_obj_scene_leaf,
                perc_obj_model_points=find_perc_obj_model_points,
            )

            follower = DepthFollower(
                img_provider,
                detector,
                finder
            )

            FollowingSquemaExploringParameterPCD(
                img_provider,
                follower,
                'pruebas_guardadas',
                'DEPTH_det_simil_thresh',
                det_simil_thresh,
            ).run()

            img_provider.restart()


def barrer_det_inlier_thresh(img_provider, scenename, scenenumber, objname):
    # Set detection parameters values
    ap_defaults = APDefaults()
    ap_defaults.leaf = 0.005
    ap_defaults.max_ransac_iters = 100
    ap_defaults.points_to_sample = 3
    ap_defaults.nearest_features_used = 2
    ap_defaults.simil_threshold = 0.4
    ap_defaults.inlier_threshold = 1.5
    ap_defaults.inlier_fraction = 0.7

    icp_detection_defaults = ICPDefaults()
    icp_detection_defaults.euc_fit = 1e-15
    icp_detection_defaults.max_corr_dist = 3
    icp_detection_defaults.max_iter = 50
    icp_detection_defaults.transf_epsilon = 1e-15

    det_umbral_score = 1e-3
    det_obj_scene_leaf = 0.005
    det_perc_obj_model_points = 0.5

    # Set following parameters values
    icp_finder_defaults = ICPDefaults()
    icp_finder_defaults.euc_fit = 1e-5
    icp_finder_defaults.max_corr_dist = 0.5
    icp_finder_defaults.max_iter = 50
    icp_finder_defaults.transf_epsilon = 1e-5

    find_umbral_score = 1e-4
    find_adapt_area = FixedSearchArea(3)
    find_adapt_leaf = AdaptLeafRatio()
    find_obj_scene_leaf = 0.002
    find_perc_obj_model_points = 0.5

    # Repetir 3 veces para evitar detecciones fallidas por RANSAC
    for i in range(1):
        for det_inlier_thresh in [4, 5, 6]:
            ap_defaults.inlier_threshold = det_inlier_thresh
            detector = StaticDetectorWithModelAlignment(
                matfile_path=('videos/rgbd/scenes/{sname}/{sname}_{snum}.mat'
                          .format(sname=scenename, snum=scenenumber)),
                obj_rgbd_name=objname,
                ap_defaults=ap_defaults,
                icp_defaults=icp_detection_defaults,
                leaf_size=det_obj_scene_leaf,
                icp_threshold=det_umbral_score,
                perc_obj_model_pts=det_perc_obj_model_points
            )

            finder = ICPFinderWithModel(
                icp_defaults=icp_finder_defaults,
                umbral_score=find_umbral_score,
                adapt_area=find_adapt_area,
                adapt_leaf=find_adapt_leaf,
                first_leaf_size=find_obj_scene_leaf,
                perc_obj_model_points=find_perc_obj_model_points,
            )

            follower = DepthFollower(
                img_provider,
                detector,
                finder
            )

            FollowingSquemaExploringParameterPCD(
                img_provider,
                follower,
                'pruebas_guardadas',
                'DEPTH_det_inlier_thresh',
                det_inlier_thresh,
            ).run()

            img_provider.restart()


def barrer_det_inlier_fraction(img_provider, scenename, scenenumber, objname):
    # Set detection parameters values
    ap_defaults = APDefaults()
    ap_defaults.leaf = 0.005
    ap_defaults.max_ransac_iters = 100
    ap_defaults.points_to_sample = 3
    ap_defaults.nearest_features_used = 2
    ap_defaults.simil_threshold = 0.4
    ap_defaults.inlier_threshold = 1.5
    ap_defaults.inlier_fraction = 0.7

    icp_detection_defaults = ICPDefaults()
    icp_detection_defaults.euc_fit = 1e-15
    icp_detection_defaults.max_corr_dist = 3
    icp_detection_defaults.max_iter = 50
    icp_detection_defaults.transf_epsilon = 1e-15

    det_umbral_score = 1e-3
    det_obj_scene_leaf = 0.005
    det_perc_obj_model_points = 0.5

    # Set following parameters values
    icp_finder_defaults = ICPDefaults()
    icp_finder_defaults.euc_fit = 1e-5
    icp_finder_defaults.max_corr_dist = 0.5
    icp_finder_defaults.max_iter = 50
    icp_finder_defaults.transf_epsilon = 1e-5

    find_umbral_score = 1e-4
    find_adapt_area = FixedSearchArea(3)
    find_adapt_leaf = AdaptLeafRatio()
    find_obj_scene_leaf = 0.002
    find_perc_obj_model_points = 0.5

    # Repetir 3 veces para evitar detecciones fallidas por RANSAC
    for i in range(1):
        for det_inlier_fraction in [0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9]:
            ap_defaults.inlier_fraction = det_inlier_fraction
            detector = StaticDetectorWithModelAlignment(
                matfile_path=('videos/rgbd/scenes/{sname}/{sname}_{snum}.mat'
                          .format(sname=scenename, snum=scenenumber)),
                obj_rgbd_name=objname,
                ap_defaults=ap_defaults,
                icp_defaults=icp_detection_defaults,
                leaf_size=det_obj_scene_leaf,
                icp_threshold=det_umbral_score,
                perc_obj_model_pts=det_perc_obj_model_points
            )

            finder = ICPFinderWithModel(
                icp_defaults=icp_finder_defaults,
                umbral_score=find_umbral_score,
                adapt_area=find_adapt_area,
                adapt_leaf=find_adapt_leaf,
                first_leaf_size=find_obj_scene_leaf,
                perc_obj_model_points=find_perc_obj_model_points,
            )

            follower = DepthFollower(
                img_provider,
                detector,
                finder
            )

            FollowingSquemaExploringParameterPCD(
                img_provider,
                follower,
                'pruebas_guardadas',
                'DEPTH_det_inlier_fraction',
                det_inlier_fraction,
            ).run()

            img_provider.restart()


def definitivo_depth(img_provider, scenename, scenenumber, objname):
    # Set detection parameters values
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

    # Set following parameters values
    icp_finder_defaults = ICPDefaults()
    icp_finder_defaults.euc_fit = 1e-10
    icp_finder_defaults.max_corr_dist = 0.1
    icp_finder_defaults.max_iter = 50
    icp_finder_defaults.transf_epsilon = 1e-6

    find_umbral_score = 1e-4
    find_adapt_area = FixedSearchArea(3)
    find_adapt_leaf = AdaptLeafRatio()
    find_obj_scene_leaf = 0.002
    find_perc_obj_model_points = 0.4

    # Repetir 3 veces para evitar detecciones fallidas por RANSAC
    for i in range(3):
        detector = StaticDetectorWithModelAlignment(
            matfile_path=('videos/rgbd/scenes/{sname}/{sname}_{snum}.mat'
                          .format(sname=scenename, snum=scenenumber)),
            obj_rgbd_name=objname,
            ap_defaults=ap_defaults,
            icp_defaults=icp_detection_defaults,
            leaf_size=det_obj_scene_leaf,
            icp_threshold=det_umbral_score,
            perc_obj_model_pts=det_perc_obj_model_points
        )

        finder = ICPFinderWithModel(
            icp_defaults=icp_finder_defaults,
            umbral_score=find_umbral_score,
            adapt_area=find_adapt_area,
            adapt_leaf=find_adapt_leaf,
            first_leaf_size=find_obj_scene_leaf,
            perc_obj_model_points=find_perc_obj_model_points,
        )

        follower = DepthFollower(
            img_provider,
            detector,
            finder
        )

        FollowingSquemaExploringParameterPCD(
            img_provider,
            follower,
            'pruebas_guardadas',
            'definitivo_DEPTH',
            'DEFINITIVO',
        ).run()

        img_provider.restart()


if __name__ == '__main__':
    # barrer_detection_frame_size('coffee_mug', '5', 'desk', '1')  # 4 hs
    # barrer_detection_frame_size('cap', '4', 'desk', '1')  # 4.9 hs
    # barrer_detection_frame_size('bowl', '3', 'desk', '2')  # 15.83 hs
    #
    # barrer_inlier_fraction('coffee_mug', '5', 'desk', '1')  # ?? hs
    # barrer_inlier_fraction('cap', '4', 'desk', '1')  # ?? hs
    # barrer_inlier_fraction('bowl', '3', 'desk', '2')
    #
    # barrer_similarity_threshold('coffee_mug', '5', 'desk', '1')  # 6.9 hs
    # barrer_similarity_threshold('cap', '4', 'desk', '1')  # 8.94 hs
    # barrer_similarity_threshold('bowl', '3', 'desk', '2')
    #
    # barrer_find_percentage_object('coffee_mug', '5', 'desk', '1')
    # barrer_find_percentage_object('cap', '4', 'desk', '1')
    # barrer_find_percentage_object('bowl', '3', 'desk', '2')

    # barrer_fixed_search_area('coffee_mug', '5', 'desk', '1')
    # barrer_fixed_search_area('cap', '4', 'desk', '1')
    # barrer_fixed_search_area('bowl', '3', 'desk', '2')


    #####################
    # Cargo las imagenes
    #####################
    desk_1_img_provider = FrameNamesAndImageProviderPreChargedForPCD(
        'videos/rgbd/scenes/', 'desk', '1',
        'videos/rgbd/objs/', 'cap', '4',
    )  # path, objname, number

    desk_2_img_provider = FrameNamesAndImageProviderPreChargedForPCD(
        'videos/rgbd/scenes/', 'desk', '2',
        'videos/rgbd/objs/', 'bowl', '3',
    )  # path, objname, number


    # #######################
    # # barrer_find_euclidean_fitness
    # #######################
    # # Primer escena
    # desk_1_img_provider.reinitialize_object('coffee_mug', '5')
    # barrer_find_euclidean_fitness(desk_1_img_provider, 'desk', '1', 'coffee_mug')
    #
    # # Segunda escena
    # desk_1_img_provider.reinitialize_object('cap', '4')
    # barrer_find_euclidean_fitness(desk_1_img_provider, 'desk', '1', 'cap')
    #
    # # Tercer escena
    # barrer_find_euclidean_fitness(desk_2_img_provider, 'desk', '2', 'bowl')
    #
    #
    # #######################
    # # barrer_find_transformation_epsilon
    # #######################
    # # Primer escena
    # desk_1_img_provider.reinitialize_object('coffee_mug', '5')
    # barrer_find_transformation_epsilon(desk_1_img_provider, 'desk', '1', 'coffee_mug')
    #
    # # Segunda escena
    # desk_1_img_provider.reinitialize_object('cap', '4')
    # barrer_find_transformation_epsilon(desk_1_img_provider, 'desk', '1', 'cap')
    #
    # # Tercer escena
    # barrer_find_transformation_epsilon(desk_2_img_provider, 'desk', '2', 'bowl')

    #######################
    # barrer_find_correspondence_distance
    #######################
    # Primer escena
    # desk_1_img_provider.reinitialize_object('coffee_mug', '5')
    # barrer_find_correspondence_distance(desk_1_img_provider, 'desk', '1', 'coffee_mug')

    # Segunda escena
    # desk_1_img_provider.reinitialize_object('cap', '4')
    # barrer_find_correspondence_distance(desk_1_img_provider, 'desk', '1', 'cap')

    # Tercer escena
    # barrer_find_correspondence_distance(desk_2_img_provider, 'desk', '2', 'bowl')

    #######################
    # barrer_find_perc_obj_model_points
    #######################
    # # Primer escena
    # desk_1_img_provider.reinitialize_object('coffee_mug', '5')
    # barrer_find_perc_obj_model_points(desk_1_img_provider, 'desk', '1', 'coffee_mug')
    #
    # # Segunda escena
    # desk_1_img_provider.reinitialize_object('cap', '4')
    # barrer_find_perc_obj_model_points(desk_1_img_provider, 'desk', '1', 'cap')
    #
    # # Tercer escena
    # barrer_find_perc_obj_model_points(desk_2_img_provider, 'desk', '2', 'bowl')


    #######################
    # barrer_find_umbral_score
    #######################
    # # Primer escena
    # desk_1_img_provider.reinitialize_object('coffee_mug', '5')
    # barrer_find_umbral_score(desk_1_img_provider, 'desk', '1', 'coffee_mug')
    #
    # # Segunda escena
    # desk_1_img_provider.reinitialize_object('cap', '4')
    # barrer_find_umbral_score(desk_1_img_provider, 'desk', '1', 'cap')
    #
    # # Tercer escena
    # barrer_find_umbral_score(desk_2_img_provider, 'desk', '2', 'bowl')



    #######################
    # barrer_det_max_iter
    #######################
    # # Primer escena
    # desk_1_img_provider.reinitialize_object('coffee_mug', '5')
    # barrer_det_max_iter(desk_1_img_provider, 'desk', '1', 'coffee_mug')
    #
    # # Segunda escena
    # desk_1_img_provider.reinitialize_object('cap', '4')
    # barrer_det_max_iter(desk_1_img_provider, 'desk', '1', 'cap')
    #
    # # Tercer escena
    # barrer_det_max_iter(desk_2_img_provider, 'desk', '2', 'bowl')


    #######################
    # barrer_det_points_to_sample
    #######################
    # # Primer escena
    # desk_1_img_provider.reinitialize_object('coffee_mug', '5')
    # barrer_det_points_to_sample(desk_1_img_provider, 'desk', '1', 'coffee_mug')
    #
    # # Segunda escena
    # desk_1_img_provider.reinitialize_object('cap', '4')
    # barrer_det_points_to_sample(desk_1_img_provider, 'desk', '1', 'cap')
    #
    # # Tercer escena
    # barrer_det_points_to_sample(desk_2_img_provider, 'desk', '2', 'bowl')


    #######################
    # barrer_det_nearest_features
    #######################
    # # Primer escena
    # desk_1_img_provider.reinitialize_object('coffee_mug', '5')
    # barrer_det_nearest_features(desk_1_img_provider, 'desk', '1', 'coffee_mug')
    #
    # # Segunda escena
    # desk_1_img_provider.reinitialize_object('cap', '4')
    # barrer_det_nearest_features(desk_1_img_provider, 'desk', '1', 'cap')
    #
    # # Tercer escena
    # barrer_det_nearest_features(desk_2_img_provider, 'desk', '2', 'bowl')


    #######################
    # barrer_det_simil_thresh
    #######################
    # # Primer escena
    # desk_1_img_provider.reinitialize_object('coffee_mug', '5')
    # barrer_det_simil_thresh(desk_1_img_provider, 'desk', '1', 'coffee_mug')
    #
    # # Segunda escena
    # desk_1_img_provider.reinitialize_object('cap', '4')
    # barrer_det_simil_thresh(desk_1_img_provider, 'desk', '1', 'cap')
    #
    # # Tercer escena
    # barrer_det_simil_thresh(desk_2_img_provider, 'desk', '2', 'bowl')


    #######################
    # barrer_det_inlier_thresh
    #######################
    # # Primer escena
    # desk_1_img_provider.reinitialize_object('coffee_mug', '5')
    # barrer_det_inlier_thresh(desk_1_img_provider, 'desk', '1', 'coffee_mug')
    #
    # # Segunda escena
    # desk_1_img_provider.reinitialize_object('cap', '4')
    # barrer_det_inlier_thresh(desk_1_img_provider, 'desk', '1', 'cap')
    #
    # # Tercer escena
    # barrer_det_inlier_thresh(desk_2_img_provider, 'desk', '2', 'bowl')


    #######################
    # barrer_det_inlier_fraction
    #######################
    # # Primer escena
    # desk_1_img_provider.reinitialize_object('coffee_mug', '5')
    # barrer_det_inlier_fraction(desk_1_img_provider, 'desk', '1', 'coffee_mug')
    #
    # # Segunda escena
    # desk_1_img_provider.reinitialize_object('cap', '4')
    # barrer_det_inlier_fraction(desk_1_img_provider, 'desk', '1', 'cap')
    #
    # # Tercer escena
    # barrer_det_inlier_fraction(desk_2_img_provider, 'desk', '2', 'bowl')


    #######################
    # definitivo_depth
    #######################
    # Primer escena
    desk_1_img_provider.reinitialize_object('coffee_mug', '5')
    definitivo_depth(desk_1_img_provider, 'desk', '1', 'coffee_mug')

    # Segunda escena
    desk_1_img_provider.reinitialize_object('cap', '4')
    definitivo_depth(desk_1_img_provider, 'desk', '1', 'cap')

    # Tercer escena
    definitivo_depth(desk_2_img_provider, 'desk', '2', 'bowl')

