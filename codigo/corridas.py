#coding=utf-8

from __future__ import (unicode_literals, division)

from cpp.alignment_prerejective import APDefaults
from cpp.icp import ICPDefaults

from buscadores import Finder, ICPFinder, ICPFinderWithModel
from metodos_comunes import AdaptLeafRatio, AdaptSearchArea, FixedSearchArea, \
    Timer
from detectores import StaticDetector, DepthStaticDetectorWithPCDFiltering, \
    StaticDetectorWithModelAlignment, DepthDetection, \
    StaticDepthTransformationDetection, SDTWithPostAlignment
from esquemas_seguimiento import FollowingScheme, FollowingSchemeSavingDataPCD, \
    FollowingSquemaExploringParameterPCD
from observar_seguimiento import MuestraSeguimientoEnVivo
from proveedores_de_imagenes import FrameNamesAndImageProvider, \
    FrameNamesAndImageProviderPreChargedForPCD, \
    SelectedFrameNamesAndImageProviderPreChargedForPCD
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
        'coffee_mug',
        '5',
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
        'coffee_mug',
        '5',
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
        'videos/rgbd/scenes/', 'desk', '1',
        'videos/rgbd/objs/', 'coffee_mug', '5',
    )  # path, objname, number

    detector = StaticDetectorWithModelAlignment(
        'videos/rgbd/scenes/desk/desk_1.mat',
        'coffee_mug',
        '5',
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

    detector = DepthDetection()

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

    detector = DepthDetection()

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

    detector = DepthDetection()

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

    detector = DepthDetection()

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

    detector = DepthDetection()

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

    detector = DepthDetection(
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
            detector = DepthDetection(
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
            detector = DepthDetection(
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
            detector = DepthDetection(
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
            detector = DepthDetection(
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
            detector = DepthDetection(
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
def barrer_find_euclidean_fitness(img_provider, scenename, scenenumber,
                                  objname, objnum):
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
                obj_rgbd_num=objnum,
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


def barrer_find_transformation_epsilon(img_provider, scenename, scenenumber,
                                       objname, objnum):
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
                obj_rgbd_num=objnum,
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


def barrer_find_correspondence_distance(img_provider, scenename, scenenumber,
                                        objname, objnum):
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
                obj_rgbd_num=objnum,
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


def barrer_find_perc_obj_model_points(img_provider, scenename, scenenumber,
                                      objname, objnum):
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
                obj_rgbd_num=objnum,
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


def barrer_find_umbral_score(img_provider, scenename, scenenumber,
                             objname, objnum):
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
                obj_rgbd_num=objnum,
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


def barrer_det_max_iter(img_provider, scenename, scenenumber,
                        objname, objnum):
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
                obj_rgbd_num=objnum,
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


def barrer_det_points_to_sample(img_provider, scenename, scenenumber,
                                objname, objnum):
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
                obj_rgbd_num=objnum,
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


def barrer_det_nearest_features(img_provider, scenename, scenenumber,
                                objname, objnum):
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
                obj_rgbd_num=objnum,
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


def barrer_det_simil_thresh(img_provider, scenename, scenenumber,
                            objname, objnum):
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
                obj_rgbd_num=objnum,
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


def barrer_det_inlier_thresh(img_provider, scenename, scenenumber,
                             objname, objnum):
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
                obj_rgbd_num=objnum,
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


def barrer_det_inlier_fraction(img_provider, scenename, scenenumber,
                               objname, objnum):
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
                obj_rgbd_num=objnum,
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


def definitivo_depth(img_provider, scenename, scenenumber,
                     objname, objnum):
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
    find_perc_obj_model_points = 0.6

    # Repetir N veces para minimizar detecciones fallidas por RANSAC
    for i in range(6):
        detector = StaticDetectorWithModelAlignment(
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


def prueba_deteccion_automatica_sola(img_provider, scenename, scenenumber,
                                     objname, objnum):
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
    find_perc_obj_model_points = 0.6

    # Repetir 3 veces para evitar detecciones fallidas por RANSAC
    for i in range(6):
        detector = StaticDetectorWithModelAlignment(
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


def prueba_deteccion_nadia_sola(img_provider, objname, objnum,
                                scenename, scenenumber):

    detector = StaticDepthTransformationDetection(
        transf_file_path='videos/rgbd/resultados_deteccion/con_ap',
        sname=scenename,
        snum=scenenumber,
        objname=objname,
        objnum=objnum,
    )

    finder = Finder()

    follower = DepthFollower(
        img_provider,
        detector,
        finder
    )

    FollowingSquemaExploringParameterPCD(
        img_provider,
        follower,
        'pruebas_guardadas',
        'solo_deteccion_nadia_con_ap',
        'deteccion',
    ).run()

    img_provider.restart()


def correr_modelo_entero_inlier_fraction(img_provider, objname, objnumber, scenename, scenenumber):
    """
    Deteccion de Nadia realineada con AP e ICP.
    revisando ap.inlier_fraction
    """
    # Set parameters values
    # Set detection parameters values
    ap_defaults = APDefaults()
    ap_defaults.leaf = 0.005
    ap_defaults.max_ransac_iters = 200
    ap_defaults.points_to_sample = 3
    ap_defaults.nearest_features_used = 4
    ap_defaults.simil_threshold = 0.05
    ap_defaults.inlier_threshold = 4
    ap_defaults.inlier_fraction = 0.05

    icp_detection_defaults = ICPDefaults()
    icp_detection_defaults.euc_fit = 1e-10
    icp_detection_defaults.max_corr_dist = 10
    icp_detection_defaults.max_iter = 50
    icp_detection_defaults.transf_epsilon = 1e-6

    icp_finder_defaults = ICPDefaults()
    icp_finder_defaults.euc_fit = 1e-5
    icp_finder_defaults.max_corr_dist = 0.5
    icp_finder_defaults.max_iter = 50
    icp_finder_defaults.transf_epsilon = 1e-5

    det_umbral_score = 1e-3
    det_obj_scene_leaf = 0.01
    det_perc_obj_model_points = 0.04

    find_umbral_score = 1e-4
    find_adapt_area = FixedSearchArea(3)
    find_adapt_leaf = AdaptLeafRatio()
    find_obj_scene_leaf = 0.008
    find_perc_obj_model_points = 0.03

    for ap_inlier_fraction in [0.01, 0.03, 0.05, 0.1, 0.3]:
        ap_defaults.inlier_fraction = ap_inlier_fraction
        # Repetir N veces para minimizar detecciones fallidas por RANSAC
        for i in range(1):
            detector = StaticDepthTransformationDetection(
                transf_file_path='videos/rgbd/resultados_deteccion',
                sname=scenename,
                snum=scenenumber,
                objname=objname,
                objnum=objnumber,
                first_leaf_size=det_obj_scene_leaf,
                perc_obj_model_points=det_perc_obj_model_points,
                ap_defaults=ap_defaults,
                icp_defaults=icp_detection_defaults,
                umbral_score=det_umbral_score,
            )

            finder = ICPFinderWithModel(
                icp_defaults=icp_finder_defaults,
                umbral_score=find_umbral_score,
                adapt_area=find_adapt_area,
                adapt_leaf=find_adapt_leaf,
                first_leaf_size=find_obj_scene_leaf,
                perc_obj_model_points=find_perc_obj_model_points,
            )

            follower = DepthFollower(img_provider, detector, finder)

            # show_following = MuestraSeguimientoEnVivo('Seguidor ICP')

            FollowingSquemaExploringParameterPCD(
                img_provider,
                follower,
                # show_following,
                'pruebas_guardadas',
                'ap_inlier_fraction',
                ap_inlier_fraction,
            ).run()

            img_provider.restart()


def correr_modelo_entero_inlier_threshold(img_provider, objname, objnumber, scenename, scenenumber):
    """
    Deteccion de Nadia realineada con AP e ICP
    """
    # Set parameters values
    # Set detection parameters values
    ap_defaults = APDefaults()
    ap_defaults.leaf = 0.005
    ap_defaults.max_ransac_iters = 200
    ap_defaults.points_to_sample = 3
    ap_defaults.nearest_features_used = 4
    ap_defaults.simil_threshold = 0.05
    ap_defaults.inlier_threshold = 4
    ap_defaults.inlier_fraction = 0.05

    icp_detection_defaults = ICPDefaults()
    icp_detection_defaults.euc_fit = 1e-10
    icp_detection_defaults.max_corr_dist = 10
    icp_detection_defaults.max_iter = 50
    icp_detection_defaults.transf_epsilon = 1e-6

    icp_finder_defaults = ICPDefaults()
    icp_finder_defaults.euc_fit = 1e-5
    icp_finder_defaults.max_corr_dist = 0.5
    icp_finder_defaults.max_iter = 50
    icp_finder_defaults.transf_epsilon = 1e-5

    det_umbral_score = 1e-3
    det_obj_scene_leaf = 0.01
    det_perc_obj_model_points = 0.04

    find_umbral_score = 1e-4
    find_adapt_area = FixedSearchArea(3)
    find_adapt_leaf = AdaptLeafRatio()
    find_obj_scene_leaf = 0.008
    find_perc_obj_model_points = 0.03

    for ap_inlier_threshold in [3, 6, 8, 10, 20]:
        ap_defaults.inlier_threshold = ap_inlier_threshold
        # Repetir N veces para minimizar detecciones fallidas por RANSAC
        for i in range(1):
            detector = SDTWithPostAlignment(
                transf_file_path='videos/rgbd/resultados_deteccion',
                sname=scenename,
                snum=scenenumber,
                objname=objname,
                objnum=objnumber,
                first_leaf_size=det_obj_scene_leaf,
                perc_obj_model_points=det_perc_obj_model_points,
                ap_defaults=ap_defaults,
                icp_defaults=icp_detection_defaults,
                umbral_score=det_umbral_score,
            )

            finder = ICPFinderWithModel(
                icp_defaults=icp_finder_defaults,
                umbral_score=find_umbral_score,
                adapt_area=find_adapt_area,
                adapt_leaf=find_adapt_leaf,
                first_leaf_size=find_obj_scene_leaf,
                perc_obj_model_points=find_perc_obj_model_points,
            )

            follower = DepthFollower(img_provider, detector, finder)

            # show_following = MuestraSeguimientoEnVivo('Seguidor ICP')

            FollowingSquemaExploringParameterPCD(
                img_provider,
                follower,
                # show_following,
                'pruebas_guardadas',
                'ap_inlier_threshold',
                ap_inlier_threshold,
            ).run()

            img_provider.restart()


def correr_modelo_entero_points_to_sample(img_provider, objname, objnumber, scenename, scenenumber):
    """
    Deteccion de Nadia realineada con AP e ICP
    """
    # Set parameters values
    # Set detection parameters values
    ap_defaults = APDefaults()
    ap_defaults.leaf = 0.005
    ap_defaults.max_ransac_iters = 200
    ap_defaults.points_to_sample = 3
    ap_defaults.nearest_features_used = 4
    ap_defaults.simil_threshold = 0.05
    ap_defaults.inlier_threshold = 4
    ap_defaults.inlier_fraction = 0.05

    icp_detection_defaults = ICPDefaults()
    icp_detection_defaults.euc_fit = 1e-10
    icp_detection_defaults.max_corr_dist = 10
    icp_detection_defaults.max_iter = 50
    icp_detection_defaults.transf_epsilon = 1e-6

    icp_finder_defaults = ICPDefaults()
    icp_finder_defaults.euc_fit = 1e-5
    icp_finder_defaults.max_corr_dist = 0.5
    icp_finder_defaults.max_iter = 50
    icp_finder_defaults.transf_epsilon = 1e-5

    det_umbral_score = 1e-3
    det_obj_scene_leaf = 0.01
    det_perc_obj_model_points = 0.04

    find_umbral_score = 1e-4
    find_adapt_area = FixedSearchArea(3)
    find_adapt_leaf = AdaptLeafRatio()
    find_obj_scene_leaf = 0.008
    find_perc_obj_model_points = 0.03

    for ap_points_to_sample in [5, 7, 19, 30]:
        ap_defaults.points_to_sample = ap_points_to_sample
        # Repetir N veces para minimizar detecciones fallidas por RANSAC
        for i in range(1):
            detector = SDTWithPostAlignment(
                transf_file_path='videos/rgbd/resultados_deteccion',
                sname=scenename,
                snum=scenenumber,
                objname=objname,
                objnum=objnumber,
                first_leaf_size=det_obj_scene_leaf,
                perc_obj_model_points=det_perc_obj_model_points,
                ap_defaults=ap_defaults,
                icp_defaults=icp_detection_defaults,
                umbral_score=det_umbral_score,
            )

            finder = ICPFinderWithModel(
                icp_defaults=icp_finder_defaults,
                umbral_score=find_umbral_score,
                adapt_area=find_adapt_area,
                adapt_leaf=find_adapt_leaf,
                first_leaf_size=find_obj_scene_leaf,
                perc_obj_model_points=find_perc_obj_model_points,
            )

            follower = DepthFollower(img_provider, detector, finder)

            # show_following = MuestraSeguimientoEnVivo('Seguidor ICP')

            FollowingSquemaExploringParameterPCD(
                img_provider,
                follower,
                # show_following,
                'pruebas_guardadas',
                'ap_points_to_sample',
                ap_points_to_sample,
            ).run()

            img_provider.restart()


def correr_modelo_entero_definitivo(img_provider, objname, objnumber,
                                    scenename, scenenumber):
    print "Corriendo para", objname, objnumber, scenename, scenenumber
    # Set parameters values
    # Set detection parameters values
    ap_defaults = APDefaults()
    ap_defaults.leaf = 0.005
    ap_defaults.max_ransac_iters = 300
    ap_defaults.points_to_sample = 30
    ap_defaults.nearest_features_used = 8
    ap_defaults.simil_threshold = 0.05
    ap_defaults.inlier_threshold = 0.8  # 80 cm
    ap_defaults.inlier_fraction = 0.05

    icp_detection_defaults = ICPDefaults()
    icp_detection_defaults.euc_fit = 1
    icp_detection_defaults.max_corr_dist = 0.1  # 10 cm
    icp_detection_defaults.max_iter = 50
    icp_detection_defaults.transf_epsilon = 1e-6

    icp_finder_defaults = ICPDefaults()
    icp_finder_defaults.euc_fit = 1e-5
    icp_finder_defaults.max_corr_dist = 0.5
    icp_finder_defaults.max_iter = 50
    icp_finder_defaults.transf_epsilon = 1e-5

    det_umbral_score = 1e-3
    det_obj_scene_leaf = 0.01
    det_perc_obj_model_points = 0.04

    find_umbral_score = 1e-4
    find_adapt_area = FixedSearchArea(3)
    find_adapt_leaf = AdaptLeafRatio()
    find_obj_scene_leaf = 0.008
    find_perc_obj_model_points = 0.03

    # Repetir N veces para minimizar detecciones fallidas por RANSAC
    for i in range(1):
        detector = SDTWithPostAlignment(
            transf_file_path='videos/rgbd/resultados_deteccion/con_ap',
            sname=scenename,
            snum=scenenumber,
            objname=objname,
            objnum=objnumber,
            first_leaf_size=det_obj_scene_leaf,
            perc_obj_model_points=det_perc_obj_model_points,
            ap_defaults=ap_defaults,
            icp_defaults=icp_detection_defaults,
            umbral_score=det_umbral_score,
        )

        finder = ICPFinderWithModel(
            icp_defaults=icp_finder_defaults,
            umbral_score=find_umbral_score,
            adapt_area=find_adapt_area,
            adapt_leaf=find_adapt_leaf,
            first_leaf_size=find_obj_scene_leaf,
            perc_obj_model_points=find_perc_obj_model_points,
        )

        follower = DepthFollower(img_provider, detector, finder)

        # show_following = MuestraSeguimientoEnVivo('Seguidor ICP')

        FollowingSquemaExploringParameterPCD(
            img_provider,
            follower,
            # show_following,
            'pruebas_guardadas',
            'probando_frames_parejos_con_ap',
            'parejos',
        ).run()

        img_provider.restart()



if __name__ == '__main__':
    #######################
    # CORRIDAS COMPUESTAS
    #######################
    # img_provider = SelectedFrameNamesAndImageProviderPreChargedForPCD(
    #     'videos/rgbd/scenes/', 'desk', '1',
    #     'videos/rgbd/objs/', 'coffee_mug', '5',
    # )  # path, objname, number
    # correr_modelo_entero_definitivo(img_provider, 'coffee_mug', '5', 'desk', '1')

    # img_provider.reinitialize_object('cap', '4')
    # correr_modelo_entero_definitivo(img_provider, 'cap', '4', 'desk', '1')
    #
    # img_provider = SelectedFrameNamesAndImageProviderPreChargedForPCD(
    #     'videos/rgbd/scenes/', 'desk', '2',
    #     'videos/rgbd/objs/', 'bowl', '3',
    # )  # path, objname, number
    # correr_modelo_entero_definitivo(img_provider, 'desk', '2', 'bowl', '3')
    #
    # img_provider = SelectedFrameNamesAndImageProviderPreChargedForPCD(
    #     'videos/rgbd/scenes/', 'table', '1',
    #     'videos/rgbd/objs/', 'coffee_mug', '1',
    # )  # path, objname, number
    # correr_modelo_entero_definitivo(img_provider, 'coffee_mug', '1', 'table', '1')
    # img_provider.reinitialize_object('soda_can', '4')
    # correr_modelo_entero_definitivo(img_provider, 'soda_can', '4', 'table', '1')
    #
    # img_provider = SelectedFrameNamesAndImageProviderPreChargedForPCD(
    #     'videos/rgbd/scenes/', 'table_small', '2',
    #     'videos/rgbd/objs/', 'cereal_box', '4',
    # )  # path, objname, number
    # correr_modelo_entero_definitivo(img_provider, 'table_small', '2', 'cereal_box', '4')

    #################################
    # CORRIDAS SOLO DETECCION NADIA
    #################################
    img_provider = SelectedFrameNamesAndImageProviderPreChargedForPCD(
        'videos/rgbd/scenes/', 'desk', '1',
        'videos/rgbd/objs/', 'coffee_mug', '5',
    )  # path, objname, number
    prueba_deteccion_nadia_sola(img_provider, 'coffee_mug', '5', 'desk', '1')
    correr_modelo_entero_definitivo(img_provider, 'coffee_mug', '5', 'desk', '1')

    img_provider.reinitialize_object('cap', '4')
    prueba_deteccion_nadia_sola(img_provider, 'cap', '4', 'desk', '1')
    correr_modelo_entero_definitivo(img_provider, 'cap', '4', 'desk', '1')

    img_provider.reinitialize_object('soda_can', '6')
    prueba_deteccion_nadia_sola(img_provider, 'soda_can', '6', 'desk', '1')
    correr_modelo_entero_definitivo(img_provider, 'soda_can', '6', 'desk', '1')



    img_provider = SelectedFrameNamesAndImageProviderPreChargedForPCD(
        'videos/rgbd/scenes/', 'desk', '2',
        'videos/rgbd/objs/', 'bowl', '3',
    )  # path, objname, number
    prueba_deteccion_nadia_sola(img_provider, 'bowl', '3', 'desk', '2')
    correr_modelo_entero_definitivo(img_provider, 'bowl', '3', 'desk', '2')

    img_provider.reinitialize_object('soda_can', '4')
    prueba_deteccion_nadia_sola(img_provider, 'soda_can', '4', 'desk', '2')
    correr_modelo_entero_definitivo(img_provider, 'soda_can', '4', 'desk', '2')



    img_provider = SelectedFrameNamesAndImageProviderPreChargedForPCD(
        'videos/rgbd/scenes/', 'table', '1',
        'videos/rgbd/objs/', 'coffee_mug', '1',
    )  # path, objname, number

    prueba_deteccion_nadia_sola(img_provider, 'coffee_mug', '1', 'table', '1')
    correr_modelo_entero_definitivo(img_provider, 'coffee_mug', '1', 'table', '1')
    img_provider.reinitialize_object('coffee_mug', '4')
    prueba_deteccion_nadia_sola(img_provider, 'coffee_mug', '4', 'table', '1')
    correr_modelo_entero_definitivo(img_provider, 'coffee_mug', '4', 'table', '1')
    img_provider.reinitialize_object('bowl', '2')
    prueba_deteccion_nadia_sola(img_provider, 'bowl', '2', 'table', '1')
    correr_modelo_entero_definitivo(img_provider, 'bowl', '2', 'table', '1')
    img_provider.reinitialize_object('cap', '1')
    prueba_deteccion_nadia_sola(img_provider, 'cap', '1', 'table', '1')
    correr_modelo_entero_definitivo(img_provider, 'cap', '1', 'table', '1')
    img_provider.reinitialize_object('cap', '4')
    prueba_deteccion_nadia_sola(img_provider, 'cap', '4', 'table', '1')
    correr_modelo_entero_definitivo(img_provider, 'cap', '4', 'table', '1')
    img_provider.reinitialize_object('cereal_box', '4')
    prueba_deteccion_nadia_sola(img_provider, 'cereal_box', '4', 'table', '1')
    correr_modelo_entero_definitivo(img_provider, 'cereal_box', '4', 'table', '1')
    img_provider.reinitialize_object('soda_can', '4')
    prueba_deteccion_nadia_sola(img_provider, 'soda_can', '4', 'table', '1')
    correr_modelo_entero_definitivo(img_provider, 'soda_can', '4', 'table', '1')
