#coding=utf-8

from __future__ import (unicode_literals, division)

from cpp.common import filter_cloud, show_clouds, get_min_max, save_pcd, points

from buscadores import Finder, ICPFinder, ICPFinderWithModel
from detectores import StaticDetector, StaticDetectorWithPCDFiltering, \
    StaticDetectorWithModelAlignment, AutomaticDetection
from esquemas_seguimiento import FollowingScheme, FollowingSchemeSavingData
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


if __name__ == '__main__':
    desk_2_bowl_3()
