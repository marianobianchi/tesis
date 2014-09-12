#coding=utf-8

from __future__ import (unicode_literals, division)

from cpp.my_pcl import filter_cloud, show_clouds, get_min_max

from buscadores import Finder, ICPFinder, ICPFinderWithModel
from detectores import StaticDetector, StaticDetectorWithPCDFiltering, \
    StaticDetectorWithModelAlignment, AutomaticDetection
from esquemas_seguimiento import FollowingScheme
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
        'videos/rgbd/scenes/', 'desk', '1', 'videos/rgbd/objs/', 'coffee_mug',
        '5',
    )  # path, objname, number


    detector = AutomaticDetection(
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


def probando_filtro_por_ejes():

    img_provider = FrameNamesAndImageProvider(
        'videos/rgbd/scenes/', 'desk', '1',
        'videos/rgbd/objs/', 'coffee_mug', '5',
    )

    pcd = img_provider.pcd()
    filtered_pcd = img_provider.pcd()

    min_max = get_min_max(pcd)

    left = min_max.max_x - 0.46
    right = min_max.max_x - 0.35
    diff = right - left

    lefter_left = left - (diff / 2)
    righter_right = right + (diff / 2)

    filter_cloud(filtered_pcd, b"x", lefter_left, righter_right)

    show_clouds(b"completo vs filtrado amplio en x", pcd, filtered_pcd)

    filter_cloud(filtered_pcd, b"x", left, right)

    show_clouds(b"completo vs filtrado angosto en x", pcd, filtered_pcd)



if __name__ == '__main__':
    deteccion_automatica_icp_con_modelo()
