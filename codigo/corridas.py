#coding=utf-8

from __future__ import (unicode_literals, division)

from cpp.my_pcl import filter_cloud, show_clouds, get_min_max, save_pcd, points

from buscadores import Finder, ICPFinder, ICPFinderWithModel
from detectores import StaticDetector, StaticDetectorWithPCDFiltering, \
    StaticDetectorWithModelAlignment, AutomaticDetection
from esquemas_seguimiento import FollowingScheme
from metodos_de_busqueda import BusquedaPorFramesSolapados
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

    filtered_pcd = filter_cloud(filtered_pcd, b"x", lefter_left, righter_right)

    show_clouds(b"completo vs filtrado amplio en x", pcd, filtered_pcd)

    filtered_pcd = filter_cloud(filtered_pcd, b"x", left, right)

    show_clouds(b"completo vs filtrado angosto en x", pcd, filtered_pcd)


def segmentando_escena():
    img_provider = FrameNamesAndImageProvider(
        'videos/rgbd/scenes/', 'desk', '1',
        'videos/rgbd/objs/', 'coffee_mug', '5',
    )
    obj_pcd = img_provider.obj_pcd()
    min_max = get_min_max(obj_pcd)

    obj_width = (min_max.max_x - min_max.min_x)
    obj_height = (min_max.max_y - min_max.min_y)


    pcd = img_provider.pcd()
    min_max = get_min_max(pcd)

    scene_min_col = min_max.min_x
    scene_max_col = min_max.max_x
    scene_min_row = min_max.min_y
    scene_max_row = min_max.max_y

    path = b'pruebas_guardadas/pcd_segmentado/'

    save_pcd(pcd, path + b'scene.pcd')

    counter = 0
    for limits in (BusquedaPorFramesSolapados()
                   .iterate_frame_boxes(scene_min_col,
                                        scene_max_col,
                                        scene_min_row,
                                        scene_max_row,
                                        obj_width,
                                        obj_height)):
        cloud = filter_cloud(pcd, b'x', limits['min_x'], limits['max_x'])
        cloud = filter_cloud(cloud, b'y', limits['min_y'], limits['max_y'])

        if points(cloud) > 0:
            save_pcd(cloud, path + b'filtered_scene_{i}_box.pcd'.format(i=counter))

        counter += 1


if __name__ == '__main__':
    segmentando_escena()
