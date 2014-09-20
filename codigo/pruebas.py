#!/usr/bin/python
#coding=utf-8

from __future__ import (unicode_literals, division, print_function)

from cpp.my_pcl import get_min_max, filter_cloud, show_clouds, save_pcd, points

from metodos_de_busqueda import BusquedaPorFramesSolapados
from proveedores_de_imagenes import FrameNamesAndImageProvider



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

    obj_width = (min_max.max_x - min_max.min_x) * 2
    obj_height = (min_max.max_y - min_max.min_y) * 2


    pcd = img_provider.pcd()
    min_max = get_min_max(pcd)

    scene_min_col = min_max.min_x
    scene_max_col = min_max.max_x
    scene_min_row = min_max.min_y
    scene_max_row = min_max.max_y

    path = b'pruebas_guardadas/pcd_segmentado/'

    save_pcd(pcd, path + b'scene.pcd')

    # #######################################################
    # Algunos calculos a mano para corroborar que ande bien
    # #######################################################
    scene_width = scene_max_col - scene_min_col
    frames_ancho = (scene_width / obj_width) * 2 - 1

    print("Frames a lo ancho:", round(frames_ancho))

    scene_height = scene_max_row - scene_min_row
    frames_alto = (scene_height / obj_height) * 2 - 1
    print("Frames a lo alto:", round(frames_alto))
    print("Frames totales supuestos:", frames_ancho * frames_alto)
    ########################################################

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


def testing_voxel_grid():
    pass


if __name__ == '__main__':
    segmentando_escena()