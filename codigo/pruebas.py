#!/usr/bin/python
#coding=utf-8

from __future__ import (unicode_literals, division, print_function)

import cv2
import matplotlib.pyplot as plt

from cpp.common import get_min_max, filter_cloud, show_clouds, save_pcd, points

from metodos_de_busqueda import BusquedaPorFramesSolapados
from proveedores_de_imagenes import FrameNamesAndImageProvider
from metodos_comunes import AdaptSearchArea


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


def prueba_histogramas():
    # Sacado de: http://www.pyimagesearch.com/2014/07/14/3-ways-compare-histograms-using-opencv-python/
    # METHOD #1: UTILIZING OPENCV
    # initialize OpenCV methods for histogram comparison
    OPENCV_METHODS = (
        ("Correlation", cv2.cv.CV_COMP_CORREL),
        ("Chi-Squared", cv2.cv.CV_COMP_CHISQR),
        ("Intersection", cv2.cv.CV_COMP_INTERSECT),
        ("Hellinger", cv2.cv.CV_COMP_BHATTACHARYYA),
    )

    # initialize the index dictionary to store the image name
    # and corresponding histograms and the images dictionary
    # to store the images themselves
    rgb_index = {}
    hsv_index = {}
    images = {}

    model_filename = 'taza_modelo'
    model_mask = cv2.imread('taza_modelo_mascara.png', cv2.COLOR_BGR2GRAY)

    # loop over the image paths
    for filename in ['taza_modelo', 'taza_maso_encontrada1',
                     'taza_maso_encontrada2', 'taza_maso_encontrada3',
                     'taza_maso_encontrada7', 'taza', 'taza2', 'taza3',
                     'taza4']:
        mask = None
        image = cv2.imread(filename + '.png')
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        images[filename] = image_rgb

        if 'modelo' in filename:
            mask = model_mask

        # extract a 3D RGB color histogram from the image,
        # using 8 bins per channel, normalize, and update
        # the index
        hist = cv2.calcHist([image_rgb], [0], mask, [70], [0, 256])
        hist = cv2.normalize(hist).flatten()
        rgb_index[filename] = hist

        # extract a 3D HSV color histogram from the image
        hist = cv2.calcHist([image_hsv], [0, 2], mask, [16, 32], [0, 180, 0, 256])
        hist = cv2.normalize(hist).flatten()
        hsv_index[filename] = hist

    # loop over the comparison methods
    for (methodName, method) in OPENCV_METHODS:
        # initialize the results dictionary and the sort
        # direction
        results = {}
        reverse = False

        # if we are using the correlation or intersection
        # method, then sort the results in reverse order
        if methodName in ("Correlation", "Intersection"):
            reverse = True

        for (k, hist) in rgb_index.items():
            # compute the distance between the two histograms
            # using the method and update the results dictionary
            d = cv2.compareHist(rgb_index[model_filename], hist, method)
            results['{f}#RGB'.format(f=k)] = d

        for (k, hist) in hsv_index.items():
            # compute the distance between the two histograms
            # using the method and update the results dictionary
            d = cv2.compareHist(hsv_index[model_filename], hist, method)
            results['{f}#HSV'.format(f=k)] = d

        # sort the results
        results = sorted([(v, k) for (k, v) in results.items()], reverse=reverse)

        # show the query image
        fig = plt.figure("Query")
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(images[model_filename])
        plt.axis("off")

        # initialize the results figure
        figrgb = plt.figure("Results RGB: %s" % methodName)
        figrgb.suptitle(methodName + ' - RGB', fontsize=20)

        fighsv = plt.figure("Results HSV: %s" % methodName)
        fighsv.suptitle(methodName + ' - HSV', fontsize=20)

        # loop over the results
        rgb_res_num = 0
        hsv_res_num = 0
        for (v, k) in results:
            # show the result
            if 'RGB' in k:
                ax = figrgb.add_subplot(3, 3, rgb_res_num + 1)
                rgb_res_num += 1
            else:
                ax = fighsv.add_subplot(3, 3, hsv_res_num + 1)
                hsv_res_num += 1

            ax.set_title("%s: %.2f" % (k, v))

            k = k.split('#')[0]
            ax.imshow(images[k])
            ax.axis("off")

    # show the OpenCV methods
    plt.show()


def prueba_mejor_canal_hsv_histogramas():
    OPENCV_METHODS = (
        ("Correlation", cv2.cv.CV_COMP_CORREL),
        ("Chi-Squared", cv2.cv.CV_COMP_CHISQR),
        ("Intersection", cv2.cv.CV_COMP_INTERSECT),
        ("Hellinger", cv2.cv.CV_COMP_BHATTACHARYYA),
    )

    # initialize the index dictionary to store the image name
    # and corresponding histograms and the images dictionary
    # to store the images themselves
    h_index = {}
    s_index = {}
    v_index = {}
    images = {}

    model_filename = 'taza_modelo'
    model_mask = cv2.imread('taza_modelo_mascara.png', cv2.COLOR_BGR2GRAY)

    # loop over the image paths
    for filename in ['taza_modelo', 'taza_maso_encontrada2',
                     'taza_maso_encontrada7', 'taza', 'taza3']:
        mask = None
        image = cv2.imread(filename + '.png')
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        images[filename] = image_rgb

        if 'modelo' in filename:
            mask = model_mask

        h_hist = cv2.calcHist([image_hsv], [0], mask, [4], [0, 180])
        h_hist = cv2.normalize(h_hist).flatten()
        h_index[filename] = h_hist

        s_hist = cv2.calcHist([image_hsv], [1], mask, [4], [0, 256])
        s_hist = cv2.normalize(s_hist).flatten()
        s_index[filename] = s_hist

        v_hist = cv2.calcHist([image_hsv], [2], mask, [4], [0, 256])
        v_hist = cv2.normalize(v_hist).flatten()
        v_index[filename] = v_hist

    # loop over the comparison methods
    for (methodName, method) in OPENCV_METHODS:
        # initialize the results dictionary and the sort
        # direction
        results = {}
        reverse = False

        # if we are using the correlation or intersection
        # method, then sort the results in reverse order
        if methodName in ("Correlation", "Intersection"):
            reverse = True

        for (k, hist) in h_index.items():
            # compute the distance between the two histograms
            # using the method and update the results dictionary
            d = cv2.compareHist(h_index[model_filename], hist, method)
            results['{f}#H'.format(f=k)] = d

        for (k, hist) in s_index.items():
            # compute the distance between the two histograms
            # using the method and update the results dictionary
            d = cv2.compareHist(s_index[model_filename], hist, method)
            results['{f}#S'.format(f=k)] = d

        for (k, hist) in v_index.items():
            # compute the distance between the two histograms
            # using the method and update the results dictionary
            d = cv2.compareHist(v_index[model_filename], hist, method)
            results['{f}#V'.format(f=k)] = d

        # sort the results
        results = sorted([(v, k) for (k, v) in results.items()], reverse=reverse)

        # initialize the results figure
        fig = plt.figure("Results for {m}".format(m=methodName))

        # loop over the results
        h_res_num = 0
        s_res_num = 4
        v_res_num = 8
        for (v, k) in results:
            if 'modelo' not in k:
                # show the result
                if 'H' in k:
                    ax = fig.add_subplot(3, 4, h_res_num + 1)
                    h_res_num += 1
                elif 'S' in k:
                    ax = fig.add_subplot(3, 4, s_res_num + 1)
                    s_res_num += 1
                elif 'V' in k:
                    ax = fig.add_subplot(3, 4, v_res_num + 1)
                    v_res_num += 1
                else:
                    raise Exception("ALGO FALLO")

                ax.set_title("%s: %.2f" % (k, v))

                k = k.split('#')[0]
                ax.imshow(images[k])
                ax.axis("off")

    # show the OpenCV methods
    plt.show()


def prueba_promedio_histogramas():
    mod1 = cv2.imread('taza_modelo.png', cv2.COLOR_BGR2RGB)
    masc1 = cv2.imread('taza_mascara.png', cv2.COLOR_BGR2GRAY)
    hist1 = cv2.calcHist(
        [mod1], [0, 1, 2], masc1, [16, 8, 8], [0, 256, 0, 256, 0, 256]
    )
    hist1 = cv2.normalize(hist1).flatten()

    mod2 = cv2.imread('taza_modelo_2.png', cv2.COLOR_BGR2RGB)
    masc2 = cv2.imread('taza_mascara_2.png', cv2.COLOR_BGR2GRAY)
    hist2 = cv2.calcHist(
        [mod2], [0, 1, 2], masc2, [16, 8, 8], [0, 256, 0, 256, 0, 256]
    )
    hist2 = cv2.normalize(hist2).flatten()

    mod3 = cv2.imread('taza_modelo_3.png', cv2.COLOR_BGR2RGB)
    masc3 = cv2.imread('taza_mascara_3.png', cv2.COLOR_BGR2GRAY)
    hist3 = cv2.calcHist(
        [mod3], [0, 1, 2], masc3, [16, 8, 8], [0, 256, 0, 256, 0, 256]
    )
    hist3 = cv2.normalize(hist3).flatten()

    hist_prom = (hist1 + hist2 + hist3) / 3

    fig = plt.figure("Histogramas")
    ax = fig.add_subplot(2, 2, 1)
    ax.plot(hist1)
    ax.set_title("modelo 1")

    ax = fig.add_subplot(2, 2, 2)
    ax.plot(hist2)
    ax.set_title("modelo 2")

    ax = fig.add_subplot(2, 2, 3)
    ax.plot(hist3)
    ax.set_title("modelo 3")

    ax = fig.add_subplot(2, 2, 4)
    ax.plot(hist_prom)
    ax.set_title("promedio")

    # show the OpenCV methods
    plt.show()


if __name__ == '__main__':
    prueba_histogramas()
