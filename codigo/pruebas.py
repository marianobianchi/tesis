#!/usr/bin/python
#coding=utf-8

from __future__ import (unicode_literals, division, print_function)

import os
import math
import cv2, numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance as dist

from cpp.common import get_min_max, filter_cloud, show_clouds, save_pcd, points

from detectores import StaticDetectorForRGBFinder, StaticDetector
from metodos_de_busqueda import BusquedaPorFramesSolapados, \
    BusquedaAlrededorCambiandoFrameSize
from proveedores_de_imagenes import FrameNamesAndImageProvider
from observar_seguimiento import MuestraSeguimientoEnVivo
from buscadores import FragmentedHistogramFinder
from esquemas_seguimiento import FollowingScheme
from seguidores import RGBFollower


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

    # Taza
    model_filename = 'imagenes/taza3'
    model_mask = cv2.imread('imagenes/taza_mascara.png', cv2.IMREAD_GRAYSCALE)

    # Gorra
    # model_filename = 'imagenes/gorra_encontrada1'
    # model_mask = cv2.imread('imagenes/gorra_modelo_mascara.png', cv2.IMREAD_GRAYSCALE)

    # loop over the image paths
    # Taza
    for filename in ['imagenes/taza_modelo', 'imagenes/taza2',
                     'imagenes/taza3', 'imagenes/taza4',
                     'imagenes/taza_maso_encontrada1', 'imagenes/taza_maso_encontrada2',
                     'imagenes/taza_maso_encontrada5', 'imagenes/taza',
                     'imagenes/taza_maso_encontrada7']:
    # Gorra
    # for filename in ['imagenes/gorra_modelo', 'imagenes/gorra_encontrada',
    #                  'imagenes/gorra_encontrada1', 'imagenes/gorra_encontrada2',
    #                  'imagenes/gorra_encontrada3', 'imagenes/gorra_seguida1',
    #                  'imagenes/gorra_seguida2', 'imagenes/gorra_seguida3',
    #                  'imagenes/gorra_seguida4']:
        mask = None
        image = cv2.imread(filename + '.png')
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        images[filename] = image_rgb

        if 'modelo' in filename:
            mask = model_mask

        # extract a 3D RGB color histogram from the image,
        # using 8 bins per channel, normalize, and update
        # the index
        hist = cv2.calcHist([image_rgb], [0, 1, 2], mask, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist).flatten()
        rgb_index[filename] = hist

        # extract a 3D HSV color histogram from the image
        hist = cv2.calcHist([image_hsv], [1, 2], mask, [8, 16], [0, 256, 0, 256])
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
        # if methodName in ("Correlation", "Intersection"):
        #     reverse = True

        for (k, hist) in rgb_index.items():
            # compute the distance between the two histograms
            # using the method and update the results dictionary
            d = cv2.compareHist(rgb_index[model_filename], hist, method)
            if methodName == 'Correlation':
                d = 1 - abs(d)
            elif methodName == 'Intersection':
                max_comp = max(sum(rgb_index[model_filename]), sum(hist))
                d = 1 - d / max_comp
            elif methodName == 'Chi-Squared':
                d = 1 - 1.0 / (1 + d)
            results['{f}#RGB'.format(f=k)] = d

        for (k, hist) in hsv_index.items():
            # compute the distance between the two histograms
            # using the method and update the results dictionary
            d = cv2.compareHist(hsv_index[model_filename], hist, method)
            if methodName == 'Correlation':
                d = 1 - abs(d)
            elif methodName == 'Intersection':
                max_comp = max(sum(hsv_index[model_filename]), sum(hist))
                d = 1 - d / max_comp
            elif methodName == 'Chi-Squared':
                d = 1 - 1.0 / (1 + d)
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

            # ax.set_title("%s: %.5f" % (k, v))
            ax.set_title("%.5f" % v)

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

    model_filename = 'imagenes/taza_modelo'
    model_mask = cv2.imread(
        'imagenes/taza_modelo_mascara.png',
        cv2.IMREAD_GRAYSCALE
    )

    # loop over the image paths
    for filename in ['imagenes/taza_modelo', 'imagenes/taza_maso_encontrada2',
                     'imagenes/taza_maso_encontrada7', 'imagenes/taza',
                     'imagenes/taza3']:
        mask = None
        image = cv2.imread(filename + '.png')
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        images[filename] = image_rgb

        if 'modelo' in filename:
            mask = model_mask

        h_hist = cv2.calcHist([image_hsv], [0], mask, [30], [0, 180])
        h_hist = cv2.normalize(h_hist).flatten()
        h_index[filename] = h_hist

        s_hist = cv2.calcHist([image_hsv], [1], mask, [30], [0, 256])
        s_hist = cv2.normalize(s_hist).flatten()
        s_index[filename] = s_hist

        v_hist = cv2.calcHist([image_hsv], [2], mask, [30], [0, 256])
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


def prueba_mejor_canal_rgb_histogramas():
    OPENCV_METHODS = (
        ("Correlation", cv2.cv.CV_COMP_CORREL),
        ("Chi-Squared", cv2.cv.CV_COMP_CHISQR),
        ("Intersection", cv2.cv.CV_COMP_INTERSECT),
        ("Hellinger", cv2.cv.CV_COMP_BHATTACHARYYA),
    )

    # initialize the index dictionary to store the image name
    # and corresponding histograms and the images dictionary
    # to store the images themselves
    r_index = {}
    g_index = {}
    b_index = {}
    images = {}

    model_filename = 'imagenes/taza_modelo'
    model_mask = cv2.imread(
        'imagenes/taza_modelo_mascara.png',
        cv2.IMREAD_GRAYSCALE
    )

    # loop over the image paths
    for filename in ['imagenes/taza_modelo', 'imagenes/taza_maso_encontrada4',
                     'imagenes/taza', 'imagenes/taza4', 'imagenes/taza2']:
        mask = None
        image = cv2.imread(filename + '.png')
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images[filename] = image_rgb

        if model_filename == filename:
            mask = model_mask

        b_hist = cv2.calcHist([image], [0], mask, [30], [0, 256])
        b_hist = cv2.normalize(b_hist).flatten()
        b_index[filename] = b_hist

        g_hist = cv2.calcHist([image], [1], mask, [30], [0, 256])
        g_hist = cv2.normalize(g_hist).flatten()
        g_index[filename] = g_hist

        r_hist = cv2.calcHist([image], [2], mask, [30], [0, 256])
        r_hist = cv2.normalize(r_hist).flatten()
        r_index[filename] = r_hist

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

        for (k, hist) in b_index.items():
            # compute the distance between the two histograms
            # using the method and update the results dictionary
            d = cv2.compareHist(b_index[model_filename], hist, method)
            results['{f}#B'.format(f=k)] = d

        for (k, hist) in g_index.items():
            # compute the distance between the two histograms
            # using the method and update the results dictionary
            d = cv2.compareHist(g_index[model_filename], hist, method)
            results['{f}#G'.format(f=k)] = d

        for (k, hist) in r_index.items():
            # compute the distance between the two histograms
            # using the method and update the results dictionary
            d = cv2.compareHist(r_index[model_filename], hist, method)
            results['{f}#R'.format(f=k)] = d

        # sort the results
        results = sorted([(v, k) for (k, v) in results.items()], reverse=reverse)

        # initialize the results figure
        fig = plt.figure("Results for {m}".format(m=methodName))

        # loop over the results
        h_res_num = 0
        s_res_num = 4
        v_res_num = 8
        for (v, k) in results:
            if (model_filename + '#') not in k:
                # show the result
                if 'B' in k:
                    ax = fig.add_subplot(3, 4, h_res_num + 1)
                    h_res_num += 1
                elif 'G' in k:
                    ax = fig.add_subplot(3, 4, s_res_num + 1)
                    s_res_num += 1
                elif 'R' in k:
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


def prueba_mezclando_canales_hsv_histogramas():
    # METHOD #1: UTILIZING OPENCV
    # initialize OpenCV methods for histogram comparison
    SCIPY_METHODS = (
        ("Euclidean", dist.euclidean),
        ("Manhattan", dist.cityblock),
        ("Chebysev", dist.chebyshev)
    )

    # initialize the index dictionary to store the image name
    # and corresponding histograms and the images dictionary
    # to store the images themselves
    h_index = {}
    s_index = {}
    v_index = {}
    images = {}

    model_filename = 'imagenes/gorra_modelo'
    model_mask = cv2.imread(
        'imagenes/gorra_modelo_mascara.png',
        cv2.IMREAD_GRAYSCALE
    )

    # loop over the image paths
    for filename in ['imagenes/gorra_modelo', 'imagenes/gorra_seguida1',
                     'imagenes/gorra_seguida2', 'imagenes/gorra_seguida3',
                     'imagenes/gorra_seguida4', 'imagenes/gorra_encontrada1',
                     'imagenes/gorra_encontrada2', 'imagenes/gorra_encontrada3',
                     'imagenes/gorra_encontrada']:
        mask = None
        image = cv2.imread(filename + '.png')
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        images[filename] = image_rgb

        if 'modelo' in filename:
            mask = model_mask

        # histograma para H
        hist = cv2.calcHist([image_hsv], [0], mask, [40], [0, 180])
        hist = cv2.normalize(hist).flatten()
        h_index[filename] = hist

        # histograma para S
        hist = cv2.calcHist([image_hsv], [1], mask, [60], [0, 256])
        hist = cv2.normalize(hist).flatten()
        s_index[filename] = hist

        # histograma para V
        hist = cv2.calcHist([image_hsv], [2], mask, [60], [0, 256])
        hist = cv2.normalize(hist).flatten()
        v_index[filename] = hist

    # Histogramas del modelo
    model_h_hist = h_index[model_filename]
    model_s_hist = s_index[model_filename]
    model_v_hist = v_index[model_filename]

    # Metodo de comparacion por canal
    h_comp = cv2.cv.CV_COMP_BHATTACHARYYA
    s_comp = cv2.cv.CV_COMP_CHISQR
    v_comp = cv2.cv.CV_COMP_CHISQR

    # Centro para calcular las distancias
    center = np.array([
        cv2.compareHist(model_h_hist, model_h_hist, h_comp),
        cv2.compareHist(model_s_hist, model_s_hist, s_comp),
        cv2.compareHist(model_v_hist, model_v_hist, v_comp),
    ])

    # loop over the comparison methods
    for (methodName, method) in SCIPY_METHODS:
        # initialize the results dictionary and the sort
        # direction
        results = {}
        reverse = False

        for fname in h_index.keys():
            h_hist = h_index[fname]
            s_hist = s_index[fname]
            v_hist = v_index[fname]

            h_point = cv2.compareHist(model_h_hist, h_hist, h_comp)
            s_point = cv2.compareHist(model_s_hist, s_hist, s_comp)
            v_point = cv2.compareHist(model_v_hist, v_hist, v_comp)
            point = np.array([h_point, s_point, v_point])

            metric = method(center, point)

            results[fname] = metric

        # sort the results
        results = sorted([(v, k) for (k, v) in results.items()], reverse=reverse)

        # initialize the results figure
        fig = plt.figure("Results: %s" % methodName)
        fig.suptitle(methodName, fontsize=20)

        # loop over the results
        for i, (v, k) in enumerate(results):
            # show the result
            ax = fig.add_subplot(3, 3, i + 1)
            ax.set_title("%s: %.2f" % (k, v))
            ax.imshow(images[k])
            ax.axis("off")

    # show the OpenCV methods
    plt.show()


def prueba_promedio_histogramas():
    mod1 = cv2.imread('imagenes/taza_modelo.png', cv2.IMREAD_COLOR)
    masc1 = cv2.imread('imagenes/taza_mascara.png', cv2.IMREAD_GRAYSCALE)
    hist1 = cv2.calcHist(
        [mod1], [0, 1, 2], masc1, [16, 8, 8], [0, 256, 0, 256, 0, 256]
    )
    hist1 = cv2.normalize(hist1).flatten()

    mod2 = cv2.imread('imagenes/taza_modelo_2.png', cv2.IMREAD_COLOR)
    masc2 = cv2.imread('imagenes/taza_mascara_2.png', cv2.IMREAD_GRAYSCALE)
    hist2 = cv2.calcHist(
        [mod2], [0, 1, 2], masc2, [16, 8, 8], [0, 256, 0, 256, 0, 256]
    )
    hist2 = cv2.normalize(hist2).flatten()

    mod3 = cv2.imread('imagenes/taza_modelo_3.png', cv2.IMREAD_COLOR)
    masc3 = cv2.imread('imagenes/taza_mascara_3.png', cv2.IMREAD_GRAYSCALE)
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


def ver_canales_e_histogramas_hsv(img_path):
    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    shape_gray = (img_hsv.shape[0], img_hsv.shape[1])

    # Separo los canales HSV
    img_h = np.zeros(shape_gray, np.uint8)
    img_h[:, :] = img_hsv[:, :, 0]

    img_s = np.zeros(shape_gray, np.uint8)
    img_s[:, :] = img_hsv[:, :, 1]

    img_v = np.zeros(shape_gray, np.uint8)
    img_v[:, :] = img_hsv[:, :, 2]

    # Mostrando canales HSV
    fig = plt.figure()
    ax = fig.add_subplot(2, 3, 1)
    ax.set_title('Hue')
    ax.imshow(img_h, cmap='gray')

    ax = fig.add_subplot(2, 3, 2)
    ax.set_title('Saturation')
    ax.imshow(img_s, cmap='gray')

    ax = fig.add_subplot(2, 3, 3)
    ax.set_title('Value')
    ax.imshow(img_v, cmap='gray')

    bins = 20
    # Mostrando histogramas de control y HSV
    hist_g = cv2.calcHist([img_h], [0], None, [bins], [0, 180])
    hist_g = cv2.normalize(hist_g)
    ax = fig.add_subplot(2, 3, 4)
    ax.set_title('Histograma H')
    ax.bar(range(0, bins), hist_g)
    ax.set_yticks(np.arange(0, 1.1, 0.1))

    hist_g = cv2.calcHist([img_s], [0], None, [bins], [0, 256])
    hist_g = cv2.normalize(hist_g)
    ax = fig.add_subplot(2, 3, 5)
    ax.set_title('Histograma S')
    ax.bar(range(0, bins), hist_g)
    ax.set_yticks(np.arange(0, 1.1, 0.1))

    hist_b = cv2.calcHist([img_v], [0], None, [bins], [0, 256])
    hist_b = cv2.normalize(hist_b)
    ax = fig.add_subplot(2, 3, 6)
    ax.set_title('Histograma V')
    ax.bar(np.arange(0, bins), hist_b)
    ax.set_yticks(np.arange(0, 1.1, 0.1))

    plt.show()


def ver_canales_e_histogramas_rgb(img_path):
    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)

    shape_gray = (img_bgr.shape[0], img_bgr.shape[1])

    # Separo los canales RGB
    img_r = np.zeros(shape_gray, np.uint8)
    img_r[:, :] = img_bgr[:, :, 2]

    img_g = np.zeros(shape_gray, np.uint8)
    img_g[:, :] = img_bgr[:, :, 1]

    img_b = np.zeros(shape_gray, np.uint8)
    img_b[:, :] = img_bgr[:, :, 0]

    # Mostrando canales HSV
    fig = plt.figure()
    ax = fig.add_subplot(2, 3, 1)
    ax.set_title('Red')
    ax.imshow(img_r, cmap='gray')

    ax = fig.add_subplot(2, 3, 2)
    ax.set_title('Green')
    ax.imshow(img_g, cmap='gray')

    ax = fig.add_subplot(2, 3, 3)
    ax.set_title('Blue')
    ax.imshow(img_b, cmap='gray')

    # Mostrando histogramas RGB
    hist_r = cv2.calcHist([img_r], [0], None, [60], [0, 256])
    hist_r = cv2.normalize(hist_r)
    ax = fig.add_subplot(2, 3, 4)
    ax.set_title('Histograma R')
    ax.bar(range(0, 60), hist_r)
    ax.set_yticks(np.arange(0, 1.1, 0.1))

    hist_g = cv2.calcHist([img_g], [0], None, [60], [0, 256])
    hist_g = cv2.normalize(hist_g)
    ax = fig.add_subplot(2, 3, 5)
    ax.set_title('Histograma G')
    ax.bar(range(0, 60), hist_g)
    ax.set_yticks(np.arange(0, 1.1, 0.1))

    hist_b = cv2.calcHist([img_b], [0], None, [60], [0, 256])
    hist_b = cv2.normalize(hist_b)
    ax = fig.add_subplot(2, 3, 6)
    ax.set_title('Histograma B')
    ax.bar(np.arange(0, 60), hist_b)
    ax.set_yticks(np.arange(0, 1.1, 0.1))

    plt.show()


def probando_mi_metodo_chebysev():
    img_provider = FrameNamesAndImageProvider(
        'videos/rgbd/scenes/',  # scene path
        'desk',  # scene
        '1',  # scene number
        'videos/rgbd/objs/',  # object path
        'coffee_mug',  # object
        '5',  # object number
    )

    # Detector
    detector = StaticDetectorForRGBFinder(
        'videos/rgbd/scenes/desk/desk_1.mat',
        'coffee_mug',
        '5'
    )

    # Buscador
    metodo_de_busqueda = BusquedaAlrededorCambiandoFrameSize()
    channels_comparators = [
        (0, 40, 180, cv2.cv.CV_COMP_BHATTACHARYYA),
        (1, 60, 256, cv2.cv.CV_COMP_BHATTACHARYYA),
        (2, 60, 256, cv2.cv.CV_COMP_BHATTACHARYYA),
    ]
    center_point = [0, 0, 0]
    extern_point = [1, 1, 1]
    finder = FragmentedHistogramFinder(
        channels_comparators=channels_comparators,
        center_point=center_point,
        extern_point=extern_point,
        distance_comparator=dist.chebyshev,
        template_perc=0.5,
        frame_perc=0.2,
        metodo_de_busqueda=metodo_de_busqueda,
    )

    # Seguidor
    follower = RGBFollower(img_provider, detector, finder)

    show_following = MuestraSeguimientoEnVivo(
        'Deteccion estatica - Seguimiento por mi metodo'
    )

    FollowingScheme(
        img_provider,
        follower,
        show_following,
    ).run()


def ver_y_guardar_static_detection():
    scenename = 'table_small'
    scenenum = '2'
    objname = 'cereal_box'
    objnum = '4'
    frames = 234
    templates_path = 'videos/rgbd/scenes/{sname}/{sname}_{snum}/templates/'.format(
        sname=scenename,
        snum=scenenum,
    )
    if not os.path.isdir(templates_path):
        os.mkdir(templates_path)

    img_provider = FrameNamesAndImageProvider(
        'videos/rgbd/scenes/', scenename, unicode(scenenum),
        'videos/rgbd/objs/', objname, unicode(objnum),
    )

    detector = StaticDetector(
        'videos/rgbd/scenes/{sname}/{sname}_{snum}.mat'.format(
            sname=scenename,
            snum=scenenum,
        ),
        objname,
        objnum
    )
    while img_provider.have_images():
        img = img_provider.rgb_img()
        nframe = img_provider.nframe()
        print('Detectando en frame', nframe)

        detector.update({'nframe': nframe})
        fue_exitoso, desc = detector.detect()

        topleft = desc['topleft']
        bottomright = desc['bottomright']

        if fue_exitoso:
            MuestraSeguimientoEnVivo('sistema RGBD').run(
                img_provider,
                topleft,
                bottomright,
                fue_exitoso,
                True
            )
            print("Desea guardar el template? s = si, q = no")
            key = cv2.waitKey(1) & 0xFF
            while key != ord('q'):
                key = cv2.waitKey(1) & 0xFF
                if key == ord('s'):
                    frame = img[topleft[0]:bottomright[0],
                                topleft[1]:bottomright[1]]
                    fname = '{sname}_{snum}_{objname}_{objnum}__{nframe:03}.png'.format(
                        sname=scenename,
                        snum=scenenum,
                        objname=objname,
                        objnum=objnum,
                        nframe=nframe
                    )
                    cv2.imwrite(os.path.join(templates_path, fname), frame)

        img_provider.next()


if __name__ == '__main__':
    prueba_histogramas()
