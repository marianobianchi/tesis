#coding=utf-8

from __future__ import (unicode_literals, division)

import numpy as np
import cv2

from cpp.icp_follow import (IntPair, FloatPair, DoubleFloatPair)


def dibujar_cuadrado(img, (fila_borde_sup_izq, col_borde_sup_izq), tam_region,
                                                        color=(0, 0, 0)):
    cv2.rectangle(
        img,
        (col_borde_sup_izq, fila_borde_sup_izq),
        (col_borde_sup_izq + tam_region, fila_borde_sup_izq + tam_region),
        color,
        3
    )
    return img


def pegar_fotos(foto1, foto2):
    """
    Pega dos fotos, una a la izquierda y la otra a la derecha y devuelve la
    foto resultante.

    Las fotos deben tener el mismo alto.
    """
    alto = len(foto1)
    ancho = len(foto1[0])

    foto3 = np.zeros((alto, ancho * 2, 3), dtype=foto1.dtype)
    foto3[:, :ancho] = foto1
    foto3[:, ancho:] = foto2
    return foto3


def dibujar_matching(train_frame, query_frame, kp_train, kp_query, good):

    good_train_kps_idx = [m[0].trainIdx for m in good]
    good_query_kps_idx = [m[0].queryIdx for m in good]

    good_train_kp = [kp_train[i] for i in good_train_kps_idx]
    good_query_kp = [kp_query[i] for i in good_query_kps_idx]

    train_with_kp = cv2.drawKeypoints(train_frame, good_train_kp)
    query_with_kp = cv2.drawKeypoints(query_frame, good_query_kp)

    frames_pegados = pegar_fotos(train_with_kp, query_with_kp)

    ancho = len(train_frame[0])

    for train_kp, query_kp in zip(good_train_kp, good_query_kp):
        pt1 = (int(train_kp.pt[0]+0.5), int(train_kp.pt[1]+0.5))
        pt2 = (int(train_kp.pt[0]+0.5) + ancho, int(query_kp.pt[1]+0.5))
        cv2.line(frames_pegados, pt1, pt2, (0,0,0))

    cv2.imshow('Frames pegados', frames_pegados)
    cv2.resizeWindow('Frames pegados', 200, 100)
    cv2.waitKey()


def keypoint_matching_entre_dos_imagenes():
    """
    ¿Que hago con los descriptores?
    """
    train_frame = cv2.imread('videos/pelotita_naranja_webcam/pelota_frame1.jpg')
    query_frame = cv2.imread('videos/pelotita_naranja_webcam/pelota_frame2.jpg')

    surf = cv2.SURF()

    # Deteccion de keypoints y generacion de descriptores
    kp_train, des_train = surf.detectAndCompute(train_frame, None)
    kp_query, des_query = surf.detectAndCompute(query_frame, None)


    # Calcular matching entre descriptores
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des_train,des_query, k=2)

    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])


    dibujar_matching(train_frame, query_frame, kp_train, kp_query, good)


def from_flat_to_cloud(imR, imC, depth):

    # return something invalid if depth is zero
    if depth == 0:
        return FloatPair(-10000, -10000)

    # cloud coordinates
    cloud_row = float(imR)
    cloud_col = float(imC)

    # images size is 640 COLS x 480 ROWS
    rows_center = 240
    cols_center = 320

    # focal distance
    constant = 570.3

    # move the coordinate (0,0) from the top-left corner to the center of the plane
    cloud_row = cloud_row - rows_center
    cloud_col = cloud_col - cols_center

    # calculate cloud
    cloud_row = cloud_row * depth / constant / 1000
    cloud_col = cloud_col * depth / constant / 1000

    return FloatPair(cloud_row, cloud_col)


def from_cloud_to_flat(cloud_row, cloud_col, depth):
    # images size is 640 COLS x 480 ROWS
    rows_center = 240
    cols_center = 320

    # focal distance
    constant = 570.3

    # inverse of cloud calculation
    imR = int(cloud_row / depth * constant * 1000)
    imC = int(cloud_col / depth * constant * 1000)


    imR = imR + rows_center
    imC = imC + cols_center

    return IntPair(imR, imC)


def from_flat_to_cloud_limits(topleft, bottomright, depth_img):
    """
     * Dada una imagen en profundidad y la ubicación de un objeto
     * en coordenadas sobre la imagen en profundidad, obtengo la nube
     * de puntos correspondiente a esas coordenadas y me quedo
     * unicamente con los valores máximos y mínimos de las coordenadas
     * "x" e "y" de dicha nube
    """

    # Inicializo los limites. Los inicializo al revés para que funcione bien al comparar
    r_top_limit    =  10.0
    r_bottom_limit = -10.0
    c_left_limit   =  10.0
    c_right_limit  = -10.0

    # TODO: paralelizar estos "for" usando funciones de OpenCV o PCL o lo que sea
    # Hint: que "from_flat_to_cloud" reciba una matriz (la imagen) directamente

    for r in range(topleft[0], bottomright[0] + 1):
        for c in range(topleft[1], bottomright[1] + 1):
            depth = depth_img[r][c]

            cloudRC = from_flat_to_cloud(r, c, depth)

            if cloudRC.first != -10000 and cloudRC.first < r_top_limit:
                r_top_limit = cloudRC.first

            if cloudRC.first != -10000 and cloudRC.first > r_bottom_limit:
                r_bottom_limit = cloudRC.first

            if cloudRC.second != -10000 and cloudRC.second < c_left_limit:
                c_left_limit = cloudRC.second

            if cloudRC.second != -10000 and cloudRC.second > c_right_limit:
                c_right_limit = cloudRC.second

    top_bottom = FloatPair(r_top_limit, r_bottom_limit)
    left_right = FloatPair(c_left_limit, c_right_limit)
    res = DoubleFloatPair(top_bottom, left_right)
    return res


def filter_cloud(cloud, field_name, lower_limit, upper_limit):
    # Create the filtering object
    pass_through_filter = cloud.make_passthrough_filter()
    pass_through_filter.set_filter_field_name(field_name)
    pass_through_filter.set_filter_limits(lower_limit, upper_limit)

    # Filter
    return pass_through_filter.filter()

#########
# TESTS #
#########
def test_flat_and_cloud_conversion():
    depth = 1300

    for i in range(0, 480):
        for j in range(0, 640):
            cloudXY = from_flat_to_cloud(i, j, depth)
            flatXY = from_cloud_to_flat(cloudXY.first, cloudXY.second, depth)

            if not ((flatXY.first -1) <= i <= (flatXY.first +1)):
                print flatXY.first -1
                print i
                print flatXY.first +1
                raise Exception('Falla la fila')
            if not ((flatXY.second -1) <= j <= (flatXY.second +1)):
                print flatXY.second -1
                print j
                print flatXY.second +1
                raise Exception('Falla la columna')
