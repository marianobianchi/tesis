#coding=utf-8

from __future__ import (unicode_literals, division)

import time
import math

import numpy as np
import cv2

from cpp.common import get_point, points, get_min_max, compute_centroid


def dibujar_cuadrado(img, topleft, bottomright, color=(0, 0, 0)):
    cv2.rectangle(img, topleft, bottomright, color, 3)
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
        return (-10000.0, -10000.0)

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

    return (cloud_row, cloud_col)


def from_cloud_to_flat(cloud_row, cloud_col, float_depth):
    # images size is 640 COLS x 480 ROWS
    rows_center = 240
    cols_center = 320

    # focal distance
    constant = 570.3

    # inverse of cloud calculation
    im_row = int(cloud_row / float_depth * constant)
    im_col = int(cloud_col / float_depth * constant)

    im_row += rows_center
    im_col += cols_center

    return im_row, im_col


def from_cloud_to_flat_limits(cloud):
    top = 1e10
    left = 1e10
    bottom = 0
    right = 0
    for i in range(points(cloud)):
        point_xyz = get_point(cloud, i)
        point_flat = from_cloud_to_flat(point_xyz.y, point_xyz.x, point_xyz.z)

        top = min(point_flat[0], top)
        bottom = max(point_flat[0], bottom)

        left = min(point_flat[1], left)
        right = max(point_flat[1], right)

    #### NUEVO Y MAS CORTO METODO
    # minmax = get_min_max(cloud)
    # topleft_1 = from_cloud_to_flat(minmax.min_y, minmax.min_x, minmax.min_z)
    # topleft_2 = from_cloud_to_flat(minmax.min_y, minmax.min_x, minmax.max_z)
    # topright_1 = from_cloud_to_flat(minmax.min_y, minmax.max_x, minmax.min_z)
    # topright_2 = from_cloud_to_flat(minmax.min_y, minmax.max_x, minmax.max_z)
    #
    # bottomleft_1 = from_cloud_to_flat(minmax.max_y, minmax.min_x, minmax.min_z)
    # bottomleft_2 = from_cloud_to_flat(minmax.max_y, minmax.min_x, minmax.max_z)
    # bottomright_1 = from_cloud_to_flat(minmax.max_y, minmax.max_x, minmax.min_z)
    # bottomright_2 = from_cloud_to_flat(minmax.max_y, minmax.max_x, minmax.max_z)
    #
    # top = min(topleft_1[0], topleft_2[0], topright_1[0], topright_2[0])
    # bottom = max(bottomleft_1[0], bottomleft_2[0],
    #              bottomright_1[0], bottomright_2[0])
    #
    # left = min(topleft_1[1], topleft_2[1], bottomleft_1[1], bottomleft_2[1])
    # right = max(topright_1[1], topright_2[1],
    #             bottomright_1[1], bottomright_2[1])
    ####

    return (top, left), (bottom, right)


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

            if cloudRC[0] != -10000 and cloudRC[0] < r_top_limit:
                r_top_limit = cloudRC[0]

            if cloudRC[0] != -10000 and cloudRC[0] > r_bottom_limit:
                r_bottom_limit = cloudRC[0]

            if cloudRC[1] != -10000 and cloudRC[1] < c_left_limit:
                c_left_limit = cloudRC[1]

            if cloudRC[1] != -10000 and cloudRC[1] > c_right_limit:
                c_right_limit = cloudRC[1]

    top_bottom = (r_top_limit, r_bottom_limit)
    left_right = (c_left_limit, c_right_limit)
    res = (top_bottom, left_right)
    return res


#########
# TESTS #
#########
def test_flat_and_cloud_conversion():
    depth = 1300

    for i in range(0, 480):
        for j in range(0, 640):
            cloudXY = from_flat_to_cloud(i, j, depth)
            flatXY = from_cloud_to_flat(cloudXY[0], cloudXY[1], depth / 1000)

            if not ((flatXY[0] -1) <= i <= (flatXY[0] +1)):
                print flatXY[0] -1
                print i
                print flatXY[0] +1
                raise Exception('Falla la fila')
            if not ((flatXY[1] -1) <= j <= (flatXY[1] +1)):
                print flatXY[1] -1
                print j
                print flatXY[1] +1
                raise Exception('Falla la columna')



def measure_time(func):
    """
    Decorador que imprime en pantalla el tiempo que tarda en ejecutarse
    una funcion
    """
    def inner(*args, **kwargs):
        start_time = time.time()
        val = func(*args, **kwargs)
        end_time = time.time()
        print func.__name__, "took", end_time - start_time, "to finish"
        return val
    return inner


class Timer(object):
    def __init__(self, func_name):
        self.func_name = func_name

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        print self.func_name, "took", self.end - self.start, "to finish"


class AdaptSearchArea(object):
    """
    It adapts the search area depending on the speed of the object
    """
    def __init__(self):
        self.centroids = []

    def _distance(self, a_point, another_point):
        x = (a_point.x - another_point.x) ** 2
        y = (a_point.y - another_point.y) ** 2
        z = (a_point.z - another_point.z) ** 2
        return math.sqrt(x + y + z)

    def save_centroid(self, x_min, y_min, z_min, x_max, y_max, z_max):
        centroid = compute_centroid(x_min, y_min, z_min, x_max, y_max, z_max)
        self.centroids.append(centroid)
        print "Centro de masa=({x},{y},{z})".format(
            x=centroid.x,
            y=centroid.y,
            z=centroid.z,
        )
        if len(self.centroids) > 1:
            print "distancia recorrida = {d}".format(
                d=self._distance(self.centroids[-2], self.centroids[-1])
            )

        return centroid

    def search_area(self):
        """
        :return: a number greater than 1 that can be used to set the area range
        to look for the object. For example, if this method returns 2, it means
        that the area to look for the object is twice the size of the box
        containing the object before, with the center of the box in the same
        place as before
        """
        return 2