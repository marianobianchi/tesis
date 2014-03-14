#!/usr/bin/python
# -*- coding: utf-8 -*-
#Con esto, todos los strings literales son unicode (no hace falta poner u'algo')
from __future__ import unicode_literals

import numpy as np
import cv2


def pegar_fotos(foto1, foto2):
    """
    Pega dos fotos, una a la izquierda y la otra a la derecha y devuelve la
    foto resultante.

    Las fotos deben tener el mismo alto.
    """
    alto = len(foto1)
    ancho = len(foto1[0])

    foto3 = np.zeros((alto, ancho*2, 3), dtype=foto1.dtype)
    foto3[:,:ancho] = foto1
    foto3[:,ancho:] = foto2
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
    train_frame = cv2.imread('../videos/pelotita_naranja_webcam/pelota_frame1.jpg')
    query_frame = cv2.imread('../videos/pelotita_naranja_webcam/pelota_frame2.jpg')

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