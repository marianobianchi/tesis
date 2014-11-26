#coding=utf-8

from __future__ import (unicode_literals, division)


import cv2

from esquemas_seguimiento import FollowingScheme
from metodos_de_busqueda import BusquedaEnEspiralCambiandoFrameSize
from observar_seguimiento import MuestraSeguimientoEnVivo
from seguidores import TemplateMatchingAndSURFFollowing


def seguir_pelota_naranja_version5():
    img_provider = cv2.VideoCapture(
        'videos/pelotita_naranja_webcam/output.avi'
    )
    template = cv2.imread(
        'videos/pelotita_naranja_webcam/template_pelota.jpg'
    )
    follower = TemplateMatchingAndSURFFollowing(
        img_provider,
        template,
        metodo_de_busqueda=BusquedaEnEspiralCambiandoFrameSize()
    )
    muestra_seguimiento = MuestraSeguimientoEnVivo(nombre='Seguimiento')
    FollowingScheme(img_provider, follower, muestra_seguimiento).run()


def seguir_nariz_boca():
    img_provider = cv2.VideoCapture(
        'videos/pelotita_naranja_webcam/output.avi'
    )
    template = cv2.imread(
        'videos/pelotita_naranja_webcam/template_bocanariz.jpg'
    )
    follower = TemplateMatchingAndSURFFollowing(
        img_provider,
        template,
        metodo_de_busqueda=BusquedaEnEspiralCambiandoFrameSize()
    )
    muestra_seguimiento = MuestraSeguimientoEnVivo(nombre='Seguimiento')
    FollowingScheme(img_provider, follower, muestra_seguimiento).run()


if __name__ == '__main__':
    seguir_nariz_boca()