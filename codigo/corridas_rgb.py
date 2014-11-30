#coding=utf-8

from __future__ import (unicode_literals, division)


import cv2

from buscadores import BhattacharyyaHistogramFinder, CorrelationHistogramFinder, \
    IntersectionHistogramFinder
from esquemas_seguimiento import FollowingScheme
from detectores import RGBTemplateDetector, StaticDetector
from metodos_de_busqueda import BusquedaEnEspiralCambiandoFrameSize
from observar_seguimiento import MuestraSeguimientoEnVivo
from proveedores_de_imagenes import FrameNamesAndImageProvider, \
    TemplateAndImageProviderFromVideo
from seguidores import FollowerStaticAndRGBTemplate


def seguir_pelota_naranja_version5():
    img_provider = cv2.VideoCapture(
        'videos/pelotita_naranja_webcam/output.avi'
    )
    template = cv2.imread(
        'videos/pelotita_naranja_webcam/template_pelota.jpg'
    )
    follower = FollowerStaticAndRGBTemplate(
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
    follower = FollowerStaticAndRGBTemplate(
        img_provider,
        template,
        metodo_de_busqueda=BusquedaEnEspiralCambiandoFrameSize()
    )
    muestra_seguimiento = MuestraSeguimientoEnVivo(nombre='Seguimiento')
    FollowingScheme(img_provider, follower, muestra_seguimiento).run()


def seguir_taza():
    img_provider = FrameNamesAndImageProvider(
        'videos/rgbd/scenes/',  # scene path
        'desk',  # scene
        '1',  # scene number
        'videos/rgbd/objs/',  # object path
        'coffee_mug',  # object
        '5',  # object number
    )

    detector = RGBTemplateDetector()

    finder = BhattacharyyaHistogramFinder()

    follower = FollowerStaticAndRGBTemplate(img_provider, detector, finder)

    show_following = MuestraSeguimientoEnVivo(
        'Deteccion por template - Seguimiento por histograma'
    )

    FollowingScheme(
        img_provider,
        follower,
        show_following,
    ).run()


def seguir_pelota():
    img_provider = TemplateAndImageProviderFromVideo(
        video_path='videos/pelotita_naranja_webcam/output.avi',
        template_path='videos/pelotita_naranja_webcam/template_pelota.jpg',
    )

    detector = RGBTemplateDetector()

    finder = IntersectionHistogramFinder()

    follower = FollowerStaticAndRGBTemplate(img_provider, detector, finder)

    show_following = MuestraSeguimientoEnVivo(
        'Deteccion por template - Seguimiento por histograma'
    )

    FollowingScheme(
        img_provider,
        follower,
        show_following,
    ).run()


def seguir_taza_det_fija():
    img_provider = FrameNamesAndImageProvider(
        'videos/rgbd/scenes/',  # scene path
        'desk',  # scene
        '1',  # scene number
        'videos/rgbd/objs/',  # object path
        'coffee_mug',  # object
        '5',  # object number
    )

    detector = StaticDetector('videos/rgbd/scenes/desk/desk_1.mat', 'coffee_mug')

    finder = IntersectionHistogramFinder()

    follower = FollowerStaticAndRGBTemplate(img_provider, detector, finder)

    show_following = MuestraSeguimientoEnVivo(
        'Deteccion por template - Seguimiento por histograma'
    )

    FollowingScheme(
        img_provider,
        follower,
        show_following,
    ).run()


def seguir_gorra_det_fija():
    img_provider = FrameNamesAndImageProvider(
        'videos/rgbd/scenes/',  # scene path
        'desk',  # scene
        '1',  # scene number
        'videos/rgbd/objs/',  # object path
        'cap',  # object
        '4',  # object number
    )

    detector = StaticDetector('videos/rgbd/scenes/desk/desk_1.mat',
                              'cap')

    finder = IntersectionHistogramFinder()

    follower = FollowerStaticAndRGBTemplate(img_provider, detector, finder)

    show_following = MuestraSeguimientoEnVivo(
        'Deteccion por template - Seguimiento por histograma'
    )

    FollowingScheme(
        img_provider,
        follower,
        show_following,
    ).run()


if __name__ == '__main__':
    seguir_gorra_det_fija()