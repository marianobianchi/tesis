#coding=utf-8

from __future__ import (unicode_literals, division)


import cv2

from buscadores import TemplateAndFrameHistogramFinder
from esquemas_seguimiento import FollowingScheme, \
    FollowingSquemaExploringParameterRGB
from detectores import RGBTemplateDetector, StaticDetector
from metodos_de_busqueda import BusquedaEnEspiralCambiandoFrameSize
from observar_seguimiento import MuestraSeguimientoEnVivo
from proveedores_de_imagenes import FrameNamesAndImageProviderPreChargedForRGB,\
    FrameNamesAndImageProvider, TemplateAndImageProviderFromVideo
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

    finder = TemplateAndFrameHistogramFinder()

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

    finder = TemplateAndFrameHistogramFinder()

    follower = FollowerStaticAndRGBTemplate(img_provider, detector, finder)

    show_following = MuestraSeguimientoEnVivo(
        'Deteccion por template - Seguimiento por histograma'
    )

    FollowingScheme(
        img_provider,
        follower,
        show_following,
    ).run()


def seguir_gorra():
    img_provider = FrameNamesAndImageProvider(
        'videos/rgbd/scenes/',  # scene path
        'desk',  # scene
        '1',  # scene number
        'videos/rgbd/objs/',  # object path
        'cap',  # object
        '4',  # object number
    )

    detector = RGBTemplateDetector()

    finder = TemplateAndFrameHistogramFinder()

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

    detector = StaticDetector(
        'videos/rgbd/scenes/desk/desk_1.mat',
        'coffee_mug'
    )

    finder = TemplateAndFrameHistogramFinder()

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

    detector = StaticDetector(
        'videos/rgbd/scenes/desk/desk_1.mat',
        'cap'
    )

    finder = TemplateAndFrameHistogramFinder()

    follower = FollowerStaticAndRGBTemplate(img_provider, detector, finder)

    show_following = MuestraSeguimientoEnVivo(
        'Deteccion por template - Seguimiento por histograma'
    )

    FollowingScheme(
        img_provider,
        follower,
        show_following,
    ).run()


def barrer_find_percentage_object(objname, objnumber, scenename, scenenumber):
    # Set parameters values
    # RGBTemplateDetector parameters for training
    det_template_threshold = 0.16
    det_templates_to_use = 3
    det_template_sizes = [0.25, 0.5, 2]
    det_templates_from_frame = 1

    # Parametros para el seguimiento
    # TODO: los parametros son los metodos de comparacion para el template y
    # para el frame a frame, los umbrales de comparacion para ambos metodos y
    # el tipo de busqueda (buscar cambiando de tamaño)
    # IDEA: hacer un objeto al que se le defina el metodo de comparacion y el
    # umbral y que decida dada la imagen "vieja" y la "nueva" si matchean o no

    # Create objects
    img_provider = FrameNamesAndImageProviderPreChargedForRGB(
        'videos/rgbd/scenes/', scenename, scenenumber,
        'videos/rgbd/objs/', objname, objnumber,
    )  # path, objname, number

    for find_perc_obj_model_points in [1]:
        detector = RGBTemplateDetector(
            template_threshold=det_template_threshold,
            templates_to_use=det_templates_to_use,
            templates_sizes=det_template_sizes,
            templates_from_frame=det_templates_from_frame,
        )

        finder = TemplateAndFrameHistogramFinder()

        follower = FollowerStaticAndRGBTemplate(img_provider, detector, finder)

        FollowingSquemaExploringParameterRGB(
            img_provider,
            follower,
            'pruebas_guardadas',
            'PROBANDO_RGB',
            find_perc_obj_model_points,
        ).run()

        img_provider.restart()


if __name__ == '__main__':
    barrer_find_percentage_object('cap', '4', 'desk', '1')