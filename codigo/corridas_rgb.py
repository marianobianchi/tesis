#coding=utf-8

from __future__ import (unicode_literals, division)


import cv2

from metodos_comunes import Timer
from buscadores import TemplateAndFrameHistogramFinder, HistogramComparator
from esquemas_seguimiento import FollowingScheme, FollowingSchemeSavingDataRGB,\
    FollowingSquemaExploringParameterRGB
from detectores import RGBTemplateDetector, StaticDetector, \
    StaticDetectorForRGBFinder
from metodos_de_busqueda import BusquedaAlrededor, \
    BusquedaAlrededorCambiandoFrameSize
from observar_seguimiento import MuestraSeguimientoEnVivo
from proveedores_de_imagenes import FrameNamesAndImageProviderPreChargedForRGB,\
    FrameNamesAndImageProvider, TemplateAndImageProviderFromVideo
from seguidores import FollowerStaticAndRGBTemplate, \
    FollowerStaticDetectionAndRGBTemplate


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

    # Detector
    detector = RGBTemplateDetector(
        template_threshold=0.16,
        templates_to_use=9,
        templates_sizes=[1],
        templates_from_frame=50,
    )

    # Buscador
    metodo_de_busqueda = BusquedaAlrededor()
    template_comparator = HistogramComparator(
        method=cv2.cv.CV_COMP_BHATTACHARYYA,
        threshold=0.6,
        reverse=False,
    )
    frame_comparator = HistogramComparator(
        method=cv2.cv.CV_COMP_BHATTACHARYYA,
        threshold=0.4,
        reverse=False,
    )

    finder = TemplateAndFrameHistogramFinder(
        template_comparator,
        frame_comparator,
        metodo_de_busqueda,
    )

    # Seguidor
    follower = FollowerStaticDetectionAndRGBTemplate(img_provider, detector, finder)

    # Muestra seguimiento
    show_following = MuestraSeguimientoEnVivo(
        'Deteccion por template - Seguimiento por histograma'
    )

    # Esquema
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

    # Detector
    detector = StaticDetectorForRGBFinder(
        'videos/rgbd/scenes/desk/desk_1.mat',
        'coffee_mug'
    )

    # Buscador
    metodo_de_busqueda = BusquedaAlrededorCambiandoFrameSize()
    template_comparator = HistogramComparator(
        method=cv2.cv.CV_COMP_BHATTACHARYYA,
        threshold=0.6,
        reverse=False,
    )
    frame_comparator = HistogramComparator(
        method=cv2.cv.CV_COMP_BHATTACHARYYA,
        threshold=0.4,
        reverse=False,
    )

    finder = TemplateAndFrameHistogramFinder(
        template_comparator,
        frame_comparator,
        metodo_de_busqueda,
    )

    # Seguidor
    follower = FollowerStaticDetectionAndRGBTemplate(img_provider, detector,
                                                     finder)

    show_following = MuestraSeguimientoEnVivo(
        'Deteccion por template - Seguimiento por histograma'
    )

    # FollowingSchemeSavingDataRGB(
    #     img_provider,
    #     follower,
    #     'pruebas_guardadas'
    # ).run()
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

    # Detector
    detector = StaticDetectorForRGBFinder(
        'videos/rgbd/scenes/desk/desk_1.mat',
        'cap'
    )

    # Buscador
    metodo_de_busqueda = BusquedaAlrededor()
    template_comparator = HistogramComparator(
        method=cv2.cv.CV_COMP_BHATTACHARYYA,
        threshold=0.6,
        reverse=False,
    )
    frame_comparator = HistogramComparator(
        method=cv2.cv.CV_COMP_BHATTACHARYYA,
        threshold=0.4,
        reverse=False,
    )

    finder = TemplateAndFrameHistogramFinder(
        template_comparator,
        frame_comparator,
        metodo_de_busqueda,
    )

    # Seguidor
    follower = FollowerStaticDetectionAndRGBTemplate(img_provider, detector, finder)

    show_following = MuestraSeguimientoEnVivo(
        'Deteccion por template - Seguimiento por histograma'
    )

    # FollowingSchemeSavingDataRGB(
    #     img_provider,
    #     follower,
    #     'pruebas_guardadas',
    # ).run()
    FollowingScheme(
        img_provider,
        follower,
        show_following,
    ).run()


def barrer_find_frame_threshold(objname, objnumber, scenename, scenenumber):
    # Set parameters values
    # RGBTemplateDetector parameters for training
    det_template_threshold = 0.16
    det_templates_to_use = 3
    det_template_sizes = [0.25, 0.5, 2]
    det_templates_from_frame = 1

    # Parametros para el seguimiento
    find_template_comp_method = cv2.cv.CV_COMP_BHATTACHARYYA
    find_template_threshold = 0.6
    find_template_reverse = False

    find_frame_comp_method = cv2.cv.CV_COMP_BHATTACHARYYA
    find_frame_threshold = 0.4
    find_frame_reverse = False

    metodo_de_busqueda = BusquedaAlrededor()

    # Create objects
    img_provider = FrameNamesAndImageProviderPreChargedForRGB(
        'videos/rgbd/scenes/', scenename, scenenumber,
        'videos/rgbd/objs/', objname, objnumber,
    )  # path, objname, number

    for find_frame_threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        with Timer('SEGUIMIENTO EN LA ESCENA ENTERA') as t:
            detector = RGBTemplateDetector(
                template_threshold=det_template_threshold,
                templates_to_use=det_templates_to_use,
                templates_sizes=det_template_sizes,
                templates_from_frame=det_templates_from_frame,
            )

            template_comparator = HistogramComparator(
                method=find_template_comp_method,
                threshold=find_template_threshold,
                reverse=find_template_reverse,
            )
            frame_comparator = HistogramComparator(
                method=find_frame_comp_method,
                threshold=find_frame_threshold,
                reverse=find_frame_reverse,
            )

            finder = TemplateAndFrameHistogramFinder(
                template_comparator,
                frame_comparator,
                metodo_de_busqueda,
            )

            follower = FollowerStaticAndRGBTemplate(img_provider, detector, finder)

            FollowingSquemaExploringParameterRGB(
                img_provider,
                follower,
                'pruebas_guardadas',
                'RGB_find_frame_threshold',
                find_frame_threshold,
            ).run()

            img_provider.restart()


def barrer_find_template_threshold(objname, objnumber, scenename, scenenumber):
    # Set parameters values
    # RGBTemplateDetector parameters for training
    det_template_threshold = 0.16
    det_templates_to_use = 3
    det_template_sizes = [0.25, 0.5, 2]
    det_templates_from_frame = 1

    # Parametros para el seguimiento
    find_template_comp_method = cv2.cv.CV_COMP_BHATTACHARYYA
    find_template_threshold = 0.6
    find_template_reverse = False

    find_frame_comp_method = cv2.cv.CV_COMP_BHATTACHARYYA
    find_frame_threshold = 0.4
    find_frame_reverse = False

    metodo_de_busqueda = BusquedaAlrededor()

    # Create objects
    img_provider = FrameNamesAndImageProviderPreChargedForRGB(
        'videos/rgbd/scenes/', scenename, scenenumber,
        'videos/rgbd/objs/', objname, objnumber,
    )  # path, objname, number

    for find_template_threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        with Timer('SEGUIMIENTO EN LA ESCENA ENTERA') as t:
            detector = RGBTemplateDetector(
                template_threshold=det_template_threshold,
                templates_to_use=det_templates_to_use,
                templates_sizes=det_template_sizes,
                templates_from_frame=det_templates_from_frame,
            )

            template_comparator = HistogramComparator(
                method=find_template_comp_method,
                threshold=find_template_threshold,
                reverse=find_template_reverse,
            )
            frame_comparator = HistogramComparator(
                method=find_frame_comp_method,
                threshold=find_frame_threshold,
                reverse=find_frame_reverse,
            )

            finder = TemplateAndFrameHistogramFinder(
                template_comparator,
                frame_comparator,
                metodo_de_busqueda,
            )

            follower = FollowerStaticAndRGBTemplate(img_provider, detector, finder)

            FollowingSquemaExploringParameterRGB(
                img_provider,
                follower,
                'pruebas_guardadas',
                'RGB_find_template_threshold',
                find_template_threshold,
            ).run()

            img_provider.restart()


def barrer_det_template_threshold(objname, objnumber, scenename, scenenumber):
    # Set parameters values
    # RGBTemplateDetector parameters for training
    det_template_threshold = 0.16
    det_templates_to_use = 3
    det_template_sizes = [0.25, 0.5, 2]
    det_templates_from_frame = 1

    # Parametros para el seguimiento
    find_template_comp_method = cv2.cv.CV_COMP_BHATTACHARYYA
    find_template_threshold = 0.6
    find_template_reverse = False

    find_frame_comp_method = cv2.cv.CV_COMP_BHATTACHARYYA
    find_frame_threshold = 0.4
    find_frame_reverse = False

    metodo_de_busqueda = BusquedaAlrededor()

    # Create objects
    img_provider = FrameNamesAndImageProviderPreChargedForRGB(
        'videos/rgbd/scenes/', scenename, scenenumber,
        'videos/rgbd/objs/', objname, objnumber,
    )  # path, objname, number

    for det_template_threshold in [0.15, 0.25]:
        with Timer('SEGUIMIENTO EN LA ESCENA ENTERA') as t:
            detector = RGBTemplateDetector(
                template_threshold=det_template_threshold,
                templates_to_use=det_templates_to_use,
                templates_sizes=det_template_sizes,
                templates_from_frame=det_templates_from_frame,
            )

            template_comparator = HistogramComparator(
                method=find_template_comp_method,
                threshold=find_template_threshold,
                reverse=find_template_reverse,
            )
            frame_comparator = HistogramComparator(
                method=find_frame_comp_method,
                threshold=find_frame_threshold,
                reverse=find_frame_reverse,
            )

            finder = TemplateAndFrameHistogramFinder(
                template_comparator,
                frame_comparator,
                metodo_de_busqueda,
            )

            follower = FollowerStaticAndRGBTemplate(img_provider, detector, finder)

            FollowingSquemaExploringParameterRGB(
                img_provider,
                follower,
                'pruebas_guardadas',
                'RGB_det_template_threshold',
                det_template_threshold,
            ).run()

            img_provider.restart()


def barrer_det_template_sizes(objname, objnumber, scenename, scenenumber):
    # Set parameters values
    # RGBTemplateDetector parameters for training
    det_template_threshold = 0.16
    det_templates_to_use = 3
    det_template_sizes = [0.25, 0.5, 2]
    det_templates_from_frame = 1

    # Parametros para el seguimiento
    find_template_comp_method = cv2.cv.CV_COMP_BHATTACHARYYA
    find_template_threshold = 0.6
    find_template_reverse = False

    find_frame_comp_method = cv2.cv.CV_COMP_BHATTACHARYYA
    find_frame_threshold = 0.4
    find_frame_reverse = False

    metodo_de_busqueda = BusquedaAlrededor()

    # Create objects
    img_provider = FrameNamesAndImageProviderPreChargedForRGB(
        'videos/rgbd/scenes/', scenename, scenenumber,
        'videos/rgbd/objs/', objname, objnumber,
    )  # path, objname, number

    for det_template_sizes in [[0.75, 1.5], [0.5, 0.75, 1.5], [0.6, 0.8, 1.2]]:
        with Timer('SEGUIMIENTO EN LA ESCENA ENTERA') as t:
            detector = RGBTemplateDetector(
                template_threshold=det_template_threshold,
                templates_to_use=det_templates_to_use,
                templates_sizes=det_template_sizes,
                templates_from_frame=det_templates_from_frame,
            )

            template_comparator = HistogramComparator(
                method=find_template_comp_method,
                threshold=find_template_threshold,
                reverse=find_template_reverse,
            )
            frame_comparator = HistogramComparator(
                method=find_frame_comp_method,
                threshold=find_frame_threshold,
                reverse=find_frame_reverse,
            )

            finder = TemplateAndFrameHistogramFinder(
                template_comparator,
                frame_comparator,
                metodo_de_busqueda,
            )

            follower = FollowerStaticAndRGBTemplate(img_provider,
                                                    detector,
                                                    finder)

            FollowingSquemaExploringParameterRGB(
                img_provider,
                follower,
                'pruebas_guardadas',
                'RGB_det_template_sizes',
                '_'.join(det_template_sizes),
            ).run()

            img_provider.restart()


if __name__ == '__main__':
    # barrer_find_frame_threshold('coffee_mug', '5', 'desk', '1')
    # barrer_find_frame_threshold('cap', '4', 'desk', '1')
    # barrer_find_frame_threshold('bowl', '3', 'desk', '2')

    # barrer_find_template_threshold('coffee_mug', '5', 'desk', '1')
    # barrer_find_template_threshold('cap', '4', 'desk', '1')
    # barrer_find_template_threshold('bowl', '3', 'desk', '2')

    # barrer_det_template_threshold('coffee_mug', '5', 'desk', '1')
    # barrer_det_template_threshold('cap', '4', 'desk', '1')
    # barrer_det_template_threshold('bowl', '3', 'desk', '2')

    # barrer_det_template_sizes('coffee_mug', '5', 'desk', '1')
    # barrer_det_template_sizes('cap', '4', 'desk', '1')
    # barrer_det_template_sizes('bowl', '3', 'desk', '2')

    seguir_taza_det_fija()
    # seguir_gorra_det_fija()
