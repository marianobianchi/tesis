#coding=utf-8

from __future__ import (unicode_literals, division)


import cv2
from scipy.spatial import distance as dist

from metodos_comunes import Timer
from buscadores import TemplateAndFrameHistogramFinder, HistogramComparator, \
    TemplateAndFrameGreenHistogramFinder, HSHistogramFinder,\
    FragmentedHistogramFinder, FragmentedReverseCompHistogramFinder

from esquemas_seguimiento import FollowingScheme, FollowingSchemeSavingDataRGB,\
    FollowingSquemaExploringParameterRGB
from detectores import RGBTemplateDetector, StaticDetectorForRGBFinder
from metodos_de_busqueda import BusquedaAlrededor, \
    BusquedaAlrededorCambiandoFrameSize
from observar_seguimiento import MuestraSeguimientoEnVivo
from proveedores_de_imagenes import FrameNamesAndImageProviderPreChargedForRGB,\
    FrameNamesAndImageProvider, TemplateAndImageProviderFromVideo
from seguidores import RGBFollower


def seguir_pelota_naranja_version5():
    img_provider = cv2.VideoCapture(
        'videos/pelotita_naranja_webcam/output.avi'
    )
    template = cv2.imread(
        'videos/pelotita_naranja_webcam/template_pelota.jpg'
    )
    follower = RGBFollower(
        img_provider,
        template,
        metodo_de_busqueda=BusquedaAlrededorCambiandoFrameSize()
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
    follower = RGBFollower(
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
    follower = RGBFollower(img_provider, detector, finder)

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

    follower = RGBFollower(img_provider, detector, finder)

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

    follower = RGBFollower(img_provider, detector, finder)

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
        method=cv2.cv.CV_COMP_CHISQR,
        threshold=0.9,
        reverse=False,
    )
    frame_comparator = HistogramComparator(
        method=cv2.cv.CV_COMP_CHISQR,
        threshold=0.85,
        reverse=False,
    )

    finder = HSHistogramFinder(
        template_comparator,
        frame_comparator,
        fixed_frame_value=True,
        metodo_de_busqueda=metodo_de_busqueda,
    )

    # Seguidor
    follower = RGBFollower(img_provider, detector, finder)

    show_following = MuestraSeguimientoEnVivo(
        'Deteccion estatica - Seguimiento por histograma'
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
    metodo_de_busqueda = BusquedaAlrededorCambiandoFrameSize()
    template_comparator = HistogramComparator(
        method=cv2.cv.CV_COMP_CORREL,
        threshold=0.8,
        reverse=True,
    )
    frame_comparator = HistogramComparator(
        method=cv2.cv.CV_COMP_CORREL,
        threshold=0.9,
        reverse=True,
    )

    finder = TemplateAndFrameLearningBaseComparissonHistogramFinder(
        template_comparator,
        frame_comparator,
        fixed_frame_value=False,
        metodo_de_busqueda=metodo_de_busqueda,
    )

    # Seguidor
    follower = RGBFollower(img_provider, detector, finder)

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

################################################################################
# TODO: Correr todos los que dicen "barrer" pero con el detector estatico
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

            follower = RGBFollower(img_provider, detector, finder)

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

            follower = RGBFollower(img_provider, detector, finder)

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

            follower = RGBFollower(img_provider, detector, finder)

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

            follower = RGBFollower(
                img_provider,
                detector,
                finder
            )

            FollowingSquemaExploringParameterRGB(
                img_provider,
                follower,
                'pruebas_guardadas',
                'RGB_det_template_sizes',
                '_'.join([unicode(t) for t in det_template_sizes]),
            ).run()

            img_provider.restart()

################################################################################

def correr_battachayyra_verde_find_template_threshold(objname, objnumber,
                                                      scenename, scenenumber):
    # Parametros para el seguimiento
    find_template_comp_method = cv2.cv.CV_COMP_BHATTACHARYYA
    template_perc = 0.5
    template_worst = 1
    find_template_reverse = False

    find_frame_comp_method = cv2.cv.CV_COMP_BHATTACHARYYA
    frame_perc = 0.4
    frame_worst = 1
    find_frame_reverse = False

    metodo_de_busqueda = BusquedaAlrededorCambiandoFrameSize()

    # Create objects
    img_provider = FrameNamesAndImageProviderPreChargedForRGB(
        'videos/rgbd/scenes/', scenename, scenenumber,
        'videos/rgbd/objs/', objname, objnumber,
    )  # path, objname, number

    for template_perc in [0.55, 0.65]:#[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        detector = StaticDetectorForRGBFinder(
            matfile_path=('videos/rgbd/scenes/{sname}/{sname}_{snum}.mat'
                          .format(sname=scenename, snum=scenenumber)),
            obj_rgbd_name=objname,
        )

        template_comparator = HistogramComparator(
            method=find_template_comp_method,
            perc=template_perc,
            worst_case=template_worst,
            reverse=find_template_reverse,
        )
        frame_comparator = HistogramComparator(
            method=find_frame_comp_method,
            perc=frame_perc,
            worst_case=frame_worst,
            reverse=find_frame_reverse,
        )

        finder = TemplateAndFrameGreenHistogramFinder(
            template_comparator,
            frame_comparator,
            metodo_de_busqueda,
        )

        follower = RGBFollower(
            img_provider,
            detector,
            finder
        )

        FollowingSquemaExploringParameterRGB(
            img_provider,
            follower,
            'pruebas_guardadas',
            'batta_green_channel_find_template_threshold',
            template_perc,
        ).run()

        img_provider.restart()


def correr_battachayyra_verde_find_frame_threshold(objname, objnumber,
                                                   scenename, scenenumber):
    # Parametros para el seguimiento
    find_template_comp_method = cv2.cv.CV_COMP_BHATTACHARYYA
    template_perc = 0.5
    template_worst = 1
    find_template_reverse = False

    find_frame_comp_method = cv2.cv.CV_COMP_BHATTACHARYYA
    frame_perc = 0.4
    frame_worst = 1
    find_frame_reverse = False

    metodo_de_busqueda = BusquedaAlrededorCambiandoFrameSize()

    # Create objects
    img_provider = FrameNamesAndImageProviderPreChargedForRGB(
        'videos/rgbd/scenes/', scenename, scenenumber,
        'videos/rgbd/objs/', objname, objnumber,
    )  # path, objname, number

    for frame_perc in [0.15, 0.25, 0.35, 0.45]:#[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        detector = StaticDetectorForRGBFinder(
            matfile_path=('videos/rgbd/scenes/{sname}/{sname}_{snum}.mat'
                          .format(sname=scenename, snum=scenenumber)),
            obj_rgbd_name=objname,
        )

        template_comparator = HistogramComparator(
            method=find_template_comp_method,
            perc=template_perc,
            worst_case=template_worst,
            reverse=find_template_reverse,
        )
        frame_comparator = HistogramComparator(
            method=find_frame_comp_method,
            perc=frame_perc,
            worst_case=frame_worst,
            reverse=find_frame_reverse,
        )

        finder = TemplateAndFrameGreenHistogramFinder(
            template_comparator,
            frame_comparator,
            metodo_de_busqueda,
        )

        follower = RGBFollower(
            img_provider,
            detector,
            finder
        )

        FollowingSquemaExploringParameterRGB(
            img_provider,
            follower,
            'pruebas_guardadas',
            'batta_green_channel_find_frame_threshold',
            frame_perc,
        ).run()

        img_provider.restart()


def correr_chi_squared_verde_find_template_threshold(objname, objnumber,
                                                     scenename, scenenumber):
    # Parametros para el seguimiento
    find_template_comp_method = cv2.cv.CV_COMP_CHISQR
    find_template_perc = 0.5
    template_worst = 300
    find_template_reverse = False

    find_frame_comp_method = cv2.cv.CV_COMP_CHISQR
    find_frame_perc = 0.15
    frame_worst = 100
    find_frame_reverse = False

    metodo_de_busqueda = BusquedaAlrededorCambiandoFrameSize()

    # Create objects
    img_provider = FrameNamesAndImageProviderPreChargedForRGB(
        'videos/rgbd/scenes/', scenename, scenenumber,
        'videos/rgbd/objs/', objname, objnumber,
    )  # path, objname, number

    for find_template_perc in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1]:
        detector = StaticDetectorForRGBFinder(
            matfile_path=('videos/rgbd/scenes/{sname}/{sname}_{snum}.mat'
                          .format(sname=scenename, snum=scenenumber)),
            obj_rgbd_name=objname,
        )

        template_comparator = HistogramComparator(
            method=find_template_comp_method,
            perc=find_template_perc,
            worst_case=template_worst,
            reverse=find_template_reverse,
        )
        frame_comparator = HistogramComparator(
            method=find_frame_comp_method,
            perc=find_frame_perc,
            worst_case=frame_worst,
            reverse=find_frame_reverse,
        )

        finder = TemplateAndFrameGreenHistogramFinder(
            template_comparator,
            frame_comparator,
            metodo_de_busqueda,
        )

        follower = RGBFollower(
            img_provider,
            detector,
            finder
        )

        FollowingSquemaExploringParameterRGB(
            img_provider,
            follower,
            'pruebas_guardadas',
            'chisquared_green_channel_find_template_threshold',
            find_template_perc,
        ).run()

        img_provider.restart()


def correr_chi_squared_verde_find_frame_threshold(objname, objnumber,
                                                  scenename, scenenumber):
    # Parametros para el seguimiento
    find_template_comp_method = cv2.cv.CV_COMP_CHISQR
    find_template_perc = 0.5
    template_worst = 300
    find_template_reverse = False

    find_frame_comp_method = cv2.cv.CV_COMP_CHISQR
    find_frame_perc = 0.15
    frame_worst = 100
    find_frame_reverse = False

    metodo_de_busqueda = BusquedaAlrededorCambiandoFrameSize()

    # Create objects
    img_provider = FrameNamesAndImageProviderPreChargedForRGB(
        'videos/rgbd/scenes/', scenename, scenenumber,
        'videos/rgbd/objs/', objname, objnumber,
    )  # path, objname, number

    for find_frame_perc in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1]:
        detector = StaticDetectorForRGBFinder(
            matfile_path=('videos/rgbd/scenes/{sname}/{sname}_{snum}.mat'
                          .format(sname=scenename, snum=scenenumber)),
            obj_rgbd_name=objname,
        )

        template_comparator = HistogramComparator(
            method=find_template_comp_method,
            perc=find_template_perc,
            worst_case=template_worst,
            reverse=find_template_reverse,
        )
        frame_comparator = HistogramComparator(
            method=find_frame_comp_method,
            perc=find_frame_perc,
            worst_case=frame_worst,
            reverse=find_frame_reverse,
        )

        finder = TemplateAndFrameGreenHistogramFinder(
            template_comparator,
            frame_comparator,
            metodo_de_busqueda,
        )

        follower = RGBFollower(
            img_provider,
            detector,
            finder
        )

        FollowingSquemaExploringParameterRGB(
            img_provider,
            follower,
            'pruebas_guardadas',
            'chisquared_green_channel_find_frame_threshold',
            find_frame_perc,
        ).run()

        img_provider.restart()

def correr_correlation_verde_find_template_threshold(objname, objnumber,
                                                     scenename, scenenumber):
    # Parametros para el seguimiento
    find_template_comp_method = cv2.cv.CV_COMP_CORREL
    find_template_perc = 500
    template_worst = 0.001
    find_template_reverse = True

    find_frame_comp_method = cv2.cv.CV_COMP_CORREL
    find_frame_perc = 700
    frame_worst = 0.001
    find_frame_reverse = True

    metodo_de_busqueda = BusquedaAlrededorCambiandoFrameSize()

    # Create objects
    img_provider = FrameNamesAndImageProviderPreChargedForRGB(
        'videos/rgbd/scenes/', scenename, scenenumber,
        'videos/rgbd/objs/', objname, objnumber,
    )  # path, objname, number

    for find_template_perc in [200, 300, 400, 500, 600, 700, 800, 900]:
        detector = StaticDetectorForRGBFinder(
            matfile_path=('videos/rgbd/scenes/{sname}/{sname}_{snum}.mat'
                          .format(sname=scenename, snum=scenenumber)),
            obj_rgbd_name=objname,
        )

        template_comparator = HistogramComparator(
            method=find_template_comp_method,
            perc=find_template_perc,
            worst_case=template_worst,
            reverse=find_template_reverse,
        )
        frame_comparator = HistogramComparator(
            method=find_frame_comp_method,
            perc=find_frame_perc,
            worst_case=frame_worst,
            reverse=find_frame_reverse,
        )

        finder = TemplateAndFrameGreenHistogramFinder(
            template_comparator,
            frame_comparator,
            metodo_de_busqueda,
        )

        follower = RGBFollower(
            img_provider,
            detector,
            finder
        )

        FollowingSquemaExploringParameterRGB(
            img_provider,
            follower,
            'pruebas_guardadas',
            'correl_green_channel_find_template_threshold',
            find_template_perc,
        ).run()

        img_provider.restart()


def correr_correlation_verde_find_frame_threshold(objname, objnumber,
                                                  scenename, scenenumber):
    # Parametros para el seguimiento
    find_template_comp_method = cv2.cv.CV_COMP_CORREL
    find_template_perc = 500
    template_worst = 0.001
    find_template_reverse = True

    find_frame_comp_method = cv2.cv.CV_COMP_CORREL
    find_frame_perc = 700
    frame_worst = 0.001
    find_frame_reverse = True

    metodo_de_busqueda = BusquedaAlrededorCambiandoFrameSize()

    # Create objects
    img_provider = FrameNamesAndImageProviderPreChargedForRGB(
        'videos/rgbd/scenes/', scenename, scenenumber,
        'videos/rgbd/objs/', objname, objnumber,
    )  # path, objname, number

    for find_frame_perc in [850, 950]:#[200, 300, 400, 500, 600, 700, 800, 900]:
        detector = StaticDetectorForRGBFinder(
            matfile_path=('videos/rgbd/scenes/{sname}/{sname}_{snum}.mat'
                          .format(sname=scenename, snum=scenenumber)),
            obj_rgbd_name=objname,
        )

        template_comparator = HistogramComparator(
            method=find_template_comp_method,
            perc=find_template_perc,
            worst_case=template_worst,
            reverse=find_template_reverse,
        )
        frame_comparator = HistogramComparator(
            method=find_frame_comp_method,
            perc=find_frame_perc,
            worst_case=frame_worst,
            reverse=find_frame_reverse,
        )

        finder = TemplateAndFrameGreenHistogramFinder(
            template_comparator,
            frame_comparator,
            metodo_de_busqueda,
        )

        follower = RGBFollower(
            img_provider,
            detector,
            finder
        )

        FollowingSquemaExploringParameterRGB(
            img_provider,
            follower,
            'pruebas_guardadas',
            'correl_green_channel_find_frame_threshold',
            find_frame_perc,
        ).run()

        img_provider.restart()


def correr_chi_squared_hs_find_template_threshold(objname, objnumber,
                                                  scenename, scenenumber):
    # Parametros para el seguimiento
    find_template_comp_method = cv2.cv.CV_COMP_CHISQR
    find_template_perc = 0.5
    template_worst = 300
    find_template_reverse = False

    find_frame_comp_method = cv2.cv.CV_COMP_CHISQR
    find_frame_perc = 0.15
    frame_worst = 100
    find_frame_reverse = False

    metodo_de_busqueda = BusquedaAlrededorCambiandoFrameSize()

    # Create objects
    img_provider = FrameNamesAndImageProviderPreChargedForRGB(
        'videos/rgbd/scenes/', scenename, scenenumber,
        'videos/rgbd/objs/', objname, objnumber,
    )  # path, objname, number

    for find_template_perc in [0.01, 0.02, 0.03, 0.04]:
        detector = StaticDetectorForRGBFinder(
            matfile_path=('videos/rgbd/scenes/{sname}/{sname}_{snum}.mat'
                          .format(sname=scenename, snum=scenenumber)),
            obj_rgbd_name=objname,
        )

        template_comparator = HistogramComparator(
            method=find_template_comp_method,
            perc=find_template_perc,
            worst_case=template_worst,
            reverse=find_template_reverse,
        )
        frame_comparator = HistogramComparator(
            method=find_frame_comp_method,
            perc=find_frame_perc,
            worst_case=frame_worst,
            reverse=find_frame_reverse,
        )

        finder = HSHistogramFinder(
            template_comparator,
            frame_comparator,
            metodo_de_busqueda,
        )

        follower = RGBFollower(
            img_provider,
            detector,
            finder
        )

        FollowingSquemaExploringParameterRGB(
            img_provider,
            follower,
            'pruebas_guardadas',
            'chisquared_hs_channels_find_template_threshold_fixed',
            find_template_perc,
        ).run()

        img_provider.restart()


def correr_chi_squared_hs_find_frame_threshold(objname, objnumber,
                                               scenename, scenenumber):
    # Parametros para el seguimiento
    find_template_comp_method = cv2.cv.CV_COMP_CHISQR
    find_template_perc = 0.5
    template_worst = 300
    find_template_reverse = False

    find_frame_comp_method = cv2.cv.CV_COMP_CHISQR
    find_frame_perc = 0.15
    frame_worst = 100
    find_frame_reverse = False

    metodo_de_busqueda = BusquedaAlrededorCambiandoFrameSize()

    # Create objects
    img_provider = FrameNamesAndImageProviderPreChargedForRGB(
        'videos/rgbd/scenes/', scenename, scenenumber,
        'videos/rgbd/objs/', objname, objnumber,
    )  # path, objname, number

    for find_frame_perc in [0.01, 0.02, 0.03, 0.04]:
        detector = StaticDetectorForRGBFinder(
            matfile_path=('videos/rgbd/scenes/{sname}/{sname}_{snum}.mat'
                          .format(sname=scenename, snum=scenenumber)),
            obj_rgbd_name=objname,
        )

        template_comparator = HistogramComparator(
            method=find_template_comp_method,
            perc=find_template_perc,
            worst_case=template_worst,
            reverse=find_template_reverse,
        )
        frame_comparator = HistogramComparator(
            method=find_frame_comp_method,
            perc=find_frame_perc,
            worst_case=frame_worst,
            reverse=find_frame_reverse,
        )

        finder = HSHistogramFinder(
            template_comparator,
            frame_comparator,
            metodo_de_busqueda,
        )

        follower = RGBFollower(
            img_provider,
            detector,
            finder
        )

        FollowingSquemaExploringParameterRGB(
            img_provider,
            follower,
            'pruebas_guardadas',
            'chisquared_hs_channels_find_frame_threshold_fixed',
            find_frame_perc,
        ).run()

        img_provider.restart()


def correr_mi_metodo_bhatta_bhatta_bhatta_template_perc(objname, objnumber,
                                                        scenename, scenenumber):
    img_provider = FrameNamesAndImageProviderPreChargedForRGB(
        'videos/rgbd/scenes/',  # scene path
        scenename,  # scene
        scenenumber,  # scene number
        'videos/rgbd/objs/',  # object path
        objname,  # object
        objnumber,  # object number
    )

    # Detector
    detector = StaticDetectorForRGBFinder(
        matfile_path=('videos/rgbd/scenes/{sname}/{sname}_{snum}.mat'
                      .format(sname=scenename, snum=scenenumber)),
        obj_rgbd_name=objname,
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
    template_perc = 0.5
    frame_perc = 0.2

    for template_perc in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        finder = FragmentedHistogramFinder(
            channels_comparators=channels_comparators,
            center_point=center_point,
            extern_point=extern_point,
            distance_comparator=dist.euclidean,
            template_perc=template_perc,
            frame_perc=frame_perc,
            metodo_de_busqueda=metodo_de_busqueda,
        )

        # Seguidor
        follower = RGBFollower(img_provider, detector, finder)

        FollowingSquemaExploringParameterRGB(
            img_provider,
            follower,
            'pruebas_guardadas',
            'mi_metodo_bhatta_bhatta_bhatta_template_perc',
            template_perc,
        ).run()

        img_provider.restart()


def correr_mi_metodo_bhatta_bhatta_bhatta_frame_perc(objname, objnumber,
                                                     scenename, scenenumber):

    img_provider = FrameNamesAndImageProviderPreChargedForRGB(
        'videos/rgbd/scenes/',  # scene path
        scenename,  # scene
        scenenumber,  # scene number
        'videos/rgbd/objs/',  # object path
        objname,  # object
        objnumber,  # object number
    )

    # Detector
    detector = StaticDetectorForRGBFinder(
        matfile_path=('videos/rgbd/scenes/{sname}/{sname}_{snum}.mat'
                      .format(sname=scenename, snum=scenenumber)),
        obj_rgbd_name=objname,
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
    template_perc = 0.5
    frame_perc = 0.2

    for frame_perc in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        finder = FragmentedHistogramFinder(
            channels_comparators=channels_comparators,
            center_point=center_point,
            extern_point=extern_point,
            distance_comparator=dist.euclidean,
            template_perc=template_perc,
            frame_perc=frame_perc,
            metodo_de_busqueda=metodo_de_busqueda,
        )

        # Seguidor
        follower = RGBFollower(img_provider, detector, finder)

        FollowingSquemaExploringParameterRGB(
            img_provider,
            follower,
            'pruebas_guardadas',
            'mi_metodo_bhatta_bhatta_bhatta_frame_perc',
            frame_perc,
        ).run()

        img_provider.restart()


def correr_mi_metodo_chi_chi_bhatta_template_perc(objname, objnumber,
                                                  scenename, scenenumber):
    img_provider = FrameNamesAndImageProviderPreChargedForRGB(
        'videos/rgbd/scenes/',  # scene path
        scenename,  # scene
        scenenumber,  # scene number
        'videos/rgbd/objs/',  # object path
        objname,  # object
        objnumber,  # object number
    )

    # Detector
    detector = StaticDetectorForRGBFinder(
        matfile_path=('videos/rgbd/scenes/{sname}/{sname}_{snum}.mat'
                      .format(sname=scenename, snum=scenenumber)),
        obj_rgbd_name=objname,
    )

    # Buscador
    metodo_de_busqueda = BusquedaAlrededorCambiandoFrameSize()
    channels_comparators = [
        (0, 40, 180, cv2.cv.CV_COMP_CHISQR),
        (1, 60, 256, cv2.cv.CV_COMP_CHISQR),
        (2, 60, 256, cv2.cv.CV_COMP_BHATTACHARYYA),
    ]
    center_point = [0, 0, 0]
    extern_point = [500, 500, 1]
    template_perc = 0.5
    frame_perc = 0.2

    for template_perc in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        finder = FragmentedHistogramFinder(
            channels_comparators=channels_comparators,
            center_point=center_point,
            extern_point=extern_point,
            distance_comparator=dist.euclidean,
            template_perc=template_perc,
            frame_perc=frame_perc,
            metodo_de_busqueda=metodo_de_busqueda,
        )

        # Seguidor
        follower = RGBFollower(img_provider, detector, finder)

        FollowingSquemaExploringParameterRGB(
            img_provider,
            follower,
            'pruebas_guardadas',
            'mi_metodo_chi_chi_bhatta_template_perc',
            template_perc,
        ).run()

        img_provider.restart()


def correr_mi_metodo_chi_chi_bhatta_frame_perc(objname, objnumber,
                                               scenename, scenenumber):
    img_provider = FrameNamesAndImageProviderPreChargedForRGB(
        'videos/rgbd/scenes/',  # scene path
        scenename,  # scene
        scenenumber,  # scene number
        'videos/rgbd/objs/',  # object path
        objname,  # object
        objnumber,  # object number
    )

    # Detector
    detector = StaticDetectorForRGBFinder(
        matfile_path=('videos/rgbd/scenes/{sname}/{sname}_{snum}.mat'
                      .format(sname=scenename, snum=scenenumber)),
        obj_rgbd_name=objname,
    )

    # Buscador
    metodo_de_busqueda = BusquedaAlrededorCambiandoFrameSize()
    channels_comparators = [
        (0, 40, 180, cv2.cv.CV_COMP_CHISQR),
        (1, 60, 256, cv2.cv.CV_COMP_CHISQR),
        (2, 60, 256, cv2.cv.CV_COMP_BHATTACHARYYA),
    ]
    center_point = [0, 0, 0]
    extern_point = [500, 500, 1]
    template_perc = 0.5
    frame_perc = 0.2

    for frame_perc in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        finder = FragmentedHistogramFinder(
            channels_comparators=channels_comparators,
            center_point=center_point,
            extern_point=extern_point,
            distance_comparator=dist.euclidean,
            template_perc=template_perc,
            frame_perc=frame_perc,
            metodo_de_busqueda=metodo_de_busqueda,
        )

        # Seguidor
        follower = RGBFollower(img_provider, detector, finder)

        FollowingSquemaExploringParameterRGB(
            img_provider,
            follower,
            'pruebas_guardadas',
            'mi_metodo_chi_chi_bhatta_frame_perc',
            frame_perc,
        ).run()

        img_provider.restart()


def correr_mi_metodo_bhatta_inter_inter_template_perc(objname, objnumber,
                                                      scenename, scenenumber):
    img_provider = FrameNamesAndImageProviderPreChargedForRGB(
        'videos/rgbd/scenes/',  # scene path
        scenename,  # scene
        scenenumber,  # scene number
        'videos/rgbd/objs/',  # object path
        objname,  # object
        objnumber,  # object number
    )

    # Detector
    detector = StaticDetectorForRGBFinder(
        matfile_path=('videos/rgbd/scenes/{sname}/{sname}_{snum}.mat'
                      .format(sname=scenename, snum=scenenumber)),
        obj_rgbd_name=objname,
    )

    # Buscador
    metodo_de_busqueda = BusquedaAlrededorCambiandoFrameSize()
    channels_comparators = [
        (0, 40, 180, cv2.cv.CV_COMP_BHATTACHARYYA),
        (1, 60, 256, cv2.cv.CV_COMP_INTERSECT),
        (2, 60, 256, cv2.cv.CV_COMP_INTERSECT),
    ]
    center_point = [0, 0, 0]
    extern_point = [1, 5, 5]
    template_perc = 0.3
    frame_perc = 0.15

    for template_perc in [0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        finder = FragmentedReverseCompHistogramFinder(
            channels_comparators=channels_comparators,
            center_point=center_point,
            extern_point=extern_point,
            distance_comparator=dist.euclidean,
            template_perc=template_perc,
            frame_perc=frame_perc,
            reverse_comp=[False, True, True],
            metodo_de_busqueda=metodo_de_busqueda,
        )

        # Seguidor
        follower = RGBFollower(img_provider, detector, finder)

        FollowingSquemaExploringParameterRGB(
            img_provider,
            follower,
            'pruebas_guardadas',
            'mi_metodo_bhatta_inter_inter_template_perc',
            template_perc,
        ).run()

        img_provider.restart()


def correr_mi_metodo_bhatta_inter_inter_frame_perc(objname, objnumber,
                                                   scenename, scenenumber):
    img_provider = FrameNamesAndImageProviderPreChargedForRGB(
        'videos/rgbd/scenes/',  # scene path
        scenename,  # scene
        scenenumber,  # scene number
        'videos/rgbd/objs/',  # object path
        objname,  # object
        objnumber,  # object number
    )

    # Detector
    detector = StaticDetectorForRGBFinder(
        matfile_path=('videos/rgbd/scenes/{sname}/{sname}_{snum}.mat'
                      .format(sname=scenename, snum=scenenumber)),
        obj_rgbd_name=objname,
    )

    # Buscador
    metodo_de_busqueda = BusquedaAlrededorCambiandoFrameSize()
    channels_comparators = [
        (0, 40, 180, cv2.cv.CV_COMP_BHATTACHARYYA),
        (1, 60, 256, cv2.cv.CV_COMP_INTERSECT),
        (2, 60, 256, cv2.cv.CV_COMP_INTERSECT),
    ]
    center_point = [0, 0, 0]
    extern_point = [1, 5, 5]
    template_perc = 0.3
    frame_perc = 0.15

    for frame_perc in [0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6]:
        finder = FragmentedReverseCompHistogramFinder(
            channels_comparators=channels_comparators,
            center_point=center_point,
            extern_point=extern_point,
            distance_comparator=dist.euclidean,
            template_perc=template_perc,
            frame_perc=frame_perc,
            reverse_comp=[False, True, True],
            metodo_de_busqueda=metodo_de_busqueda,
        )

        # Seguidor
        follower = RGBFollower(img_provider, detector, finder)

        FollowingSquemaExploringParameterRGB(
            img_provider,
            follower,
            'pruebas_guardadas',
            'mi_metodo_bhatta_inter_inter_frame_perc',
            frame_perc,
        ).run()

        img_provider.restart()

###
# RGB que habia corrido al principio pero esta vez usando la deteccion estatica
###
def correr_find_template_threshold(objname, objnumber, scenename, scenenumber):
    # Parametros para el seguimiento
    find_template_comp_method = cv2.cv.CV_COMP_BHATTACHARYYA
    find_template_threshold = 0.6
    find_template_reverse = False

    find_frame_comp_method = cv2.cv.CV_COMP_BHATTACHARYYA
    find_frame_threshold = 0.4
    find_frame_reverse = False

    metodo_de_busqueda = BusquedaAlrededorCambiandoFrameSize()

    # Create objects
    img_provider = FrameNamesAndImageProviderPreChargedForRGB(
        'videos/rgbd/scenes/', scenename, scenenumber,
        'videos/rgbd/objs/', objname, objnumber,
    )  # path, objname, number

    for find_template_threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        detector = StaticDetectorForRGBFinder(
            matfile_path=('videos/rgbd/scenes/{sname}/{sname}_{snum}.mat'
                          .format(sname=scenename, snum=scenenumber)),
            obj_rgbd_name=objname,
        )

        template_comparator = HistogramComparator(
            method=find_template_comp_method,
            perc=find_template_threshold,
            worst_case=1,
            reverse=find_template_reverse,
        )
        frame_comparator = HistogramComparator(
            method=find_frame_comp_method,
            perc=find_frame_threshold,
            worst_case=1,
            reverse=find_frame_reverse,
        )

        finder = TemplateAndFrameHistogramFinder(
            template_comparator,
            frame_comparator,
            metodo_de_busqueda,
        )

        follower = RGBFollower(img_provider, detector, finder)

        FollowingSquemaExploringParameterRGB(
            img_provider,
            follower,
            'pruebas_guardadas',
            'RGB_staticdet_find_template_threshold',
            find_template_threshold,
        ).run()

        img_provider.restart()


def correr_find_frame_threshold(objname, objnumber, scenename, scenenumber):
    # Parametros para el seguimiento
    find_template_comp_method = cv2.cv.CV_COMP_BHATTACHARYYA
    find_template_threshold = 0.6
    find_template_reverse = False

    find_frame_comp_method = cv2.cv.CV_COMP_BHATTACHARYYA
    find_frame_threshold = 0.4
    find_frame_reverse = False

    metodo_de_busqueda = BusquedaAlrededorCambiandoFrameSize()

    # Create objects
    img_provider = FrameNamesAndImageProviderPreChargedForRGB(
        'videos/rgbd/scenes/', scenename, scenenumber,
        'videos/rgbd/objs/', objname, objnumber,
    )  # path, objname, number

    for find_frame_threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        detector = StaticDetectorForRGBFinder(
            matfile_path=('videos/rgbd/scenes/{sname}/{sname}_{snum}.mat'
                          .format(sname=scenename, snum=scenenumber)),
            obj_rgbd_name=objname,
        )

        template_comparator = HistogramComparator(
            method=find_template_comp_method,
            perc=find_template_threshold,
            worst_case=1,
            reverse=find_template_reverse,
        )
        frame_comparator = HistogramComparator(
            method=find_frame_comp_method,
            perc=find_frame_threshold,
            worst_case=1,
            reverse=find_frame_reverse,
        )

        finder = TemplateAndFrameHistogramFinder(
            template_comparator,
            frame_comparator,
            metodo_de_busqueda,
        )

        follower = RGBFollower(img_provider, detector, finder)

        FollowingSquemaExploringParameterRGB(
            img_provider,
            follower,
            'pruebas_guardadas',
            'RGB_staticdet_find_frame_threshold',
            find_frame_threshold,
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

    # seguir_taza_det_fija()
    # seguir_gorra_det_fija()

    # print "#################################################################"
    # print "correr_battachayyra_verde_find_template_threshold"
    # print "#################################################################"
    # correr_battachayyra_verde_find_template_threshold('coffee_mug', '5', 'desk', '1')
    # correr_battachayyra_verde_find_template_threshold('cap', '4', 'desk', '1')
    # correr_battachayyra_verde_find_template_threshold('bowl', '3', 'desk', '2')
    #
    # print "#################################################################"
    # print "correr_battachayyra_verde_find_frame_threshold"
    # print "#################################################################"
    # correr_battachayyra_verde_find_frame_threshold('coffee_mug', '5', 'desk', '1')
    # correr_battachayyra_verde_find_frame_threshold('cap', '4', 'desk', '1')
    # correr_battachayyra_verde_find_frame_threshold('bowl', '3', 'desk', '2')
    #
    # print "#################################################################"
    # print "correr_chi_squared_verde_find_template_threshold"
    # print "#################################################################"
    # correr_chi_squared_verde_find_template_threshold('coffee_mug', '5', 'desk', '1')
    # correr_chi_squared_verde_find_template_threshold('cap', '4', 'desk', '1')
    # correr_chi_squared_verde_find_template_threshold('bowl', '3', 'desk', '2')
    #
    # print "#################################################################"
    # print "correr_chi_squared_verde_find_frame_threshold"
    # print "#################################################################"
    # correr_chi_squared_verde_find_frame_threshold('coffee_mug', '5', 'desk', '1')
    # correr_chi_squared_verde_find_frame_threshold('cap', '4', 'desk', '1')
    # correr_chi_squared_verde_find_frame_threshold('bowl', '3', 'desk', '2')
    #
    # print "#################################################################"
    # print "correr_correlation_verde_find_template_threshold"
    # print "#################################################################"
    # correr_correlation_verde_find_template_threshold('coffee_mug', '5', 'desk', '1')
    # correr_correlation_verde_find_template_threshold('cap', '4', 'desk', '1')
    # correr_correlation_verde_find_template_threshold('bowl', '3', 'desk', '2')
    #
    # print "#################################################################"
    # print "correr_correlation_verde_find_frame_threshold"
    # print "#################################################################"
    # correr_correlation_verde_find_frame_threshold('coffee_mug', '5', 'desk', '1')
    # correr_correlation_verde_find_frame_threshold('cap', '4', 'desk', '1')
    # correr_correlation_verde_find_frame_threshold('bowl', '3', 'desk', '2')
    #
    # print "#################################################################"
    # print "correr_chi_squared_hs_find_template_threshold"
    # print "#################################################################"
    # correr_chi_squared_hs_find_template_threshold('coffee_mug', '5', 'desk', '1')
    # correr_chi_squared_hs_find_template_threshold('cap', '4', 'desk', '1')
    # correr_chi_squared_hs_find_template_threshold('bowl', '3', 'desk', '2')
    #
    # print "#################################################################"
    # print "correr_chi_squared_hs_find_frame_threshold"
    # print "#################################################################"
    # correr_chi_squared_hs_find_frame_threshold('coffee_mug', '5', 'desk', '1')
    # correr_chi_squared_hs_find_frame_threshold('cap', '4', 'desk', '1')
    # correr_chi_squared_hs_find_frame_threshold('bowl', '3', 'desk', '2')

    # print "#################################################################"
    # print "correr_mi_metodo_bhatta_bhatta_bhatta_template_perc"
    # print "#################################################################"
    # correr_mi_metodo_bhatta_bhatta_bhatta_template_perc('coffee_mug', '5', 'desk', '1')
    # correr_mi_metodo_bhatta_bhatta_bhatta_template_perc('cap', '4', 'desk', '1')
    # correr_mi_metodo_bhatta_bhatta_bhatta_template_perc('bowl', '3', 'desk', '2')
    #
    #
    # print "#################################################################"
    # print "correr_mi_metodo_bhatta_bhatta_bhatta_frame_perc"
    # print "#################################################################"
    # correr_mi_metodo_bhatta_bhatta_bhatta_frame_perc('coffee_mug', '5', 'desk', '1')
    # correr_mi_metodo_bhatta_bhatta_bhatta_frame_perc('cap', '4', 'desk', '1')
    # correr_mi_metodo_bhatta_bhatta_bhatta_frame_perc('bowl', '3', 'desk', '2')
    #
    # print "#################################################################"
    # print "correr_mi_metodo_chi_chi_bhatta_template_perc"
    # print "#################################################################"
    # correr_mi_metodo_chi_chi_bhatta_template_perc('coffee_mug', '5', 'desk', '1')
    # correr_mi_metodo_chi_chi_bhatta_template_perc('cap', '4', 'desk', '1')
    # correr_mi_metodo_chi_chi_bhatta_template_perc('bowl', '3', 'desk', '2')
    #
    # print "#################################################################"
    # print "correr_mi_metodo_chi_chi_bhatta_frame_perc"
    # print "#################################################################"
    # correr_mi_metodo_chi_chi_bhatta_frame_perc('coffee_mug', '5', 'desk', '1')
    # correr_mi_metodo_chi_chi_bhatta_frame_perc('cap', '4', 'desk', '1')
    # correr_mi_metodo_chi_chi_bhatta_frame_perc('bowl', '3', 'desk', '2')

    # print "#################################################################"
    # print "correr_mi_metodo_bhatta_inter_inter_template_perc"
    # print "#################################################################"
    # correr_mi_metodo_bhatta_inter_inter_template_perc('coffee_mug', '5', 'desk', '1')
    # correr_mi_metodo_bhatta_inter_inter_template_perc('cap', '4', 'desk', '1')
    # correr_mi_metodo_bhatta_inter_inter_template_perc('bowl', '3', 'desk', '2')
    #
    # print "#################################################################"
    # print "correr_mi_metodo_bhatta_inter_inter_frame_perc"
    # print "#################################################################"
    # correr_mi_metodo_bhatta_inter_inter_frame_perc('coffee_mug', '5', 'desk', '1')
    # correr_mi_metodo_bhatta_inter_inter_frame_perc('cap', '4', 'desk', '1')
    # correr_mi_metodo_bhatta_inter_inter_frame_perc('bowl', '3', 'desk', '2')


    print "#################################################################"
    print "correr_find_template_threshold"
    print "#################################################################"
    correr_find_template_threshold('coffee_mug', '5', 'desk', '1')
    correr_find_template_threshold('cap', '4', 'desk', '1')
    correr_find_template_threshold('bowl', '3', 'desk', '2')

    print "#################################################################"
    print "correr_find_frame_threshold"
    print "#################################################################"
    correr_find_frame_threshold('coffee_mug', '5', 'desk', '1')
    correr_find_frame_threshold('cap', '4', 'desk', '1')
    correr_find_frame_threshold('bowl', '3', 'desk', '2')
