#coding=utf-8

from __future__ import unicode_literals, division, print_function

import os
import codecs
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np

from detectores import StaticDetector
from analisis import Rectangle


def analizar_overlapping_por_parametro(matfile, scenenamenum, objname, objnum,
                                       param, path,
                                       include_detections_in_following=False):
    # Minimal overlapping area
    min_overlap_area = 0.3

    # Creo el objeto que provee los datos del ground truth
    ground_truth = StaticDetector(
        matfile,
        objname,
        objnum,
    )

    # Obtengo los valores del parametro para analizar
    objnamenum = '{name}_{num}'.format(name=objname, num=objnum)
    param_values = os.listdir(
        os.path.join(path, scenenamenum, objnamenum, param)
    )
    param_values = [pv for pv in param_values
                    if os.path.isdir(os.path.join(path, scenenamenum, objnamenum, param, pv))]
    param_values.sort()

    if len(param_values) == 0:
        param_values = ['']

    # Aqui guardare la informacion recolectada
    index = {}

    # Para cada valor del parametro voy a juntar la informacion de todas las
    # corridas
    for param_value in param_values:
        param_path = os.path.join(
            path,
            scenenamenum,
            objnamenum,
            param,
            param_value,
        )
        overlapping_areas = []
        frames = 0
        times_object_appear = 0
        times_object_detected = 0
        times_object_followed = 0

        fp = 0  # False positives
        fn = 0  # False negatives
        tp = 0  # True positives
        tn = 0  # True positives

        # Obtengo las direcciones de cada corrida y busco la informacion
        run_nums = os.listdir(param_path)
        run_nums = [rn for rn in run_nums if os.path.isdir(os.path.join(param_path, rn))]
        if len(run_nums) == 0:
            run_nums = ['']

        for run_num in run_nums:
            resultfile = os.path.join(param_path, run_num, 'results.txt')
            with codecs.open(resultfile, 'r', 'utf-8') as file_:

                # Salteo los valores de los parametros
                reach_result_zone = False
                while not reach_result_zone:
                    line = file_.next()
                    reach_result_zone = line.startswith('RESULTS_SECTION')

                # Junto los valores por linea/frame
                for line in file_:
                    # Cuento los frames
                    frames += 1

                    # Valores devueltos por el algoritmo
                    values = [int(v) for v in line.split(';')]
                    nframe = values[0]
                    fue_exitoso = values[1]
                    metodo = values[2]
                    fila_sup = values[3]
                    col_izq = values[4]
                    fila_inf = values[5]
                    col_der = values[6]

                    # Valores segun el ground truth
                    ground_truth.update({'nframe': nframe})
                    gt_fue_exitoso, gt_desc = ground_truth.detect()
                    gt_col_izq = gt_desc['topleft'][1]
                    gt_fila_sup = gt_desc['topleft'][0]
                    gt_col_der = gt_desc['bottomright'][1]
                    gt_fila_inf = gt_desc['bottomright'][0]

                    # Armo rectangulos con los resultados
                    rectangle_found = Rectangle(
                        (fila_sup, col_izq),
                        (fila_inf, col_der)
                    )
                    found_area = rectangle_found.area()
                    ground_truth_rectangle = Rectangle(
                        (gt_fila_sup, gt_col_izq),
                        (gt_fila_inf, gt_col_der)
                    )
                    ground_truth_area = ground_truth_rectangle.area()
                    intersection = rectangle_found.intersection(
                        ground_truth_rectangle
                    )
                    intersection_area = intersection.area()

                    # To be considered a correct detection, the area of
                    # overlap A0 between the predicted bounding box Bp and
                    # ground truth bounding box Bgt must exceed 50% by the
                    # formula:
                    # A0 = area(Bp intersection Bgt) / area(Bp union Bgt)
                    se_solaparon_poco = True
                    overlap_area = 0
                    if found_area > 0 and ground_truth_area > 0:
                        union_area = (
                            found_area + ground_truth_area - intersection_area
                        )
                        overlap_area = intersection_area / union_area
                        se_solaparon_poco = overlap_area < min_overlap_area

                    # Chequeo de falsos positivos y negativos y
                    # verdaderos negativos y positivos
                    estaba_y_no_se_encontro = gt_fue_exitoso and not fue_exitoso

                    no_estaba_y_se_encontro = (
                        ground_truth_area == 0 and found_area > 0
                    )

                    no_estaba_y_no_se_encontro = (
                        not (gt_fue_exitoso or fue_exitoso)
                    )

                    estaba_y_se_encontro = (
                        fue_exitoso and ground_truth_area > 0
                    )

                    if (estaba_y_no_se_encontro or
                            (estaba_y_se_encontro and se_solaparon_poco)):
                        fn += 1
                    elif no_estaba_y_se_encontro:
                        fp += 1
                    elif no_estaba_y_no_se_encontro:
                        tn += 1
                    elif estaba_y_se_encontro and not se_solaparon_poco:
                        tp += 1
                        if metodo == 0:
                            times_object_detected += 1
                        else:
                            times_object_followed += 1
                    else:
                        raise Exception('Algo anda mal. Revisar condiciones')

                    # El objeto aparece siempre que el ground truth asi lo
                    # indique, o si estando en un borde el algoritmo igual lo
                    # detecto
                    if gt_fue_exitoso or (found_area > 0 and
                                          ground_truth_area > 0):
                        times_object_appear += 1

                        # Incluyo el solapamiento si asi se necesita
                        if include_detections_in_following or metodo == 1:
                            overlapping_areas.append(overlap_area)

        if tp + tn + fp + fn != frames:
            raise Exception('No da bien el analisis de fp,fn,tp,tn')

        # Obtengo la media de los solapamientos
        mean = np.mean(overlapping_areas) if overlapping_areas else 0

        # Obtengo el desvío estandar de los solapamientos
        std = np.std(overlapping_areas) if overlapping_areas else 0

        # Almaceno estos valores
        index[param_value] = {
            'mean': mean,
            'std': std,
            'total_frames': frames,
            'times_object_appear': times_object_appear,
            'times_object_detected': times_object_detected,
            'times_object_followed': times_object_followed,
            'false_positives': fp,
            'false_negatives': fn,
            'true_positives': tp,
            'true_negatives': tn,
        }

    # Imprimo en pantalla para cada valor del parametro el promedio de
    # solapamiento en la escena para cada corrida y el promedio de todas las
    # corridas
    print('###############')
    print('Analizando {p} para el objeto {o} en la escena {e}'.format(
        p=param.upper(),
        o=objnamenum,
        e=scenenamenum,
    ))
    print('###############')
    just = 15
    print('Param. value'.rjust(just), end=' ')
    print('~Overlap'.rjust(just), end=' ')
    # print('Std. Dev. Overlap'.rjust(just), end=' ')
    print('# appear obj'.rjust(just), end=' ')
    print('% follow/appear'.rjust(just), end=' ')
    print('% FP'.rjust(just), end=' ')
    print('% FN'.rjust(just), end=' ')
    print('% TP'.rjust(just), end=' ')
    print('% TN'.rjust(just), end=' ')
    print('F-measure'.rjust(just), end=' ')
    print('Accuracy'.rjust(just))

    try:
        ordered_items = sorted(index.items(), key=lambda a: float(a[0]))
    except ValueError:
        ordered_items = sorted(index.items())

    for val, dd in ordered_items:
        avg = dd['mean']
        # std = dd['std']

        # Param value
        print('{a}'.format(a=val).rjust(just), end=' ')

        # Prom. overlap
        print(
            '{a}'.format(a=round(np.array(avg) * 100, 2)).rjust(just),
            end=' '
        )

        # Standard deviation
        # print(
            # '{a:{j}.2f}'.format(a=round(np.array(std) * 100, 2), j=just),
            # end=' '
        # )

        # Times obj. appeared
        print('{a:d}'.format(a=dd['times_object_appear']).rjust(just), end=' ')

        # % obj followed
        times_followed = dd['times_object_followed']
        if include_detections_in_following:
            times_followed += dd['times_object_detected']

        print(
            '{a}%'.format(
                a=round(times_followed / dd['times_object_appear'] * 100, 2),
            ).rjust(just),
            end=' ',
        )

        # % false positives
        print(
            '{a}%'.format(
                a=round(dd['false_positives'] / dd['total_frames'] * 100, 2),
            ).rjust(just),
            end=' ',
        )

        # % false negatives
        print(
            '{a}%'.format(
                a=round(dd['false_negatives'] / dd['total_frames'] * 100, 2),
            ).rjust(just),
            end=' ',
        )

        # % true positives
        print(
            '{a}%'.format(
                a=round(dd['true_positives'] / dd['total_frames'] * 100, 2),
            ).rjust(just),
            end=' ',
        )

        # % true negatives
        print(
            '{a}%'.format(
                a=round(dd['true_negatives'] / dd['total_frames'] * 100, 2),
            ).rjust(just),
            end=' ',
        )

        # F-measure
        precision = (
            dd['true_positives'] /
            (dd['true_positives'] + dd['false_positives'])
        )
        recall = (
            dd['true_positives'] /
            (dd['true_positives'] + dd['false_negatives'])
        )
        fmeasure = 2 * precision * recall / (precision + recall)
        print('{a}'.format(a=round(fmeasure, 2)).rjust(just), end=' ')

        # Accuracy
        tptn = dd['true_positives'] + dd['true_negatives']
        fpfn = dd['false_positives'] + dd['false_negatives']
        accuracy = (
            tptn / (tptn + fpfn)
        )
        print('{a}'.format(a=round(accuracy, 2)).rjust(just))


if __name__ == '__main__':
    # # Frame size
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='coffee_mug',
    #     objnum='5',
    #     param='detection_frame_size',
    #     path='pruebas_guardadas',
    # )
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='cap',
    #     objnum='4',
    #     param='detection_frame_size',
    #     path='pruebas_guardadas',
    # )
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_2.mat',
    #     scenenamenum='desk_2',
    #     objname='bowl',
    #     objnum='3',
    #     param='detection_frame_size',
    #     path='pruebas_guardadas',
    # )
    #
    # # Similarity threshold
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='coffee_mug',
    #     objnum='5',
    #     param='detection_similarity_threshold',
    #     path='pruebas_guardadas',
    # )
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='cap',
    #     objnum='4',
    #     param='detection_similarity_threshold',
    #     path='pruebas_guardadas',
    # )
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_2.mat',
    #     scenenamenum='desk_2',
    #     objname='bowl',
    #     objnum='3',
    #     param='detection_similarity_threshold',
    #     path='pruebas_guardadas',
    # )
    #
    # # Inlier fraction
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='coffee_mug',
    #     objnum='5',
    #     param='detection_inlier_fraction',
    #     path='pruebas_guardadas',
    # )
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='cap',
    #     objnum='4',
    #     param='detection_inlier_fraction',
    #     path='pruebas_guardadas',
    # )
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_2.mat',
    #     scenenamenum='desk_2',
    #     objname='bowl',
    #     objnum='3',
    #     param='detection_inlier_fraction',
    #     path='pruebas_guardadas',
    # )
    #
    # # Find percentage obj. model points
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='coffee_mug',
    #     objnum='5',
    #     param='find_perc_obj_model_points',
    #     path='pruebas_guardadas',
    # )
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='cap',
    #     objnum='4',
    #     param='find_perc_obj_model_points',
    #     path='pruebas_guardadas',
    # )
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_2.mat',
    #     scenenamenum='desk_2',
    #     objname='bowl',
    #     objnum='3',
    #     param='find_perc_obj_model_points',
    #     path='pruebas_guardadas',
    # )
    #
    # # Fixed search area
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='coffee_mug',
    #     objnum='5',
    #     param='find_fixed_search_area',
    #     path='pruebas_guardadas',
    # )
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='cap',
    #     objnum='4',
    #     param='find_fixed_search_area',
    #     path='pruebas_guardadas',
    # )
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_2.mat',
    #     scenenamenum='desk_2',
    #     objname='bowl',
    #     objnum='3',
    #     param='find_fixed_search_area',
    #     path='pruebas_guardadas',
    # )

    ##################
    # RGB analisis
    ##################
    # # Find frame threshold
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='coffee_mug',
    #     objnum='5',
    #     param='RGB_find_frame_threshold',
    #     path='pruebas_guardadas',
    # )
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='cap',
    #     objnum='4',
    #     param='RGB_find_frame_threshold',
    #     path='pruebas_guardadas',
    # )
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_2.mat',
    #     scenenamenum='desk_2',
    #     objname='bowl',
    #     objnum='3',
    #     param='RGB_find_frame_threshold',
    #     path='pruebas_guardadas',
    # )

    # # Find template threshold
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='coffee_mug',
    #     objnum='5',
    #     param='RGB_find_template_threshold',
    #     path='pruebas_guardadas',
    # )
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='cap',
    #     objnum='4',
    #     param='RGB_find_template_threshold',
    #     path='pruebas_guardadas',
    # )
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_2.mat',
    #     scenenamenum='desk_2',
    #     objname='bowl',
    #     objnum='3',
    #     param='RGB_find_template_threshold',
    #     path='pruebas_guardadas',
    # )
    #
    # # Detection template threshold
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='coffee_mug',
    #     objnum='5',
    #     param='RGB_det_template_threshold',
    #     path='pruebas_guardadas',
    # )
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='cap',
    #     objnum='4',
    #     param='RGB_det_template_threshold',
    #     path='pruebas_guardadas',
    # )
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_2.mat',
    #     scenenamenum='desk_2',
    #     objname='bowl',
    #     objnum='3',
    #     param='RGB_det_template_threshold',
    #     path='pruebas_guardadas',
    # )
    #
    # # Detection template sizes
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='coffee_mug',
    #     objnum='5',
    #     param='RGB_det_template_sizes',
    #     path='pruebas_guardadas',
    # )
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='cap',
    #     objnum='4',
    #     param='RGB_det_template_sizes',
    #     path='pruebas_guardadas',
    # )
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_2.mat',
    #     scenenamenum='desk_2',
    #     objname='bowl',
    #     objnum='3',
    #     param='RGB_det_template_sizes',
    #     path='pruebas_guardadas',
    # )

    ####################################
    # STATIC DETECTION and RGB analisis
    ####################################

    # # BATTA GREEN
    # # Find template threshold
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='coffee_mug',
    #     objnum='5',
    #     param='batta_green_channel_find_template_threshold',
    #     path='pruebas_guardadas',
    # )
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='cap',
    #     objnum='4',
    #     param='batta_green_channel_find_template_threshold',
    #     path='pruebas_guardadas',
    # )
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_2.mat',
    #     scenenamenum='desk_2',
    #     objname='bowl',
    #     objnum='3',
    #     param='batta_green_channel_find_template_threshold',
    #     path='pruebas_guardadas',
    # )
    # # Find frame threshold
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='coffee_mug',
    #     objnum='5',
    #     param='batta_green_channel_find_frame_threshold',
    #     path='pruebas_guardadas',
    # )
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='cap',
    #     objnum='4',
    #     param='batta_green_channel_find_frame_threshold',
    #     path='pruebas_guardadas',
    # )
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_2.mat',
    #     scenenamenum='desk_2',
    #     objname='bowl',
    #     objnum='3',
    #     param='batta_green_channel_find_frame_threshold',
    #     path='pruebas_guardadas',
    # )

    # # CHISQUARED GREEN
    # # Find template threshold
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='coffee_mug',
    #     objnum='5',
    #     param='chisquared_green_channel_find_template_threshold',
    #     path='pruebas_guardadas',
    # )
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='cap',
    #     objnum='4',
    #     param='chisquared_green_channel_find_template_threshold',
    #     path='pruebas_guardadas',
    # )
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_2.mat',
    #     scenenamenum='desk_2',
    #     objname='bowl',
    #     objnum='3',
    #     param='chisquared_green_channel_find_template_threshold',
    #     path='pruebas_guardadas',
    # )
    #
    # # Find frame threshold
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='coffee_mug',
    #     objnum='5',
    #     param='chisquared_green_channel_find_frame_threshold',
    #     path='pruebas_guardadas',
    # )
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='cap',
    #     objnum='4',
    #     param='chisquared_green_channel_find_frame_threshold',
    #     path='pruebas_guardadas',
    # )
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_2.mat',
    #     scenenamenum='desk_2',
    #     objname='bowl',
    #     objnum='3',
    #     param='chisquared_green_channel_find_frame_threshold',
    #     path='pruebas_guardadas',
    # )
    #
    # # CORRELATION GREEN
    # # Find template threshold
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='coffee_mug',
    #     objnum='5',
    #     param='correl_green_channel_find_template_threshold',
    #     path='pruebas_guardadas',
    # )
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='cap',
    #     objnum='4',
    #     param='correl_green_channel_find_template_threshold',
    #     path='pruebas_guardadas',
    # )
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_2.mat',
    #     scenenamenum='desk_2',
    #     objname='bowl',
    #     objnum='3',
    #     param='correl_green_channel_find_template_threshold',
    #     path='pruebas_guardadas',
    # )
    #
    # # Find frame threshold
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='coffee_mug',
    #     objnum='5',
    #     param='correl_green_channel_find_frame_threshold',
    #     path='pruebas_guardadas',
    # )
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='cap',
    #     objnum='4',
    #     param='correl_green_channel_find_frame_threshold',
    #     path='pruebas_guardadas',
    # )
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_2.mat',
    #     scenenamenum='desk_2',
    #     objname='bowl',
    #     objnum='3',
    #     param='correl_green_channel_find_frame_threshold',
    #     path='pruebas_guardadas',
    # )

    # # CHISQUARED HS
    # # Find template threshold
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='coffee_mug',
    #     objnum='5',
    #     param='chisquared_hs_channels_find_template_threshold_fixed',
    #     path='pruebas_guardadas',
    # )
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='cap',
    #     objnum='4',
    #     param='chisquared_hs_channels_find_template_threshold_fixed',
    #     path='pruebas_guardadas',
    # )
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_2.mat',
    #     scenenamenum='desk_2',
    #     objname='bowl',
    #     objnum='3',
    #     param='chisquared_hs_channels_find_template_threshold_fixed',
    #     path='pruebas_guardadas',
    # )
    #
    # # Find frame threshold
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='coffee_mug',
    #     objnum='5',
    #     param='chisquared_hs_channels_find_frame_threshold_fixed',
    #     path='pruebas_guardadas',
    # )
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='cap',
    #     objnum='4',
    #     param='chisquared_hs_channels_find_frame_threshold_fixed',
    #     path='pruebas_guardadas',
    # )
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_2.mat',
    #     scenenamenum='desk_2',
    #     objname='bowl',
    #     objnum='3',
    #     param='chisquared_hs_channels_find_frame_threshold_fixed',
    #     path='pruebas_guardadas',
    # )

    # # MI METODO BHATTA BHATTA BHATTA
    # # Find template threshold
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='coffee_mug',
    #     objnum='5',
    #     param='mi_metodo_bhatta_bhatta_bhatta_template_perc',
    #     path='pruebas_guardadas',
    # )
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='cap',
    #     objnum='4',
    #     param='mi_metodo_bhatta_bhatta_bhatta_template_perc',
    #     path='pruebas_guardadas',
    # )
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_2.mat',
    #     scenenamenum='desk_2',
    #     objname='bowl',
    #     objnum='3',
    #     param='mi_metodo_bhatta_bhatta_bhatta_template_perc',
    #     path='pruebas_guardadas',
    # )
    #
    # # Find frame threshold
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='coffee_mug',
    #     objnum='5',
    #     param='mi_metodo_bhatta_bhatta_bhatta_frame_perc',
    #     path='pruebas_guardadas',
    # )
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='cap',
    #     objnum='4',
    #     param='mi_metodo_bhatta_bhatta_bhatta_frame_perc',
    #     path='pruebas_guardadas',
    # )
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_2.mat',
    #     scenenamenum='desk_2',
    #     objname='bowl',
    #     objnum='3',
    #     param='mi_metodo_bhatta_bhatta_bhatta_frame_perc',
    #     path='pruebas_guardadas',
    # )
    #
    # # MI METODO CHI CHI BHATTA
    # # Find template threshold
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='coffee_mug',
    #     objnum='5',
    #     param='mi_metodo_chi_chi_bhatta_template_perc',
    #     path='pruebas_guardadas',
    # )
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='cap',
    #     objnum='4',
    #     param='mi_metodo_chi_chi_bhatta_template_perc',
    #     path='pruebas_guardadas',
    # )
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_2.mat',
    #     scenenamenum='desk_2',
    #     objname='bowl',
    #     objnum='3',
    #     param='mi_metodo_chi_chi_bhatta_template_perc',
    #     path='pruebas_guardadas',
    # )
    #
    # # Find frame threshold
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='coffee_mug',
    #     objnum='5',
    #     param='mi_metodo_chi_chi_bhatta_frame_perc',
    #     path='pruebas_guardadas',
    # )
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='cap',
    #     objnum='4',
    #     param='mi_metodo_chi_chi_bhatta_frame_perc',
    #     path='pruebas_guardadas',
    # )
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_2.mat',
    #     scenenamenum='desk_2',
    #     objname='bowl',
    #     objnum='3',
    #     param='mi_metodo_chi_chi_bhatta_frame_perc',
    #     path='pruebas_guardadas',
    # )

    # # MI METODO CHI CHI BHATTA
    # # Find template threshold
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='coffee_mug',
    #     objnum='5',
    #     param='mi_metodo_bhatta_inter_inter_template_perc',
    #     path='pruebas_guardadas',
    # )
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='cap',
    #     objnum='4',
    #     param='mi_metodo_bhatta_inter_inter_template_perc',
    #     path='pruebas_guardadas',
    # )
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_2.mat',
    #     scenenamenum='desk_2',
    #     objname='bowl',
    #     objnum='3',
    #     param='mi_metodo_bhatta_inter_inter_template_perc',
    #     path='pruebas_guardadas',
    # )
    #
    # # Find frame threshold
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='coffee_mug',
    #     objnum='5',
    #     param='mi_metodo_bhatta_inter_inter_frame_perc',
    #     path='pruebas_guardadas',
    # )
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='cap',
    #     objnum='4',
    #     param='mi_metodo_bhatta_inter_inter_frame_perc',
    #     path='pruebas_guardadas',
    # )
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_2.mat',
    #     scenenamenum='desk_2',
    #     objname='bowl',
    #     objnum='3',
    #     param='mi_metodo_bhatta_inter_inter_frame_perc',
    #     path='pruebas_guardadas',
    # )

    # RGB y HSV
    # Find template threshold
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='coffee_mug',
    #     objnum='5',
    #     param='RGB_staticdet_find_template_threshold',
    #     path='pruebas_guardadas',
    # )
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='cap',
    #     objnum='4',
    #     param='RGB_staticdet_find_template_threshold',
    #     path='pruebas_guardadas',
    # )
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_2.mat',
    #     scenenamenum='desk_2',
    #     objname='bowl',
    #     objnum='3',
    #     param='RGB_staticdet_find_template_threshold',
    #     path='pruebas_guardadas',
    # )
    #
    # # Find frame threshold
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='coffee_mug',
    #     objnum='5',
    #     param='RGB_staticdet_find_frame_threshold',
    #     path='pruebas_guardadas',
    # )
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='cap',
    #     objnum='4',
    #     param='RGB_staticdet_find_frame_threshold',
    #     path='pruebas_guardadas',
    # )
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_2.mat',
    #     scenenamenum='desk_2',
    #     objname='bowl',
    #     objnum='3',
    #     param='RGB_staticdet_find_frame_threshold',
    #     path='pruebas_guardadas',
    # )

    ##################
    # DEFINITIVOS RGB
    ##################
    # Bhatta verde
    analizar_overlapping_por_parametro(
        matfile='videos/rgbd/scenes/desk/desk_1.mat',
        scenenamenum='desk_1',
        objname='coffee_mug',
        objnum='5',
        param='definitivo_batta_green_channel',
        path='pruebas_guardadas',
    )
    analizar_overlapping_por_parametro(
        matfile='videos/rgbd/scenes/desk/desk_1.mat',
        scenenamenum='desk_1',
        objname='cap',
        objnum='4',
        param='definitivo_batta_green_channel',
        path='pruebas_guardadas',
    )
    analizar_overlapping_por_parametro(
        matfile='videos/rgbd/scenes/desk/desk_2.mat',
        scenenamenum='desk_2',
        objname='bowl',
        objnum='3',
        param='definitivo_batta_green_channel',
        path='pruebas_guardadas',
    )

    # Correlation verde
    analizar_overlapping_por_parametro(
        matfile='videos/rgbd/scenes/desk/desk_1.mat',
        scenenamenum='desk_1',
        objname='coffee_mug',
        objnum='5',
        param='definitivo_correl_green_channel',
        path='pruebas_guardadas',
    )
    analizar_overlapping_por_parametro(
        matfile='videos/rgbd/scenes/desk/desk_1.mat',
        scenenamenum='desk_1',
        objname='cap',
        objnum='4',
        param='definitivo_correl_green_channel',
        path='pruebas_guardadas',
    )
    analizar_overlapping_por_parametro(
        matfile='videos/rgbd/scenes/desk/desk_2.mat',
        scenenamenum='desk_2',
        objname='bowl',
        objnum='3',
        param='definitivo_correl_green_channel',
        path='pruebas_guardadas',
    )
    #
    # # Mi metodo tripe bhatachayyra
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='coffee_mug',
    #     objnum='5',
    #     param='definitivo_mi_metodo_bhatta_bhatta_bhatta',
    #     path='pruebas_guardadas',
    # )
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='cap',
    #     objnum='4',
    #     param='definitivo_mi_metodo_bhatta_bhatta_bhatta',
    #     path='pruebas_guardadas',
    # )
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_2.mat',
    #     scenenamenum='desk_2',
    #     objname='bowl',
    #     objnum='3',
    #     param='definitivo_mi_metodo_bhatta_bhatta_bhatta',
    #     path='pruebas_guardadas',
    # )

    #####################################
    # STATIC DETECTION y seguimiento ICP
    ######################################
    # #  Euclidean fitness
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='coffee_mug',
    #     objnum='5',
    #     param='DEPTH_find_euclidean_fitness',
    #     path='pruebas_guardadas',
    # )
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='cap',
    #     objnum='4',
    #     param='DEPTH_find_euclidean_fitness',
    #     path='pruebas_guardadas',
    # )
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_2.mat',
    #     scenenamenum='desk_2',
    #     objname='bowl',
    #     objnum='3',
    #     param='DEPTH_find_euclidean_fitness',
    #     path='pruebas_guardadas',
    # )
    #
    # # Transformation epsilon
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='coffee_mug',
    #     objnum='5',
    #     param='DEPTH_find_transformation_epsilon',
    #     path='pruebas_guardadas',
    # )
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='cap',
    #     objnum='4',
    #     param='DEPTH_find_transformation_epsilon',
    #     path='pruebas_guardadas',
    # )
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_2.mat',
    #     scenenamenum='desk_2',
    #     objname='bowl',
    #     objnum='3',
    #     param='DEPTH_find_transformation_epsilon',
    #     path='pruebas_guardadas',
    # )
    #
    # # Correspondence distance
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='coffee_mug',
    #     objnum='5',
    #     param='DEPTH_find_correspondence_distance',
    #     path='pruebas_guardadas',
    # )
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='cap',
    #     objnum='4',
    #     param='DEPTH_find_correspondence_distance',
    #     path='pruebas_guardadas',
    # )
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_2.mat',
    #     scenenamenum='desk_2',
    #     objname='bowl',
    #     objnum='3',
    #     param='DEPTH_find_correspondence_distance',
    #     path='pruebas_guardadas',
    # )
    #
    # # Percentage obj. model points
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='coffee_mug',
    #     objnum='5',
    #     param='DEPTH_find_perc_obj_model_points',
    #     path='pruebas_guardadas',
    # )
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='cap',
    #     objnum='4',
    #     param='DEPTH_find_perc_obj_model_points',
    #     path='pruebas_guardadas',
    # )
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_2.mat',
    #     scenenamenum='desk_2',
    #     objname='bowl',
    #     objnum='3',
    #     param='DEPTH_find_perc_obj_model_points',
    #     path='pruebas_guardadas',
    # )


    # Umbral score
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='coffee_mug',
    #     objnum='5',
    #     param='DEPTH_find_umbral_score',
    #     path='pruebas_guardadas',
    # )
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='cap',
    #     objnum='4',
    #     param='DEPTH_find_umbral_score',
    #     path='pruebas_guardadas',
    # )
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_2.mat',
    #     scenenamenum='desk_2',
    #     objname='bowl',
    #     objnum='3',
    #     param='DEPTH_find_umbral_score',
    #     path='pruebas_guardadas',
    # )
    #
    #
    # # Detection max iterations
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='coffee_mug',
    #     objnum='5',
    #     param='DEPTH_det_max_iter',
    #     path='pruebas_guardadas',
    # )
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='cap',
    #     objnum='4',
    #     param='DEPTH_det_max_iter',
    #     path='pruebas_guardadas',
    # )
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_2.mat',
    #     scenenamenum='desk_2',
    #     objname='bowl',
    #     objnum='3',
    #     param='DEPTH_det_max_iter',
    #     path='pruebas_guardadas',
    # )
    #
    #
    # # Detection points to sample
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='coffee_mug',
    #     objnum='5',
    #     param='DEPTH_det_points_to_sample',
    #     path='pruebas_guardadas',
    # )
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='cap',
    #     objnum='4',
    #     param='DEPTH_det_points_to_sample',
    #     path='pruebas_guardadas',
    # )
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_2.mat',
    #     scenenamenum='desk_2',
    #     objname='bowl',
    #     objnum='3',
    #     param='DEPTH_det_points_to_sample',
    #     path='pruebas_guardadas',
    # )
    #
    #
    # # Detection nearest features
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='coffee_mug',
    #     objnum='5',
    #     param='DEPTH_det_nearest_features',
    #     path='pruebas_guardadas',
    # )
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='cap',
    #     objnum='4',
    #     param='DEPTH_det_nearest_features',
    #     path='pruebas_guardadas',
    # )
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_2.mat',
    #     scenenamenum='desk_2',
    #     objname='bowl',
    #     objnum='3',
    #     param='DEPTH_det_nearest_features',
    #     path='pruebas_guardadas',
    # )
    #
    #
    # # Detection similarity threshold
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='coffee_mug',
    #     objnum='5',
    #     param='DEPTH_det_simil_thresh',
    #     path='pruebas_guardadas',
    # )
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='cap',
    #     objnum='4',
    #     param='DEPTH_det_simil_thresh',
    #     path='pruebas_guardadas',
    # )
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_2.mat',
    #     scenenamenum='desk_2',
    #     objname='bowl',
    #     objnum='3',
    #     param='DEPTH_det_simil_thresh',
    #     path='pruebas_guardadas',
    # )
    #
    #
    # # detection inlier threshold
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='coffee_mug',
    #     objnum='5',
    #     param='DEPTH_det_inlier_thresh',
    #     path='pruebas_guardadas',
    # )
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='cap',
    #     objnum='4',
    #     param='DEPTH_det_inlier_thresh',
    #     path='pruebas_guardadas',
    # )
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_2.mat',
    #     scenenamenum='desk_2',
    #     objname='bowl',
    #     objnum='3',
    #     param='DEPTH_det_inlier_thresh',
    #     path='pruebas_guardadas',
    # )
    #
    #
    # # Detection inlier fraction
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='coffee_mug',
    #     objnum='5',
    #     param='DEPTH_det_inlier_fraction',
    #     path='pruebas_guardadas',
    # )
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='cap',
    #     objnum='4',
    #     param='DEPTH_det_inlier_fraction',
    #     path='pruebas_guardadas',
    # )
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_2.mat',
    #     scenenamenum='desk_2',
    #     objname='bowl',
    #     objnum='3',
    #     param='DEPTH_det_inlier_fraction',
    #     path='pruebas_guardadas',
    # )

    ###########################################################################
    #                           INICIO DEFINITIVOS                            #
    ###########################################################################
    ##########################################
    # STATIC DETECTION y definitivo RGB y HSV
    ##########################################
    analizar_overlapping_por_parametro(
        matfile='videos/rgbd/scenes/desk/desk_1.mat',
        scenenamenum='desk_1',
        objname='coffee_mug',
        objnum='5',
        param='definitivo_RGB_staticdet',
        path='pruebas_guardadas',
    )
    analizar_overlapping_por_parametro(
        matfile='videos/rgbd/scenes/desk/desk_1.mat',
        scenenamenum='desk_1',
        objname='cap',
        objnum='4',
        param='definitivo_RGB_staticdet',
        path='pruebas_guardadas',
    )
    analizar_overlapping_por_parametro(
        matfile='videos/rgbd/scenes/desk/desk_2.mat',
        scenenamenum='desk_2',
        objname='bowl',
        objnum='3',
        param='definitivo_RGB_staticdet',
        path='pruebas_guardadas',
    )
    analizar_overlapping_por_parametro(
        matfile='videos/rgbd/scenes/table/table_1.mat',
        scenenamenum='table_1',
        objname='coffee_mug',
        objnum='1',
        param='definitivo_RGB_staticdet',
        path='pruebas_guardadas',
    )
    analizar_overlapping_por_parametro(
        matfile='videos/rgbd/scenes/table/table_1.mat',
        scenenamenum='table_1',
        objname='soda_can',
        objnum='4',
        param='definitivo_RGB_staticdet',
        path='pruebas_guardadas',
    )
    analizar_overlapping_por_parametro(
        matfile='videos/rgbd/scenes/table_small/table_small_2.mat',
        scenenamenum='table_small_2',
        objname='cereal_box',
        objnum='4',
        param='definitivo_RGB_staticdet',
        path='pruebas_guardadas',
    )

    ##########################################################
    # STATIC DETECTION y definitivo DEPTH
    ##########################################################
    analizar_overlapping_por_parametro(
        matfile='videos/rgbd/scenes/desk/desk_1.mat',
        scenenamenum='desk_1',
        objname='coffee_mug',
        objnum='5',
        param='definitivo_DEPTH',
        path='pruebas_guardadas',
    )
    analizar_overlapping_por_parametro(
        matfile='videos/rgbd/scenes/desk/desk_1.mat',
        scenenamenum='desk_1',
        objname='cap',
        objnum='4',
        param='definitivo_DEPTH',
        path='pruebas_guardadas',
    )
    analizar_overlapping_por_parametro(
        matfile='videos/rgbd/scenes/desk/desk_2.mat',
        scenenamenum='desk_2',
        objname='bowl',
        objnum='3',
        param='definitivo_DEPTH',
        path='pruebas_guardadas',
    )
    analizar_overlapping_por_parametro(
        matfile='videos/rgbd/scenes/table/table_1.mat',
        scenenamenum='table_1',
        objname='coffee_mug',
        objnum='1',
        param='definitivo_DEPTH',
        path='pruebas_guardadas',
    )
    analizar_overlapping_por_parametro(
        matfile='videos/rgbd/scenes/table/table_1.mat',
        scenenamenum='table_1',
        objname='soda_can',
        objnum='4',
        param='definitivo_DEPTH',
        path='pruebas_guardadas',
    )
    analizar_overlapping_por_parametro(
        matfile='videos/rgbd/scenes/table_small/table_small_2.mat',
        scenenamenum='table_small_2',
        objname='cereal_box',
        objnum='4',
        param='definitivo_DEPTH',
        path='pruebas_guardadas',
    )

    ##################################################################
    # STATIC DETECTION y seguimiento RGB-D, preferentemente D
    ##################################################################
    analizar_overlapping_por_parametro(
        matfile='videos/rgbd/scenes/desk/desk_1.mat',
        scenenamenum='desk_1',
        objname='coffee_mug',
        objnum='5',
        param='definitivo_RGBD_preferD',
        path='pruebas_guardadas',
    )

    analizar_overlapping_por_parametro(
        matfile='videos/rgbd/scenes/desk/desk_1.mat',
        scenenamenum='desk_1',
        objname='cap',
        objnum='4',
        param='definitivo_RGBD_preferD',
        path='pruebas_guardadas',
    )

    analizar_overlapping_por_parametro(
        matfile='videos/rgbd/scenes/desk/desk_2.mat',
        scenenamenum='desk_2',
        objname='bowl',
        objnum='3',
        param='definitivo_RGBD_preferD',
        path='pruebas_guardadas',
    )
    analizar_overlapping_por_parametro(
        matfile='videos/rgbd/scenes/table/table_1.mat',
        scenenamenum='table_1',
        objname='coffee_mug',
        objnum='1',
        param='definitivo_RGBD_preferD',
        path='pruebas_guardadas',
    )
    analizar_overlapping_por_parametro(
        matfile='videos/rgbd/scenes/table/table_1.mat',
        scenenamenum='table_1',
        objname='soda_can',
        objnum='4',
        param='definitivo_RGBD_preferD',
        path='pruebas_guardadas',
    )
    analizar_overlapping_por_parametro(
        matfile='videos/rgbd/scenes/table_small/table_small_2.mat',
        scenenamenum='table_small_2',
        objname='cereal_box',
        objnum='4',
        param='definitivo_RGBD_preferD',
        path='pruebas_guardadas',
    )

    ##################################################################
    # STATIC DETECTION y seguimiento RGB-D, preferentemente RGB
    ##################################################################
    analizar_overlapping_por_parametro(
        matfile='videos/rgbd/scenes/desk/desk_1.mat',
        scenenamenum='desk_1',
        objname='coffee_mug',
        objnum='5',
        param='definitivo_RGBD_preferRGB',
        path='pruebas_guardadas',
    )

    analizar_overlapping_por_parametro(
        matfile='videos/rgbd/scenes/desk/desk_1.mat',
        scenenamenum='desk_1',
        objname='cap',
        objnum='4',
        param='definitivo_RGBD_preferRGB',
        path='pruebas_guardadas',
    )

    analizar_overlapping_por_parametro(
        matfile='videos/rgbd/scenes/desk/desk_2.mat',
        scenenamenum='desk_2',
        objname='bowl',
        objnum='3',
        param='definitivo_RGBD_preferRGB',
        path='pruebas_guardadas',
    )
    analizar_overlapping_por_parametro(
        matfile='videos/rgbd/scenes/table/table_1.mat',
        scenenamenum='table_1',
        objname='coffee_mug',
        objnum='1',
        param='definitivo_RGBD_preferRGB',
        path='pruebas_guardadas',
    )
    analizar_overlapping_por_parametro(
        matfile='videos/rgbd/scenes/table/table_1.mat',
        scenenamenum='table_1',
        objname='soda_can',
        objnum='4',
        param='definitivo_RGBD_preferRGB',
        path='pruebas_guardadas',
    )
    analizar_overlapping_por_parametro(
        matfile='videos/rgbd/scenes/table_small/table_small_2.mat',
        scenenamenum='table_small_2',
        objname='cereal_box',
        objnum='4',
        param='definitivo_RGBD_preferRGB',
        path='pruebas_guardadas',
    )

    ##################################################################
    # Definitivo sistema RGBD priorizando D en seguimiento
    ##################################################################
    analizar_overlapping_por_parametro(
        matfile='videos/rgbd/scenes/desk/desk_1.mat',
        scenenamenum='desk_1',
        objname='coffee_mug',
        objnum='5',
        param='definitivo_automatico_RGBD',
        path='pruebas_guardadas',
        include_detections_in_following=True,
    )

    analizar_overlapping_por_parametro(
        matfile='videos/rgbd/scenes/desk/desk_1.mat',
        scenenamenum='desk_1',
        objname='cap',
        objnum='4',
        param='definitivo_automatico_RGBD',
        path='pruebas_guardadas',
        include_detections_in_following=True,
    )

    analizar_overlapping_por_parametro(
        matfile='videos/rgbd/scenes/desk/desk_2.mat',
        scenenamenum='desk_2',
        objname='bowl',
        objnum='3',
        param='definitivo_automatico_RGBD',
        path='pruebas_guardadas',
        include_detections_in_following=True,
    )
    analizar_overlapping_por_parametro(
        matfile='videos/rgbd/scenes/table/table_1.mat',
        scenenamenum='table_1',
        objname='coffee_mug',
        objnum='1',
        param='definitivo_automatico_RGBD',
        path='pruebas_guardadas',
        include_detections_in_following=True,
    )
    analizar_overlapping_por_parametro(
        matfile='videos/rgbd/scenes/table/table_1.mat',
        scenenamenum='table_1',
        objname='soda_can',
        objnum='4',
        param='definitivo_automatico_RGBD',
        path='pruebas_guardadas',
        include_detections_in_following=True,
    )
    analizar_overlapping_por_parametro(
        matfile='videos/rgbd/scenes/table_small/table_small_2.mat',
        scenenamenum='table_small_2',
        objname='cereal_box',
        objnum='4',
        param='definitivo_automatico_RGBD',
        path='pruebas_guardadas',
        include_detections_in_following=True,
    )

    ##################################################################
    # Definitivo sistema RGBD priorizando RGB en seguimiento
    ##################################################################
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='coffee_mug',
    #     objnum='5',
    #     param='definitivo_automatico_RGB_RGBD',
    #     path='pruebas_guardadas',
    #     include_detections_in_following=True,
    # )
    #
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='cap',
    #     objnum='4',
    #     param='definitivo_automatico_RGB_RGBD',
    #     path='pruebas_guardadas',
    #     include_detections_in_following=True,
    # )
    #
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_2.mat',
    #     scenenamenum='desk_2',
    #     objname='bowl',
    #     objnum='3',
    #     param='definitivo_automatico_RGB_RGBD',
    #     path='pruebas_guardadas',
    #     include_detections_in_following=True,
    # )
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/table/table_1.mat',
    #     scenenamenum='table_1',
    #     objname='coffee_mug',
    #     objnum='1',
    #     param='definitivo_automatico_RGB_RGBD',
    #     path='pruebas_guardadas',
    #     include_detections_in_following=True,
    # )
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/table/table_1.mat',
    #     scenenamenum='table_1',
    #     objname='soda_can',
    #     objnum='4',
    #     param='definitivo_automatico_RGB_RGBD',
    #     path='pruebas_guardadas',
    #     include_detections_in_following=True,
    # )
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/table_small/table_small_2.mat',
    #     scenenamenum='table_small_2',
    #     objname='cereal_box',
    #     objnum='4',
    #     param='definitivo_automatico_RGB_RGBD',
    #     path='pruebas_guardadas',
    #     include_detections_in_following=True,
    # )

    ############################################################################
    #                          FIN DEFINITIVOS                                 #
    ############################################################################


    # Prueba colgada del RGB-HSV
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='coffee_mug',
    #     objnum='5',
    #     param='probando_RGB_solo_frame',
    #     path='pruebas_guardadas',
    # )
    #
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='cap',
    #     objnum='4',
    #     param='probando_RGB_solo_frame',
    #     path='pruebas_guardadas',
    # )
    #
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_2.mat',
    #     scenenamenum='desk_2',
    #     objname='bowl',
    #     objnum='3',
    #     param='probando_RGB_solo_frame',
    #     path='pruebas_guardadas',
    # )

    # # Prueba colgada del DEPTH
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='coffee_mug',
    #     objnum='5',
    #     param='probando_deteccion_estatica_DEPTH',
    #     path='pruebas_guardadas',
    # )

    # # Pruebas deteccion por templates
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='coffee_mug',
    #     objnum='5',
    #     param='deteccion_template_threshold',
    #     path='pruebas_guardadas',
    #     include_detections_in_following=True,
    # )
    #
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='cap',
    #     objnum='4',
    #     param='deteccion_template_threshold',
    #     path='pruebas_guardadas',
    #     include_detections_in_following=True,
    # )
    #
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_2.mat',
    #     scenenamenum='desk_2',
    #     objname='bowl',
    #     objnum='3',
    #     param='deteccion_template_threshold',
    #     path='pruebas_guardadas',
    #     include_detections_in_following=True,
    # )
    #
    # # Pruebas deteccion RGBD
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='coffee_mug',
    #     objnum='5',
    #     param='probando_deteccion_automatica_RGBD',
    #     path='pruebas_guardadas',
    #     include_detections_in_following=True,
    # )
    #
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='cap',
    #     objnum='4',
    #     param='probando_deteccion_automatica_RGBD',
    #     path='pruebas_guardadas',
    #     include_detections_in_following=True,
    # )
    #
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_2.mat',
    #     scenenamenum='desk_2',
    #     objname='bowl',
    #     objnum='3',
    #     param='probando_deteccion_automatica_RGBD',
    #     path='pruebas_guardadas',
    #     include_detections_in_following=True,
    # )
