#coding=utf-8

from __future__ import unicode_literals, division, print_function

import os
import codecs
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np

from detectores import StaticDetector
from analisis import Rectangle


def analizar_overlapping_por_parametro(matfile, scenenamenum, objname, objnum,
                                       param, path):
    ground_truth = StaticDetector(
        matfile,
        objname,
    )
    objnamenum = '{name}_{num}'.format(name=objname, num=objnum)
    param_values = os.listdir(
        os.path.join(path, scenenamenum, objnamenum, param)
    )
    param_values = [pv for pv in param_values
                    if os.path.isdir(os.path.join(path, scenenamenum, objnamenum, param, pv))]
    param_values.sort()

    if len(param_values) == 0:
        param_values = ['']

    index = {}

    # Para cada valor del parametro
    for param_value in param_values:
        param_path = os.path.join(
            path,
            scenenamenum,
            objnamenum,
            param,
            param_value,
        )
        overlapping_areas = []
        times_object_appear = 0
        times_object_detected = 0
        times_object_followed = 0

        # Para cada corrida
        run_nums = os.listdir(param_path)
        run_nums = [rn for rn in run_nums if os.path.isdir(os.path.join(param_path, rn))]
        if len(run_nums) == 0:
            run_nums = ['']

        for run_num in run_nums:
            resultfile = os.path.join(param_path, run_num, 'results.txt')
            with codecs.open(resultfile, 'r', 'utf-8') as file_:
                reach_result_zone = False
                while not reach_result_zone:
                    line = file_.next()
                    reach_result_zone = line.startswith('RESULTS_SECTION')

                for line in file_:
                    values = [int(v) for v in line.split(';')]

                    nframe = values[0]
                    # fue_exitoso = values[1]
                    metodo = values[2]
                    fila_sup = values[3]
                    col_izq = values[4]
                    fila_inf = values[5]
                    col_der = values[6]

                    ground_truth.update({'nframe': nframe})
                    gt_fue_exitoso, gt_desc = ground_truth.detect()
                    gt_col_izq = gt_desc['topleft'][1]
                    gt_fila_sup = gt_desc['topleft'][0]
                    gt_col_der = gt_desc['bottomright'][1]
                    gt_fila_inf = gt_desc['bottomright'][0]

                    rectangle_found = Rectangle(
                        (fila_sup, col_izq),
                        (fila_inf, col_der)
                    )
                    ground_truth_rectangle = Rectangle(
                        (gt_fila_sup, gt_col_izq),
                        (gt_fila_inf, gt_col_der)
                    )
                    intersection = rectangle_found.intersection(
                        ground_truth_rectangle
                    )

                    # Si el ground truth dice que hay algo, comparamos el
                    # resultado
                    if ground_truth_rectangle.area() > 0:
                        times_object_appear += 1
                        found_area = rectangle_found.area()

                        if found_area > 0 and metodo == 0:
                            times_object_detected += 1
                        elif found_area > 0 and metodo == 1:
                            times_object_followed += 1

                        ground_truth_area = ground_truth_rectangle.area()
                        intersection_area = intersection.area()

                        # To be considered a correct detection, the area of
                        # overlap A0 between the predicted bounding box Bp and
                        # ground truth bounding box Bgt must exceed 50% by the
                        # formula:
                        # A0 = area(Bp intersection Bgt) / area(Bp union Bgt)
                        union_area = (
                            found_area + ground_truth_area - intersection_area
                        )
                        overlap_area = intersection_area / union_area

                        if overlap_area < 0:
                            pass

                        overlapping_areas.append(overlap_area)

        mean = np.mean(overlapping_areas) if overlapping_areas else 0
        std = np.std(overlapping_areas) if overlapping_areas else 0
        index[param_value] = {
            'mean': mean,
            'std': std,
            'times_object_appear': times_object_appear,
            'times_object_detected': times_object_detected,
            'times_object_followed': times_object_followed,
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
    just = 18
    print('Param. value'.rjust(just), end=' ')
    print('Prom. Overlap'.rjust(just), end=' ')
    print('Std. Dev. Overlap'.rjust(just), end=' ')
    print('Cant. appear obj'.rjust(just), end=' ')
    print('Cant. found obj'.rjust(just), end=' ')
    print('Cant. follow obj'.rjust(just), end=' ')
    print('% follow/appear'.rjust(just))

    try:
        ordered_items = sorted(index.items(), key=lambda a: float(a[0]))
    except ValueError:
        ordered_items = sorted(index.items())

    for val, dd in ordered_items:
        avg = dd['mean']
        std = dd['std']

        print('{a}'.format(a=val).rjust(just), end=' ')
        # print('{v}:'.format(v=val))
        print('{a:{j}.2f}'.format(a=round(np.array(avg) * 100, 2), j=just), end=' ')
        # print('    prom_overlap: {p}'.format(
        #     p=round(np.array(avg) * 100, 2)
        # ))
        print('{a:{j}.2f}'.format(a=round(np.array(std) * 100, 2), j=just), end=' ')
        # print('    prom_overlap_std: {p}'.format(
        #     p=round(np.array(std) * 100, 2),
        # ))
        print('{a:{j}d}'.format(a=dd['times_object_appear'], j=just), end=' ')
        # print('    veces aparece el obj: {v}'.format(
        #     v=dd['times_object_appear'])
        # )
        print('{a:{j}d}'.format(a=dd['times_object_detected'], j=just), end=' ')
        # print('    veces encontrado el obj: {v}'.format(
        #     v=dd['times_object_detected'])
        # )
        print('{a:{j}d}'.format(a=dd['times_object_followed'], j=just), end=' ')
        # print('    veces seguido el obj: {v}'.format(
        #     v=dd['times_object_followed'])
        # )
        print('{a:{j}.2f}%'.format(
            a=round(dd['times_object_followed']/dd['times_object_appear'] * 100, 2),
            j=just)
        )


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
    # # Bhatta verde
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='coffee_mug',
    #     objnum='5',
    #     param='definitivo_batta_green_channel',
    #     path='pruebas_guardadas',
    # )
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='cap',
    #     objnum='4',
    #     param='definitivo_batta_green_channel',
    #     path='pruebas_guardadas',
    # )
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_2.mat',
    #     scenenamenum='desk_2',
    #     objname='bowl',
    #     objnum='3',
    #     param='definitivo_batta_green_channel',
    #     path='pruebas_guardadas',
    # )
    #
    # # Correlation verde
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='coffee_mug',
    #     objnum='5',
    #     param='definitivo_correl_green_channel',
    #     path='pruebas_guardadas',
    # )
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='cap',
    #     objnum='4',
    #     param='definitivo_correl_green_channel',
    #     path='pruebas_guardadas',
    # )
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_2.mat',
    #     scenenamenum='desk_2',
    #     objname='bowl',
    #     objnum='3',
    #     param='definitivo_correl_green_channel',
    #     path='pruebas_guardadas',
    # )
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
    #
    # # RGB y HSV
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='coffee_mug',
    #     objnum='5',
    #     param='definitivo_RGB_staticdet',
    #     path='pruebas_guardadas',
    # )
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='cap',
    #     objnum='4',
    #     param='definitivo_RGB_staticdet',
    #     path='pruebas_guardadas',
    # )
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_2.mat',
    #     scenenamenum='desk_2',
    #     objname='bowl',
    #     objnum='3',
    #     param='definitivo_RGB_staticdet',
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

    # # Definitivo DEPTH
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






    # Prueba colgada del RGBD
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='coffee_mug',
    #     objnum='5',
    #     param='prueba_002',
    #     path='pruebas_guardadas',
    # )
    #
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='cap',
    #     objnum='4',
    #     param='prueba_001',
    #     path='pruebas_guardadas',
    # )
    #
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_2.mat',
    #     scenenamenum='desk_2',
    #     objname='bowl',
    #     objnum='3',
    #     param='prueba_001',
    #     path='pruebas_guardadas',
    # )

    # # Prueba colgada del RGB-HSV
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='coffee_mug',
    #     objnum='5',
    #     param='prueba_003',
    #     path='pruebas_guardadas',
    # )
    #
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='cap',
    #     objnum='4',
    #     param='prueba_002',
    #     path='pruebas_guardadas',
    # )
    #
    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_2.mat',
    #     scenenamenum='desk_2',
    #     objname='bowl',
    #     objnum='3',
    #     param='prueba_002',
    #     path='pruebas_guardadas',
    # )

