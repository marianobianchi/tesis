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
    param_values.sort()

    paramval_avgsvarsareas = []

    # Para cada parametro
    for param_value in param_values:
        param_path = os.path.join(
            path,
            scenenamenum,
            objnamenum,
            param,
            param_value,
        )
        overlapping_areas = []

        # Para cada corrida
        for run_num in os.listdir(param_path):
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
                    # metodo = values[2]
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
                        found_area = rectangle_found.area()
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
        paramval_avgsvarsareas.append((param_value, mean, std))

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
    for val, avg, std in paramval_avgsvarsareas:
        print('{v}:'.format(v=val))
        print('    prom_overlap: {p}'.format(
            p=round(np.array(avg) * 100, 2)
        ))
        print('    prom_overlap_std: {p}'.format(
            p=round(np.array(std) * 100, 2),
        ))


if __name__ == '__main__':
    # Frame size
    analizar_overlapping_por_parametro(
        matfile='videos/rgbd/scenes/desk/desk_1.mat',
        scenenamenum='desk_1',
        objname='coffee_mug',
        objnum='5',
        param='detection_frame_size',
        path='pruebas_guardadas',
    )
    analizar_overlapping_por_parametro(
        matfile='videos/rgbd/scenes/desk/desk_1.mat',
        scenenamenum='desk_1',
        objname='cap',
        objnum='4',
        param='detection_frame_size',
        path='pruebas_guardadas',
    )
    analizar_overlapping_por_parametro(
        matfile='videos/rgbd/scenes/desk/desk_2.mat',
        scenenamenum='desk_2',
        objname='bowl',
        objnum='3',
        param='detection_frame_size',
        path='pruebas_guardadas',
    )

    # Similarity threshold
    analizar_overlapping_por_parametro(
        matfile='videos/rgbd/scenes/desk/desk_1.mat',
        scenenamenum='desk_1',
        objname='coffee_mug',
        objnum='5',
        param='detection_similarity_threshold',
        path='pruebas_guardadas',
    )
    analizar_overlapping_por_parametro(
        matfile='videos/rgbd/scenes/desk/desk_1.mat',
        scenenamenum='desk_1',
        objname='cap',
        objnum='4',
        param='detection_similarity_threshold',
        path='pruebas_guardadas',
    )
    analizar_overlapping_por_parametro(
        matfile='videos/rgbd/scenes/desk/desk_2.mat',
        scenenamenum='desk_2',
        objname='bowl',
        objnum='3',
        param='detection_similarity_threshold',
        path='pruebas_guardadas',
    )

    # Inlier fraction
    analizar_overlapping_por_parametro(
        matfile='videos/rgbd/scenes/desk/desk_1.mat',
        scenenamenum='desk_1',
        objname='coffee_mug',
        objnum='5',
        param='detection_inlier_fraction',
        path='pruebas_guardadas',
    )
    analizar_overlapping_por_parametro(
        matfile='videos/rgbd/scenes/desk/desk_1.mat',
        scenenamenum='desk_1',
        objname='cap',
        objnum='4',
        param='detection_inlier_fraction',
        path='pruebas_guardadas',
    )
    analizar_overlapping_por_parametro(
        matfile='videos/rgbd/scenes/desk/desk_2.mat',
        scenenamenum='desk_2',
        objname='bowl',
        objnum='3',
        param='detection_inlier_fraction',
        path='pruebas_guardadas',
    )

    # Find percentage obj. model points
    analizar_overlapping_por_parametro(
        matfile='videos/rgbd/scenes/desk/desk_1.mat',
        scenenamenum='desk_1',
        objname='coffee_mug',
        objnum='5',
        param='find_perc_obj_model_points',
        path='pruebas_guardadas',
    )
    analizar_overlapping_por_parametro(
        matfile='videos/rgbd/scenes/desk/desk_1.mat',
        scenenamenum='desk_1',
        objname='cap',
        objnum='4',
        param='find_perc_obj_model_points',
        path='pruebas_guardadas',
    )
    analizar_overlapping_por_parametro(
        matfile='videos/rgbd/scenes/desk/desk_2.mat',
        scenenamenum='desk_2',
        objname='bowl',
        objnum='3',
        param='find_perc_obj_model_points',
        path='pruebas_guardadas',
    )

    # Fixed search area
    analizar_overlapping_por_parametro(
        matfile='videos/rgbd/scenes/desk/desk_1.mat',
        scenenamenum='desk_1',
        objname='coffee_mug',
        objnum='5',
        param='find_fixed_search_area',
        path='pruebas_guardadas',
    )
    analizar_overlapping_por_parametro(
        matfile='videos/rgbd/scenes/desk/desk_1.mat',
        scenenamenum='desk_1',
        objname='cap',
        objnum='4',
        param='find_fixed_search_area',
        path='pruebas_guardadas',
    )
    analizar_overlapping_por_parametro(
        matfile='videos/rgbd/scenes/desk/desk_2.mat',
        scenenamenum='desk_2',
        objname='bowl',
        objnum='3',
        param='find_fixed_search_area',
        path='pruebas_guardadas',
    )

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



