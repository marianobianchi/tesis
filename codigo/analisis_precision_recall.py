#coding=utf-8

from __future__ import unicode_literals, division, print_function

import os
import codecs
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

from detectores import StaticDetector
from analisis import Rectangle


def analizar_precision_recall_por_parametro(matfile, scenenamenum, objname,
                                            objnum, param, path):
    """
    sacado de https://en.wikipedia.org/wiki/Precision_and_recall

    precision = positive predictive value = correctas_encontradas/total_encontradas
    recall = sensitivity = correctas_encontradas/totales_correctas_ground_truth

    Considero como encontrada correctamente a aquellas instancias solapadas en
    más de un 50%
    """
    ground_truth = StaticDetector(
        matfile,
        objname,
        objnum,
    )
    objnamenum = '{name}_{num}'.format(name=objname, num=objnum)
    param_values = os.listdir(
        os.path.join(path, scenenamenum, objnamenum, param)
    )
    param_values.sort()

    paramval_precs_recs = []
    for param_value in param_values:
        param_path = os.path.join(
            path,
            scenenamenum,
            objnamenum,
            param,
            param_value
        )
        precs = []
        recs = []
        for run_num in os.listdir(param_path):
            correctas_encontradas = 0
            total_encontradas = 0
            totales_correctas_ground_truth = 0

            resultfile = os.path.join(param_path, run_num, 'results.txt')

            with codecs.open(resultfile, 'r', 'utf-8') as file_:
                reach_result_zone = False
                while not reach_result_zone:
                    line = file_.next()
                    reach_result_zone = line.startswith('RESULTS_SECTION')

                for line in file_:
                    values = [int(v) for v in line.split(';')]

                    nframe = values[0]
                    fue_exitoso = values[1]
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

                    total_encontradas += 1 if fue_exitoso else 0
                    totales_correctas_ground_truth += 1 if gt_fue_exitoso else 0

                    # Si el objeto está en la escena y se encontro,
                    # calculo si lo que se encontro es correcto
                    if gt_fue_exitoso and fue_exitoso:
                        found_area = rectangle_found.area()
                        ground_truth_area = ground_truth_rectangle.area()
                        intersection_area = intersection.area()
                        union_area = (
                            found_area + ground_truth_area - intersection_area
                        )
                        overlap_area = intersection_area / union_area

                        correctas_encontradas += 1 if overlap_area >= 0.5 else 0

            if total_encontradas > 0:
                precision = correctas_encontradas / total_encontradas
            else:
                precision = 0

            recall = correctas_encontradas / totales_correctas_ground_truth
            precs.append(precision)
            recs.append(recall)

        paramval_precs_recs.append(
            (param_value, np.array(precs), np.array(recs))
        )

    # # the figure
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # # the data
    n = len(paramval_precs_recs)
    mean_precs_recalls = [
        (prm, round(np.mean(precs * 100), 2), round(np.mean(recs * 100), 2))
        for prm, precs, recs in paramval_precs_recs
    ]

    mean_precs_recalls = sorted(
        mean_precs_recalls,
        key=lambda (val, avgprec, avgrec): avgprec
    )

    precs = [prec for prm, prec, rec in mean_precs_recalls]
    recs = [rec for prm, prec, rec in mean_precs_recalls]

    line, = ax.plot(precs, recs, '-o')

    # # axes and labels
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel('Precision')
    ax.set_ylabel('Recall')
    ax.set_title(
        'Precision vs recall para {scn}, {obj} y el parametro {prm}'.format(
            scn=scenenamenum,
            obj=objnamenum,
            prm=param,
        )
    )

    for i, (prm, prec, rec) in enumerate(mean_precs_recalls):
        plt.text(
            prec,
            rec + 1.1 + i * 2.5,
            '{h}'.format(h=prm),
            ha='center',
            va='bottom',
        )

    plt.show()


if __name__ == '__main__':
    # analizar_precision_recall_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='coffee_mug',
    #     objnum='5',
    #     param='detection_frame_size',
    #     path='pruebas_guardadas',
    # )
    # analizar_precision_recall_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='cap',
    #     objnum='4',
    #     param='detection_frame_size',
    #     path='pruebas_guardadas',
    # )
    # analizar_precision_recall_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_2.mat',
    #     scenenamenum='desk_2',
    #     objname='bowl',
    #     objnum='3',
    #     param='detection_frame_size',
    #     path='pruebas_guardadas',
    # )
    #
    # analizar_precision_recall_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='coffee_mug',
    #     objnum='5',
    #     param='find_perc_obj_model_points',
    #     path='pruebas_guardadas',
    # )
    # analizar_precision_recall_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='cap',
    #     objnum='4',
    #     param='find_perc_obj_model_points',
    #     path='pruebas_guardadas',
    # )
    # analizar_precision_recall_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_2.mat',
    #     scenenamenum='desk_2',
    #     objname='bowl',
    #     objnum='3',
    #     param='find_perc_obj_model_points',
    #     path='pruebas_guardadas',
    # )

    ###############
    # Analisis RGB
    ###############

    # Find frame threshold
    analizar_precision_recall_por_parametro(
        matfile='videos/rgbd/scenes/desk/desk_1.mat',
        scenenamenum='desk_1',
        objname='coffee_mug',
        objnum='5',
        param='RGB_find_frame_threshold',
        path='pruebas_guardadas',
    )
    analizar_precision_recall_por_parametro(
        matfile='videos/rgbd/scenes/desk/desk_1.mat',
        scenenamenum='desk_1',
        objname='cap',
        objnum='4',
        param='RGB_find_frame_threshold',
        path='pruebas_guardadas',
    )
    analizar_precision_recall_por_parametro(
        matfile='videos/rgbd/scenes/desk/desk_2.mat',
        scenenamenum='desk_2',
        objname='bowl',
        objnum='3',
        param='RGB_find_frame_threshold',
        path='pruebas_guardadas',
    )

    # Find template threshold
    analizar_precision_recall_por_parametro(
        matfile='videos/rgbd/scenes/desk/desk_1.mat',
        scenenamenum='desk_1',
        objname='coffee_mug',
        objnum='5',
        param='RGB_find_template_threshold',
        path='pruebas_guardadas',
    )
    analizar_precision_recall_por_parametro(
        matfile='videos/rgbd/scenes/desk/desk_1.mat',
        scenenamenum='desk_1',
        objname='cap',
        objnum='4',
        param='RGB_find_template_threshold',
        path='pruebas_guardadas',
    )
    analizar_precision_recall_por_parametro(
        matfile='videos/rgbd/scenes/desk/desk_2.mat',
        scenenamenum='desk_2',
        objname='bowl',
        objnum='3',
        param='RGB_find_template_threshold',
        path='pruebas_guardadas',
    )

    # Detection template threshold
    analizar_precision_recall_por_parametro(
        matfile='videos/rgbd/scenes/desk/desk_1.mat',
        scenenamenum='desk_1',
        objname='coffee_mug',
        objnum='5',
        param='RGB_det_template_threshold',
        path='pruebas_guardadas',
    )
    analizar_precision_recall_por_parametro(
        matfile='videos/rgbd/scenes/desk/desk_1.mat',
        scenenamenum='desk_1',
        objname='cap',
        objnum='4',
        param='RGB_det_template_threshold',
        path='pruebas_guardadas',
    )
    analizar_precision_recall_por_parametro(
        matfile='videos/rgbd/scenes/desk/desk_2.mat',
        scenenamenum='desk_2',
        objname='bowl',
        objnum='3',
        param='RGB_det_template_threshold',
        path='pruebas_guardadas',
    )

    # Detection template sizes
    analizar_precision_recall_por_parametro(
        matfile='videos/rgbd/scenes/desk/desk_1.mat',
        scenenamenum='desk_1',
        objname='coffee_mug',
        objnum='5',
        param='RGB_det_template_sizes',
        path='pruebas_guardadas',
    )
    analizar_precision_recall_por_parametro(
        matfile='videos/rgbd/scenes/desk/desk_1.mat',
        scenenamenum='desk_1',
        objname='cap',
        objnum='4',
        param='RGB_det_template_sizes',
        path='pruebas_guardadas',
    )
    analizar_precision_recall_por_parametro(
        matfile='videos/rgbd/scenes/desk/desk_2.mat',
        scenenamenum='desk_2',
        objname='bowl',
        objnum='3',
        param='RGB_det_template_sizes',
        path='pruebas_guardadas',
    )