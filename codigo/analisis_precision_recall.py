#coding=utf-8

from __future__ import unicode_literals, division, print_function

import os
import codecs
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np

from detectores import StaticDetector
from analisis import Rectangle


class ResultsParser(object):
    def __init__(self, scenename, scenenum, objname, objnum, param, path):
        # Obtengo la lista de valores del parametro explorados
        self.path = path
        self.scenename = scenename
        self.scenenum = scenenum
        self.objname = objname
        self.objnum = objnum
        self.param = param

        objnamenum = '{name}_{num}'.format(name=objname, num=objnum)
        param_values = os.listdir(
            os.path.join(
                path,
                scenename + '_' + scenenum,
                objname + '_' + objnum,
                param
            )
        )
        param_values.sort()
        self.param_values = param_values

    def parameter_values(self):
        return self.param_values[:]

    def iterar_todo(self):
        for param_value in self.param_values:
            for run_num, dict_values in self.iterar_por_parametro(param_value):
                yield param_value, run_num, dict_values

    def iterar_por_parametro(self, param_value):
        """
        Es un iterador que para el valor de parametro explorado va devolviendo
        el numero de corrida y los valores recolectados del archivo de
        resultados
        """
        param_path = os.path.join(
            self.path,
            self.scenename + '_' + self.scenenum,
            self.objname + '_' + self.objnum,
            self.param,
            param_value
        )
        # Para cada corrida con ese valor
        for run_num in os.listdir(param_path):
            resultfile = os.path.join(param_path, run_num, 'results.txt')

            with codecs.open(resultfile, 'r', 'utf-8') as file_:
                reach_result_zone = False
                while not reach_result_zone:
                    line = file_.next()
                    reach_result_zone = line.startswith('RESULTS_SECTION')

                for line in file_:
                    values = [int(v) for v in line.split(';')]

                    dict_values = {
                        'nframe': values[0],
                        'fue_exitoso': values[1],
                        'metodo': values[2],
                        'fila_sup': values[3],
                        'col_izq': values[4],
                        'fila_inf': values[5],
                        'col_der': values[6],
                    }
                    yield run_num, dict_values


def analizar_precision_recall_por_parametro(matfile, scenename, scenenum,
                                            objname, objnum,
                                            param, path,
                                            thresholds=None):
    """
    sacado de https://en.wikipedia.org/wiki/Precision_and_recall

    precision = positive predictive value = tp / tp + fp
    recall = sensitivity = correctas_encontradas/totales_correctas_ground_truth

    Considero como encontrada correctamente a aquellas instancias solapadas en
    más de un cierto porcentaje
    """
    if thresholds is None:
        thresholds = [round(x, 3) for x in np.linspace(0.1, 0.5, 401)]

    ground_truth = StaticDetector(
        matfile,
        objname,
        objnum,
    )

    parser = ResultsParser(scenename, scenenum, objname, objnum, param, path)

    # Guardo por cada frame el area encontrada, el area del ground truth,
    # el area de la interseccion, si el algoritmo fue exitoso y si el ground
    # truth dice haber sido exitoso
    data_per_frame = []

    for run_num, dict_values in parser.iterar_por_parametro('UNICO'):
        # Resultados de mi algoritmo
        nframe = dict_values['nframe']
        fue_exitoso = True if dict_values['fue_exitoso'] == 1 else False
        fila_sup = dict_values['fila_sup']
        col_izq = dict_values['col_izq']
        fila_inf = dict_values['fila_inf']
        col_der = dict_values['col_der']

        # Resultados del ground truth
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

        data_per_frame.append({
            'found_area': found_area,
            'ground_truth_area': ground_truth_area,
            'intersection_area': intersection_area,
            'fue_exitoso': fue_exitoso,
            'gt_fue_exitoso': gt_fue_exitoso,
        })

    precs = []
    recs = []

    for min_overlapped_area in thresholds:

        tp = 0
        fp = 0
        fn = 0
        tn = 0

        for data in data_per_frame:

            se_solaparon_poco = True
            if data['found_area'] > 0 and data['ground_truth_area'] > 0:
                union_area = (
                    data['found_area'] +
                    data['ground_truth_area'] -
                    data['intersection_area']
                )
                overlap_area = data['intersection_area'] / union_area
                se_solaparon_poco = overlap_area < min_overlapped_area

            # Chequeo de falsos positivos y negativos y
            # verdaderos negativos y positivos
            estaba_y_no_se_encontro = (
                data['gt_fue_exitoso'] and not data['fue_exitoso']
            )

            no_estaba_y_se_encontro = (
                data['ground_truth_area'] == 0 and data['fue_exitoso']
            )

            no_estaba_y_no_se_encontro = (
                not (data['gt_fue_exitoso'] or data['fue_exitoso'])
            )

            estaba_y_se_encontro = (
                data['ground_truth_area'] > 0 and data['fue_exitoso']
            )

            if estaba_y_no_se_encontro:
                fn += 1
            elif (no_estaba_y_se_encontro or
                    (estaba_y_se_encontro and se_solaparon_poco)):
                fp += 1
            elif no_estaba_y_no_se_encontro:
                tn += 1
            elif estaba_y_se_encontro and not se_solaparon_poco:
                tp += 1
            else:
                raise Exception('Algo anda mal. Revisar condiciones')

        precs.append(tp / (tp + fp))
        recs.append(tp / (tp + fn))

    # the figure
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # the data
    tupled_data = zip(precs, recs, thresholds)
    tupled_data.sort()

    precs = np.array([prec for prec, rec, thresh in tupled_data]) * 100
    recs = np.array([rec for prec, rec, thresh in tupled_data]) * 100

    print(
        'La cantidad de objetos repetidos es: {n}'.format(
            n=len(zip(precs, recs)) - len(set(zip(precs, recs)))
        )
    )

    ax.plot(precs, recs, '-o')

    # # axes and labels
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel('Precision')
    ax.set_ylabel('Recall')
    xticks = range(0, 101, 20)
    yticks = range(0, 101, 20)
    plt.xticks(xticks)
    plt.yticks(yticks)
    ax.set_title(
        'Precision vs recall para {scn}, {obj} y el parametro {prm}'.format(
            scn=scenename + '_' + scenenum,
            obj=objname + '_' + objnum,
            prm=param,
        )
    )

    for i, (prec, rec, thresh) in enumerate(tupled_data[::100]):
        plt.text(
            prec * 100,
            rec * 100 + 5 * (-1 * (i % 2)),
            '{h}'.format(h=thresh),
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

    # # Find frame threshold
    # analizar_precision_recall_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='coffee_mug',
    #     objnum='5',
    #     param='RGB_find_frame_threshold',
    #     path='pruebas_guardadas',
    # )
    # analizar_precision_recall_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='cap',
    #     objnum='4',
    #     param='RGB_find_frame_threshold',
    #     path='pruebas_guardadas',
    # )
    # analizar_precision_recall_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_2.mat',
    #     scenenamenum='desk_2',
    #     objname='bowl',
    #     objnum='3',
    #     param='RGB_find_frame_threshold',
    #     path='pruebas_guardadas',
    # )
    #
    # # Find template threshold
    # analizar_precision_recall_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='coffee_mug',
    #     objnum='5',
    #     param='RGB_find_template_threshold',
    #     path='pruebas_guardadas',
    # )
    # analizar_precision_recall_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='cap',
    #     objnum='4',
    #     param='RGB_find_template_threshold',
    #     path='pruebas_guardadas',
    # )
    # analizar_precision_recall_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_2.mat',
    #     scenenamenum='desk_2',
    #     objname='bowl',
    #     objnum='3',
    #     param='RGB_find_template_threshold',
    #     path='pruebas_guardadas',
    # )
    #
    # # Detection template threshold
    # analizar_precision_recall_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='coffee_mug',
    #     objnum='5',
    #     param='RGB_det_template_threshold',
    #     path='pruebas_guardadas',
    # )
    # analizar_precision_recall_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='cap',
    #     objnum='4',
    #     param='RGB_det_template_threshold',
    #     path='pruebas_guardadas',
    # )
    # analizar_precision_recall_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_2.mat',
    #     scenenamenum='desk_2',
    #     objname='bowl',
    #     objnum='3',
    #     param='RGB_det_template_threshold',
    #     path='pruebas_guardadas',
    # )
    #
    # # Detection template sizes
    # analizar_precision_recall_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='coffee_mug',
    #     objnum='5',
    #     param='RGB_det_template_sizes',
    #     path='pruebas_guardadas',
    # )
    # analizar_precision_recall_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='cap',
    #     objnum='4',
    #     param='RGB_det_template_sizes',
    #     path='pruebas_guardadas',
    # )
    # analizar_precision_recall_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_2.mat',
    #     scenenamenum='desk_2',
    #     objname='bowl',
    #     objnum='3',
    #     param='RGB_det_template_sizes',
    #     path='pruebas_guardadas',
    # )

    ###########################################################################
    # SISTEMA DE SEGUIMIENTO RGB-D
    ###########################################################################
    analizar_precision_recall_por_parametro(
        matfile='videos/rgbd/scenes/desk/desk_1.mat',
        scenename='desk',
        scenenum='1',
        objname='coffee_mug',
        objnum='5',
        param='definitivo_automatico_RGBD',
        path='pruebas_guardadas',
    )
    analizar_precision_recall_por_parametro(
        matfile='videos/rgbd/scenes/desk/desk_1.mat',
        scenename='desk',
        scenenum='1',
        objname='cap',
        objnum='4',
        param='definitivo_automatico_RGBD',
        path='pruebas_guardadas',
    )
    analizar_precision_recall_por_parametro(
        matfile='videos/rgbd/scenes/desk/desk_2.mat',
        scenename='desk',
        scenenum='2',
        objname='bowl',
        objnum='3',
        param='definitivo_automatico_RGBD',
        path='pruebas_guardadas',
    )
    analizar_precision_recall_por_parametro(
        matfile='videos/rgbd/scenes/table/table_1.mat',
        scenename='table',
        scenenum='1',
        objname='coffee_mug',
        objnum='1',
        param='definitivo_automatico_RGBD',
        path='pruebas_guardadas',
    )
    analizar_precision_recall_por_parametro(
        matfile='videos/rgbd/scenes/table/table_1.mat',
        scenename='table',
        scenenum='1',
        objname='soda_can',
        objnum='4',
        param='definitivo_automatico_RGBD',
        path='pruebas_guardadas',
    )
    analizar_precision_recall_por_parametro(
        matfile='videos/rgbd/scenes/table_small/table_small_2.mat',
        scenename='table_small',
        scenenum='2',
        objname='cereal_box',
        objnum='4',
        param='definitivo_automatico_RGBD',
        path='pruebas_guardadas',
    )
