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
                                            param, param_value, path,
                                            thresholds=None):
    """
    sacado de https://en.wikipedia.org/wiki/Precision_and_recall

    precision = positive predictive value = tp / tp + fp
    recall = sensitivity = correctas_encontradas/totales_correctas_ground_truth

    Considero como encontrada correctamente a aquellas instancias solapadas en
    más de un cierto porcentaje
    """
    if thresholds is None:
        thresholds = [round(x, 3) for x in np.linspace(10, 50, 401)]

    interest_thresholds = [15.0, 20.0, 25.0, 30.0, 40.0, 50.0]

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

    for run_num, dict_values in parser.iterar_por_parametro(param_value):
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
                overlap_area = data['intersection_area'] / union_area * 100
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

            if (estaba_y_no_se_encontro or
                    (estaba_y_se_encontro and se_solaparon_poco)):
                fn += 1
            elif no_estaba_y_se_encontro:
                fp += 1
            elif no_estaba_y_no_se_encontro:
                tn += 1
            elif estaba_y_se_encontro and not se_solaparon_poco:
                tp += 1
            else:
                raise Exception('Algo anda mal. Revisar condiciones')

        try:
            precs.append(tp / (tp + fp))
        except ZeroDivisionError:
            precs.append(0)
        try:
            recs.append(tp / (tp + fn))
        except ZeroDivisionError:
            recs.append(0)

    # the figure
    # ==========  ========
    # character   color
    # ==========  ========
    # 'b'         blue
    # 'g'         green
    # 'r'         red
    # 'c'         cyan
    # 'm'         magenta
    # 'y'         yellow
    # 'k'         black
    # 'w'         white
    # ==========  ========

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Paso a array de np
    precs = np.array(precs) * 100
    recs = np.array(recs) * 100
    thresholds = np.array(thresholds)

    # the data
    tupled_data = zip(precs, recs, thresholds)
    tupled_data.sort(reverse=True)

    ordered_precs = [p for p, r, t in tupled_data]
    ordered_recs = [r for p, r, t in tupled_data]

    ax.plot(ordered_precs, ordered_recs, 'bo-')

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
    i = 0
    for prec, rec, thresh in tupled_data[:1]:
        plt.text(
            prec,
            rec + 5 * (-1 * (i % 2)),
            '{h}'.format(h=thresh),
            ha='center',
            va='bottom',
        )
        i += 1

    colors = iter('grcmyk')
    interest = []
    for prec, rec, thresh in tupled_data[:-1]:
        if thresh in interest_thresholds:
            p, = ax.plot([prec], [rec], colors.next() + 'o')
            p.set_label('Threshold = {t}'.format(t=thresh))

            interest.append((prec, rec, thresh))

    # Ordeno por recall
    interest.sort(key=lambda (x, y, z): y, reverse=True)
    interest = [unicode(tuple([round(v, 2) for v in i])) for i in interest]

    ax.legend()

    print(
        'Precision vs recall para {scn}, {obj} y el parametro {prm}'.format(
            scn=scenename + '_' + scenenum,
            obj=objname + '_' + objnum,
            prm=param,
        )
    )
    print('Mejor (precision, recall, threshold): {x}'.format(
        x=unicode(tuple([round(v, 2) for v in tupled_data[0]])))
    )

    print(('Los de interes ordenados de mejor a peor recall (precision, '
           'recall, threshold): {xs}').format(xs=', '.join(interest)))
    plt.show()


def analizar_accuracy_por_parametro(matfile, scenename, scenenum, objname,
                                    objnum, param, param_value, path,
                                    thresholds=None):
    if thresholds is None:
        thresholds = [round(x, 3) for x in np.linspace(0, 100, 1001)]

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

    for run_num, dict_values in parser.iterar_por_parametro(param_value):
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

    accuracies = []

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
                overlap_area = data['intersection_area'] / union_area * 100
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

            if (estaba_y_no_se_encontro or
                    (estaba_y_se_encontro and se_solaparon_poco)):
                fn += 1
            elif no_estaba_y_se_encontro:
                fp += 1
            elif no_estaba_y_no_se_encontro:
                tn += 1
            elif estaba_y_se_encontro and not se_solaparon_poco:
                tp += 1
            else:
                raise Exception('Algo anda mal. Revisar condiciones')

        try:
            tptn = tp + tn
            fpfn = fp + fn
            accuracy = tptn / (tptn + fpfn)
            accuracies.append(accuracy)
        except ZeroDivisionError:
            accuracies.append(0)

    # the figure
    # ==========  ========
    # character   color
    # ==========  ========
    # 'b'         blue
    # 'g'         green
    # 'r'         red
    # 'c'         cyan
    # 'm'         magenta
    # 'y'         yellow
    # 'k'         black
    # 'w'         white
    # ==========  ========

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Paso a array de np
    accuracies = np.array(accuracies) * 100
    thresholds = np.array(thresholds)

    # the data
    ax.plot(thresholds, accuracies, 'b-')

    # # axes and labels
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel('Thresholds')
    ax.set_ylabel('Accuracy')
    xticks = range(0, 101, 10)
    yticks = range(0, 101, 10)
    plt.xticks(xticks)
    plt.yticks(yticks)
    ax.set_title(
        ('Accuracy por umbral de solapamiento para {scn}, {obj} y el parametro'
         ' {prm}').format(
            scn=scenename + '_' + scenenum,
            obj=objname + '_' + objnum,
            prm=param,
        )
    )

    accuracy_range__thr = {}
    step = 5
    for thr, acc in zip(thresholds, accuracies):
        acc_range = int(acc // step) * step
        if acc_range not in accuracy_range__thr:
            accuracy_range__thr[acc_range] = 0

        if accuracy_range__thr[acc_range] < thr:
            accuracy_range__thr[acc_range] = thr

    accuracies = accuracies.tolist()
    colors = iter('grcmyk' * 4)
    acc_range_thr = accuracy_range__thr.items()
    acc_range_thr.sort(reverse=True)
    for acc_range, thr in acc_range_thr:
        p, = ax.plot([thr], [acc_range], colors.next() + 'o')
        p.set_label('A:{a:.1f}, T:{t:.1f}'.format(a=acc_range, t=thr))

    ax.legend()
    plt.show()


def analizar_accuracy_por_sistema(sistema_data, param, param_value, path,
                                  thresholds=None):
    """
    sistema_data es una lista de diccionarios y cada diccionario tiene
    definidas las claves matfile, scenename, scenenum, objname, objnum
    """
    if thresholds is None:
        thresholds = [round(x, 3) for x in np.linspace(0, 100, 1001)]

    for d in sistema_data:
        # Obtengo los valores que necesito. Luego en d voy a guardar los datos
        # del seguimiento y el analisis de accuracy
        matfile = d['matfile']
        scenename = d['scenename']
        scenenum = d['scenenum']
        objname = d['objname']
        objnum = d['objnum']

        ground_truth = StaticDetector(
            matfile,
            objname,
            objnum,
        )

        parser = ResultsParser(
            scenename,
            scenenum,
            objname,
            objnum,
            param,
            path
        )

        # Guardo por cada frame el area encontrada, el area del ground truth,
        # el area de la interseccion, si el algoritmo fue exitoso y si el
        # ground truth dice haber sido exitoso
        data_per_frame = []

        for run_num, dict_values in parser.iterar_por_parametro(param_value):
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

        d.update({'data_per_frame': data_per_frame})

        accuracies = []

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
                    overlap_area = data['intersection_area'] / union_area * 100
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

                if (estaba_y_no_se_encontro or
                        (estaba_y_se_encontro and se_solaparon_poco)):
                    fn += 1
                elif no_estaba_y_se_encontro:
                    fp += 1
                elif no_estaba_y_no_se_encontro:
                    tn += 1
                elif estaba_y_se_encontro and not se_solaparon_poco:
                    tp += 1
                else:
                    raise Exception('Algo anda mal. Revisar condiciones')

            try:
                tptn = tp + tn
                fpfn = fp + fn
                accuracy = tptn / (tptn + fpfn)
                accuracies.append(accuracy)
            except ZeroDivisionError:
                accuracies.append(0)

        d.update({'accuracies': accuracies})

    # the figure
    # ==========  ========
    # character   color
    # ==========  ========
    # 'b'         blue
    # 'g'         green
    # 'r'         red
    # 'c'         cyan
    # 'm'         magenta
    # 'y'         yellow
    # 'k'         black
    # 'w'         white
    # ==========  ========

    fig = plt.figure()
    ax = fig.add_subplot(111)

    thresholds = np.array(thresholds)

    colors = iter('grcmyk' * 4)

    for d in sistema_data:
        accuracies = np.array(d['accuracies']) * 100

        # plotting
        p, = ax.plot(thresholds, accuracies, colors.next() + '-')
        p.set_label('{objname}_{objnum}'.format(
            objname=d['objname'],
            objnum=d['objnum'],
        ))

    # # axes and labels
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel('Umbral de solapamiento')
    ax.set_ylabel('Accuracy')
    xticks = range(0, 101, 10)
    yticks = range(0, 101, 10)
    plt.xticks(xticks)
    plt.yticks(yticks)
    ax.set_title('Accuracy por umbral de solapamiento para el sistema RGBD')

    plt.axvline(x=30, ymin=0, ymax=100, ls='--')

    ax.legend()
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
    # thresholds = [round(x, 3) for x in np.linspace(5, 80, 151)]
    # analizar_precision_recall_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenename='desk',
    #     scenenum='1',
    #     objname='coffee_mug',
    #     objnum='5',
    #     param='definitivo_automatico_RGBD',
    #     param_value='UNICO',
    #     path='pruebas_guardadas',
    #     thresholds=thresholds,
    # )
    # analizar_precision_recall_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenename='desk',
    #     scenenum='1',
    #     objname='cap',
    #     objnum='4',
    #     param='definitivo_automatico_RGBD',
    #     param_value='UNICO',
    #     path='pruebas_guardadas',
    #     thresholds=thresholds,
    # )
    # analizar_precision_recall_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_2.mat',
    #     scenename='desk',
    #     scenenum='2',
    #     objname='bowl',
    #     objnum='3',
    #     param='definitivo_automatico_RGBD',
    #     param_value='UNICO',
    #     path='pruebas_guardadas',
    #     thresholds=thresholds,
    # )
    # analizar_precision_recall_por_parametro(
    #     matfile='videos/rgbd/scenes/table/table_1.mat',
    #     scenename='table',
    #     scenenum='1',
    #     objname='coffee_mug',
    #     objnum='1',
    #     param='definitivo_automatico_RGBD',
    #     param_value='UNICO',
    #     path='pruebas_guardadas',
    #     thresholds=thresholds,
    # )
    # analizar_precision_recall_por_parametro(
    #     matfile='videos/rgbd/scenes/table/table_1.mat',
    #     scenename='table',
    #     scenenum='1',
    #     objname='soda_can',
    #     objnum='4',
    #     param='definitivo_automatico_RGBD',
    #     param_value='UNICO',
    #     path='pruebas_guardadas',
    #     thresholds=thresholds,
    # )
    # analizar_precision_recall_por_parametro(
    #     matfile='videos/rgbd/scenes/table_small/table_small_2.mat',
    #     scenename='table_small',
    #     scenenum='2',
    #     objname='cereal_box',
    #     objnum='4',
    #     param='definitivo_automatico_RGBD',
    #     param_value='UNICO',
    #     path='pruebas_guardadas',
    #     thresholds=thresholds,
    # )

    # RGB
    # analizar_precision_recall_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenename='desk',
    #     scenenum='1',
    #     objname='coffee_mug',
    #     objnum='5',
    #     param='definitivo_RGB_staticdet',
    #     param_value='DEFINITIVO',
    #     path='pruebas_guardadas',
    #     thresholds=thresholds,
    # )
    # analizar_precision_recall_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenename='desk',
    #     scenenum='1',
    #     objname='cap',
    #     objnum='4',
    #     param='definitivo_RGB_staticdet',
    #     param_value='DEFINITIVO',
    #     path='pruebas_guardadas',
    #     thresholds=thresholds,
    # )
    # analizar_precision_recall_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_2.mat',
    #     scenename='desk',
    #     scenenum='2',
    #     objname='bowl',
    #     objnum='3',
    #     param='definitivo_RGB_staticdet',
    #     param_value='DEFINITIVO',
    #     path='pruebas_guardadas',
    #     thresholds=thresholds,
    # )
    # analizar_precision_recall_por_parametro(
    #     matfile='videos/rgbd/scenes/table/table_1.mat',
    #     scenename='table',
    #     scenenum='1',
    #     objname='coffee_mug',
    #     objnum='1',
    #     param='definitivo_RGB_staticdet',
    #     param_value='DEFINITIVO',
    #     path='pruebas_guardadas',
    #     thresholds=thresholds,
    # )
    # analizar_precision_recall_por_parametro(
    #     matfile='videos/rgbd/scenes/table/table_1.mat',
    #     scenename='table',
    #     scenenum='1',
    #     objname='soda_can',
    #     objnum='4',
    #     param='definitivo_RGB_staticdet',
    #     param_value='DEFINITIVO',
    #     path='pruebas_guardadas',
    #     thresholds=thresholds,
    # )
    # analizar_precision_recall_por_parametro(
    #     matfile='videos/rgbd/scenes/table_small/table_small_2.mat',
    #     scenename='table_small',
    #     scenenum='2',
    #     objname='cereal_box',
    #     objnum='4',
    #     param='definitivo_RGB_staticdet',
    #     param_value='DEFINITIVO',
    #     path='pruebas_guardadas',
    #     thresholds=thresholds,
    # )

    # # DEPTH
    # analizar_precision_recall_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenename='desk',
    #     scenenum='1',
    #     objname='coffee_mug',
    #     objnum='5',
    #     param='definitivo_DEPTH',
    #     param_value='DEFINITIVO',
    #     path='pruebas_guardadas',
    #     thresholds=thresholds,
    # )
    # analizar_precision_recall_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenename='desk',
    #     scenenum='1',
    #     objname='cap',
    #     objnum='4',
    #     param='definitivo_DEPTH',
    #     param_value='DEFINITIVO',
    #     path='pruebas_guardadas',
    #     thresholds=thresholds,
    # )
    # analizar_precision_recall_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_2.mat',
    #     scenename='desk',
    #     scenenum='2',
    #     objname='bowl',
    #     objnum='3',
    #     param='definitivo_DEPTH',
    #     param_value='DEFINITIVO',
    #     path='pruebas_guardadas',
    #     thresholds=thresholds,
    # )
    # analizar_precision_recall_por_parametro(
    #     matfile='videos/rgbd/scenes/table/table_1.mat',
    #     scenename='table',
    #     scenenum='1',
    #     objname='coffee_mug',
    #     objnum='1',
    #     param='definitivo_DEPTH',
    #     param_value='DEFINITIVO',
    #     path='pruebas_guardadas',
    #     thresholds=thresholds,
    # )
    # analizar_precision_recall_por_parametro(
    #     matfile='videos/rgbd/scenes/table/table_1.mat',
    #     scenename='table',
    #     scenenum='1',
    #     objname='soda_can',
    #     objnum='4',
    #     param='definitivo_DEPTH',
    #     param_value='DEFINITIVO',
    #     path='pruebas_guardadas',
    #     thresholds=thresholds,
    # )
    # analizar_precision_recall_por_parametro(
    #     matfile='videos/rgbd/scenes/table_small/table_small_2.mat',
    #     scenename='table_small',
    #     scenenum='2',
    #     objname='cereal_box',
    #     objnum='4',
    #     param='definitivo_DEPTH',
    #     param_value='DEFINITIVO',
    #     path='pruebas_guardadas',
    #     thresholds=thresholds,
    # )
    ##############################
    # ANALISIS DE ACCURACY
    ##############################
    # thresholds = None
    #
    # analizar_accuracy_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenename='desk',
    #     scenenum='1',
    #     objname='coffee_mug',
    #     objnum='5',
    #     param='definitivo_automatico_RGBD',
    #     param_value='UNICO',
    #     path='pruebas_guardadas',
    #     thresholds=thresholds,
    # )
    # analizar_accuracy_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenename='desk',
    #     scenenum='1',
    #     objname='cap',
    #     objnum='4',
    #     param='definitivo_automatico_RGBD',
    #     param_value='UNICO',
    #     path='pruebas_guardadas',
    #     thresholds=thresholds,
    # )
    # analizar_accuracy_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_2.mat',
    #     scenename='desk',
    #     scenenum='2',
    #     objname='bowl',
    #     objnum='3',
    #     param='definitivo_automatico_RGBD',
    #     param_value='UNICO',
    #     path='pruebas_guardadas',
    #     thresholds=thresholds,
    # )
    # analizar_accuracy_por_parametro(
    #     matfile='videos/rgbd/scenes/table/table_1.mat',
    #     scenename='table',
    #     scenenum='1',
    #     objname='coffee_mug',
    #     objnum='1',
    #     param='definitivo_automatico_RGBD',
    #     param_value='UNICO',
    #     path='pruebas_guardadas',
    #     thresholds=thresholds,
    # )
    # analizar_accuracy_por_parametro(
    #     matfile='videos/rgbd/scenes/table/table_1.mat',
    #     scenename='table',
    #     scenenum='1',
    #     objname='soda_can',
    #     objnum='4',
    #     param='definitivo_automatico_RGBD',
    #     param_value='UNICO',
    #     path='pruebas_guardadas',
    #     thresholds=thresholds,
    # )
    # analizar_accuracy_por_parametro(
    #     matfile='videos/rgbd/scenes/table_small/table_small_2.mat',
    #     scenename='table_small',
    #     scenenum='2',
    #     objname='cereal_box',
    #     objnum='4',
    #     param='definitivo_automatico_RGBD',
    #     param_value='UNICO',
    #     path='pruebas_guardadas',
    #     thresholds=thresholds,
    # )
    #
    # # RGB
    # analizar_accuracy_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenename='desk',
    #     scenenum='1',
    #     objname='coffee_mug',
    #     objnum='5',
    #     param='definitivo_RGB_staticdet',
    #     param_value='DEFINITIVO',
    #     path='pruebas_guardadas',
    #     thresholds=thresholds,
    # )
    # analizar_accuracy_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenename='desk',
    #     scenenum='1',
    #     objname='cap',
    #     objnum='4',
    #     param='definitivo_RGB_staticdet',
    #     param_value='DEFINITIVO',
    #     path='pruebas_guardadas',
    #     thresholds=thresholds,
    # )
    # analizar_accuracy_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_2.mat',
    #     scenename='desk',
    #     scenenum='2',
    #     objname='bowl',
    #     objnum='3',
    #     param='definitivo_RGB_staticdet',
    #     param_value='DEFINITIVO',
    #     path='pruebas_guardadas',
    #     thresholds=thresholds,
    # )
    # analizar_accuracy_por_parametro(
    #     matfile='videos/rgbd/scenes/table/table_1.mat',
    #     scenename='table',
    #     scenenum='1',
    #     objname='coffee_mug',
    #     objnum='1',
    #     param='definitivo_RGB_staticdet',
    #     param_value='DEFINITIVO',
    #     path='pruebas_guardadas',
    #     thresholds=thresholds,
    # )
    # analizar_accuracy_por_parametro(
    #     matfile='videos/rgbd/scenes/table/table_1.mat',
    #     scenename='table',
    #     scenenum='1',
    #     objname='soda_can',
    #     objnum='4',
    #     param='definitivo_RGB_staticdet',
    #     param_value='DEFINITIVO',
    #     path='pruebas_guardadas',
    #     thresholds=thresholds,
    # )
    # analizar_accuracy_por_parametro(
    #     matfile='videos/rgbd/scenes/table_small/table_small_2.mat',
    #     scenename='table_small',
    #     scenenum='2',
    #     objname='cereal_box',
    #     objnum='4',
    #     param='definitivo_RGB_staticdet',
    #     param_value='DEFINITIVO',
    #     path='pruebas_guardadas',
    #     thresholds=thresholds,
    # )
    #
    # # DEPTH
    # analizar_accuracy_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenename='desk',
    #     scenenum='1',
    #     objname='coffee_mug',
    #     objnum='5',
    #     param='definitivo_DEPTH',
    #     param_value='DEFINITIVO',
    #     path='pruebas_guardadas',
    #     thresholds=thresholds,
    # )
    # analizar_accuracy_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenename='desk',
    #     scenenum='1',
    #     objname='cap',
    #     objnum='4',
    #     param='definitivo_DEPTH',
    #     param_value='DEFINITIVO',
    #     path='pruebas_guardadas',
    #     thresholds=thresholds,
    # )
    # analizar_accuracy_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_2.mat',
    #     scenename='desk',
    #     scenenum='2',
    #     objname='bowl',
    #     objnum='3',
    #     param='definitivo_DEPTH',
    #     param_value='DEFINITIVO',
    #     path='pruebas_guardadas',
    #     thresholds=thresholds,
    # )
    # analizar_accuracy_por_parametro(
    #     matfile='videos/rgbd/scenes/table/table_1.mat',
    #     scenename='table',
    #     scenenum='1',
    #     objname='coffee_mug',
    #     objnum='1',
    #     param='definitivo_DEPTH',
    #     param_value='DEFINITIVO',
    #     path='pruebas_guardadas',
    #     thresholds=thresholds,
    # )
    # analizar_accuracy_por_parametro(
    #     matfile='videos/rgbd/scenes/table/table_1.mat',
    #     scenename='table',
    #     scenenum='1',
    #     objname='soda_can',
    #     objnum='4',
    #     param='definitivo_DEPTH',
    #     param_value='DEFINITIVO',
    #     path='pruebas_guardadas',
    #     thresholds=thresholds,
    # )
    # analizar_accuracy_por_parametro(
    #     matfile='videos/rgbd/scenes/table_small/table_small_2.mat',
    #     scenename='table_small',
    #     scenenum='2',
    #     objname='cereal_box',
    #     objnum='4',
    #     param='definitivo_DEPTH',
    #     param_value='DEFINITIVO',
    #     path='pruebas_guardadas',
    #     thresholds=thresholds,
    # )
    #
    # # STATIC DETECTION, RGBD prefer D
    # analizar_accuracy_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenename='desk',
    #     scenenum='1',
    #     objname='coffee_mug',
    #     objnum='5',
    #     param='definitivo_RGBD_preferD',
    #     param_value='DEFINITIVO',
    #     path='pruebas_guardadas',
    #     thresholds=thresholds,
    # )
    # analizar_accuracy_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenename='desk',
    #     scenenum='1',
    #     objname='cap',
    #     objnum='4',
    #     param='definitivo_RGBD_preferD',
    #     param_value='DEFINITIVO',
    #     path='pruebas_guardadas',
    #     thresholds=thresholds,
    # )
    # analizar_accuracy_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_2.mat',
    #     scenename='desk',
    #     scenenum='2',
    #     objname='bowl',
    #     objnum='3',
    #     param='definitivo_RGBD_preferD',
    #     param_value='DEFINITIVO',
    #     path='pruebas_guardadas',
    #     thresholds=thresholds,
    # )
    # analizar_accuracy_por_parametro(
    #     matfile='videos/rgbd/scenes/table/table_1.mat',
    #     scenename='table',
    #     scenenum='1',
    #     objname='coffee_mug',
    #     objnum='1',
    #     param='definitivo_RGBD_preferD',
    #     param_value='DEFINITIVO',
    #     path='pruebas_guardadas',
    #     thresholds=thresholds,
    # )
    # analizar_accuracy_por_parametro(
    #     matfile='videos/rgbd/scenes/table/table_1.mat',
    #     scenename='table',
    #     scenenum='1',
    #     objname='soda_can',
    #     objnum='4',
    #     param='definitivo_RGBD_preferD',
    #     param_value='DEFINITIVO',
    #     path='pruebas_guardadas',
    #     thresholds=thresholds,
    # )
    # analizar_accuracy_por_parametro(
    #     matfile='videos/rgbd/scenes/table_small/table_small_2.mat',
    #     scenename='table_small',
    #     scenenum='2',
    #     objname='cereal_box',
    #     objnum='4',
    #     param='definitivo_RGBD_preferD',
    #     param_value='DEFINITIVO',
    #     path='pruebas_guardadas',
    #     thresholds=thresholds,
    # )

    sistema_data = [
        {
            'matfile': 'videos/rgbd/scenes/desk/desk_1.mat',
            'scenename': 'desk',
            'scenenum': '1',
            'objname': 'coffee_mug',
            'objnum': '5',
        },
        {
            'matfile': 'videos/rgbd/scenes/desk/desk_1.mat',
            'scenename': 'desk',
            'scenenum': '1',
            'objname': 'cap',
            'objnum': '4',
        },
        {
            'matfile': 'videos/rgbd/scenes/desk/desk_2.mat',
            'scenename': 'desk',
            'scenenum': '2',
            'objname': 'bowl',
            'objnum': '3',
        },
        {
            'matfile': 'videos/rgbd/scenes/table/table_1.mat',
            'scenename': 'table',
            'scenenum': '1',
            'objname': 'coffee_mug',
            'objnum': '1',
        },
        {
            'matfile': 'videos/rgbd/scenes/table/table_1.mat',
            'scenename': 'table',
            'scenenum': '1',
            'objname': 'soda_can',
            'objnum': '4',
        },
        {
            'matfile': 'videos/rgbd/scenes/table_small/table_small_2.mat',
            'scenename': 'table_small',
            'scenenum': '2',
            'objname': 'cereal_box',
            'objnum': '4',
        },
    ]
    analizar_accuracy_por_sistema(
        sistema_data,
        'definitivo_automatico_RGBD',
        'UNICO',
        'pruebas_guardadas',
    )
