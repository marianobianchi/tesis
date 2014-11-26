#coding=utf-8

from __future__ import unicode_literals, division, print_function

import os
import codecs
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
import cv2

from metodos_comunes import dibujar_cuadrado
from detectores import StaticDetector


class Rectangle(object):
    def __init__(self, (top, left), (bottom, right)):
        self.top = float(top)
        self.left = float(left)
        self.bottom = float(bottom)
        self.right = float(right)

    def __repr__(self):
        return '({t}, {l}) - ({b}, {r})'.format(
            t=self.top,
            l=self.left,
            b=self.bottom,
            r=self.right,
        )

    def __eq__(self, other):
        eq = self.top == other.top
        eq = eq and self.left == other.left
        eq = eq and self.bottom == other.bottom
        eq = eq and self.right == other.right
        return eq

    def intersection(self, other):

        top = bottom = left = right = 0

        se_solapan = self.top <= other.top <= self.bottom
        se_solapan = other.top <= self.top <= other.bottom or se_solapan

        if se_solapan:
            top = max(self.top, other.top)
            bottom = min(self.bottom, other.bottom)
            left = max(self.left, other.left)
            right = min(self.right, other.right)

        return Rectangle((top, left), (bottom, right))

    def area(self):
        base = self.right - self.left
        height = self.bottom - self.top
        return base * height


def test_rectangle():
    a = Rectangle((0, 0), (2, 2))
    b = Rectangle((0, 0), (3, 1))

    gt_intersection = Rectangle((0, 0), (2, 1))

    a_intersection = a.intersection(b)
    assert gt_intersection == a_intersection, "Mal la interseccion"
    print(".", end='')

    b_intersection = b.intersection(a)
    assert b_intersection == a_intersection, "La interseccion no es conmutativa"
    print(".", end='')

    assert a.area() == 4, "Mal el area de a"
    print(".", end='')

    assert b.area() == 3, "Mal el area de b"
    print(".", end='')

    assert a_intersection.area() == 2, "Mal el area de la interseccion"
    print(".", end='')

    rect1 = Rectangle((202, 2), (345, 145))
    rect2 = Rectangle((203, 15), (314, 126))

    intersection = rect1.intersection(rect2)
    assert intersection == rect2, "La interseccion no dio bien"
    print(".", end='')


def promedio_frame_a_frame(matfile, scenenamenum, objname, objnum, param, path):
    # TODO: tratar de reducir los NEGROS (NN). Esto está relacionado con la
    # robustes del algoritmo

    RESULT_OVERAP = {
        'FP': 'Lo encontro pero no estaba',
        'FN': 'No lo encontro pero estaba',
        'BF': 'Lo encontro pero no se solapan',
        'VN': 'Ninguno lo encontro',
    }

    ground_truth = StaticDetector(
        matfile,
        objname,
    )

    objnamenum = '{name}_{num}'.format(name=objname, num=objnum)
    param_values = os.listdir(
        os.path.join(path, scenenamenum, objnamenum, param)
    )
    param_values.sort()

    # Valor de parametro y el promedio de solapamiento por frame
    paramval_avgsareaperframe = []

    # Valor de parametro y la explicacion del 0 en promedio de solapamiento
    paramval_explanationperframe = {}

    for param_value in param_values:
        param_path = os.path.join(
            path,
            scenenamenum,
            objnamenum,
            param,
            param_value,
        )

        frames_overlappedareas = []
        frames_explanations = []

        for run_num in os.listdir(param_path):
            frames_overlappedareas_per_run = []
            frames_explanations_per_run = []

            resultfile = os.path.join(param_path, run_num, 'results.txt')
            with codecs.open(resultfile, 'r', 'utf-8') as file_:
                reach_result_zone = False
                while not reach_result_zone:
                    line = file_.next()
                    reach_result_zone = line.startswith('RESULTS_SECTION')

                for line in file_.readlines():
                    values = [int(v) for v in line.split(';')]
                    nframe = values[0]
                    fue_exitoso = values[1]
                    metodo = values[2]
                    fila_sup = values[3]
                    col_izq = values[4]
                    fila_inf = values[5]
                    col_der = values[6]
                    # size = fila_inf - fila_sup

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
                    found_area = rectangle_found.area()
                    ground_truth_area = ground_truth_rectangle.area()
                    intersection_area = intersection.area()

                    overlap_area = 0

                    if gt_fue_exitoso and fue_exitoso and intersection_area > 0:
                        # To be considered a correct detection, the area of overlap A0
                        # between the predicted bounding box Bp and ground truth
                        # bounding box Bgt must exceed 50% by the formula:
                        # A0 = area(Bp intersection Bgt) / area(Bp union Bgt)
                        union_area = found_area + ground_truth_area - intersection_area
                        overlap_area = intersection_area / union_area
                        frames_explanations_per_run.append('LL')
                    elif gt_fue_exitoso and fue_exitoso and intersection_area == 0:
                        frames_explanations_per_run.append('BF')
                    elif not gt_fue_exitoso and fue_exitoso:
                        frames_explanations_per_run.append('FP')
                    elif gt_fue_exitoso and not fue_exitoso:
                        frames_explanations_per_run.append('FN')
                    else:  # ambos no exitosos
                        frames_explanations_per_run.append('VN')

                    frames_overlappedareas_per_run.append(overlap_area)

            frames_overlappedareas.append(frames_overlappedareas_per_run)
            frames_explanations.append(frames_explanations_per_run)

        overlappedareas_per_frame = zip(*frames_overlappedareas)
        explanations_per_frame = zip(*frames_explanations)

        avg_overlappedareas_per_frame = [np.mean(l) for l in overlappedareas_per_frame]
        explanations_per_frame = [exps[0] if len(set(exps)) == 1 else 'NN'
                                  for exps in explanations_per_frame]

        paramval_avgsareaperframe.append((param_value, avg_overlappedareas_per_frame))
        paramval_explanationperframe.update({param_value: explanations_per_frame})

    for param_val, avg_per_frame in paramval_avgsareaperframe:
        # # the figure
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # # the data
        n = len(avg_per_frame)
        avg_per_frame = np.array(avg_per_frame) * 100
        explanation_per_frame = paramval_explanationperframe[param_val]
        fps_x = [i+1 for i, exp in enumerate(explanation_per_frame)
                 if exp == 'FP']
        fns_x = [i+1 for i, exp in enumerate(explanation_per_frame)
                 if exp == 'FN']
        bfs_x = [i+1 for i, exp in enumerate(explanation_per_frame)
                 if exp == 'BF']
        vns_x = [i+1 for i, exp in enumerate(explanation_per_frame)
                 if exp == 'VN']
        nns_x = [i+1 for i, exp in enumerate(explanation_per_frame)
                 if exp == 'NN']

        # # plot
        line, = ax.plot(np.arange(1, n + 1), avg_per_frame, '-o')
        line.set_label('overlap per frame')

        if fps_x:
            fpsl, = ax.plot(fps_x, np.zeros(len(fps_x)), 'o', color='red')
            fpsl.set_label('Lo encontro pero no estaba')

        if fns_x:
            fnsl, = ax.plot(fns_x, np.zeros(len(fns_x)), 'o', color='orange')
            fnsl.set_label('No lo encontro pero estaba')

        if bfs_x:
            bfsl, = ax.plot(bfs_x, np.zeros(len(bfs_x)), 'o', color='yellow')
            bfsl.set_label('No se solaparon')

        if vns_x:
            vnsl, = ax.plot(vns_x, np.zeros(len(vns_x)), 'o', color='green')
            vnsl.set_label('Ninguno lo encontro')

        if nns_x:
            nnsl, = ax.plot(nns_x, np.zeros(len(nns_x)), 'o', color='black')
            nnsl.set_label('Distintos resultados por corrida')

        # # axes and labels
        ax.set_ylim(0, 100)
        ax.set_xlabel('Frame number')
        ax.set_xticks(np.arange(0, n+4, 5))
        ax.set_ylabel('average % of overlapping')
        ax.set_yticks(np.arange(0, 101, 5))
        ax.set_title(
            ('Overlapping entre ground truth y el algoritmo para {scn}, {obj}, '
             'con {prm} = {v}').format(
                scn=scenenamenum,
                obj=objnamenum,
                prm=param,
                v=param_val,
            )
        )

        ax.legend()

        plt.show()


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
    for param_value in param_values:
        param_path = os.path.join(
            path,
            scenenamenum,
            objnamenum,
            param,
            param_value,
        )
        means = []
        vars = []
        for run_num in os.listdir(param_path):
            overlapping_areas = []
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

                    if ground_truth_rectangle.area() > 0:
                        found_area = rectangle_found.area()
                        ground_truth_area = ground_truth_rectangle.area()
                        intersection_area = intersection.area()

                        # To be considered a correct detection, the area of overlap
                        # A0 between the predicted bounding box Bp and ground truth
                        # bounding box Bgt must exceed 50% by the formula:
                        # A0 = area(Bp intersection Bgt) / area(Bp union Bgt)
                        union_area = (
                            found_area + ground_truth_area - intersection_area
                        )
                        overlap_area = intersection_area / union_area

                        overlapping_areas.append(overlap_area)

                means.append(
                    np.mean(overlapping_areas) if overlapping_areas else 0
                )
                vars.append(
                    np.var(overlapping_areas) if overlapping_areas else 0
                )
        paramval_avgsvarsareas.append((param_value, means, vars))

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
    for val, avgs, variances in paramval_avgsvarsareas:
        strmeans = [unicode(round(m * 100, 2)) for m in avgs]
        strvars = [unicode(round(m * 100, 2)) for m in variances]
        print('{v}:'.format(v=val))
        print('    {m} ==> prom: {p}'.format(
            m=' | '.join(strmeans),
            p=round(np.mean(np.array(avgs) * 100), 2),
        ))
        print('    {m} ==> prom_var: {p}'.format(
            m=' | '.join(strvars),
            p=round(np.mean(np.array(variances) * 100), 2),
        ))


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

    for prm, prec, rec in mean_precs_recalls:
        plt.text(
            prec,
            rec * 1.1,
            '{h}'.format(h=prm),
            ha='center',
            va='bottom',
        )

    plt.show()


def dibujar_cuadros_encontrados_y_del_ground_truth():
    results_path = ('pruebas_guardadas/desk_1/coffee_mug_5/'
                    'detection_frame_size/2/01/results.txt')
    img_path_re = 'videos/rgbd/scenes/desk/desk_1/desk_1_{nframe}.png'

    ground_truth = StaticDetector(
        matfile_path='videos/rgbd/scenes/desk/desk_1.mat',
        obj_rgbd_name='coffee_mug',
    )

    with codecs.open(results_path, 'r', 'utf-8') as file_:
        reach_result_zone = False
        while not reach_result_zone:
            line = file_.next()
            reach_result_zone = line.startswith('RESULTS_SECTION')

        for line in file_.readlines():
            values = [int(v) for v in line.split(';')]

            nframe = values[0]
            # fue_exitoso = values[1]
            metodo = values[2]
            fila_sup = values[3]
            col_izq = values[4]
            fila_inf = values[5]
            col_der = values[6]
            # size = fila_inf - fila_sup

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

            if intersection.area() > 0:
                union_area = (
                    rectangle_found.area() + ground_truth_rectangle.area() - intersection.area()
                )
                overlap_area = intersection.area() / union_area
                if nframe == 56:
                    print(rectangle_found)
                    print(ground_truth_rectangle)
                    print("Overlap_area: " + unicode(overlap_area))

                fname = img_path_re.format(nframe=nframe)
                img = cv2.imread(fname, cv2.IMREAD_COLOR)

                img = dibujar_cuadrado(
                    img,
                    (gt_fila_sup, gt_col_izq),
                    (gt_fila_inf, gt_col_der),
                    color=(0, 255, 0)
                )
                algoritmo_color = (0, 0, 255)  # rojo si fue deteccion
                if metodo == 1:  # si fue seguimiento
                    algoritmo_color = (255, 0, 0)  # azul
                img = dibujar_cuadrado(
                    img,
                    (fila_sup, col_izq),
                    (fila_inf, col_der),
                    color=algoritmo_color,
                )

                # Muestro el resultado y espero que se apriete la tecla q
                # cv2.imshow(
                #     'Frame {n}, {obj}, {scene}'.format(n=nframe, obj='coffee_mug', scene='desk_1'),
                #     img
                # )
                #
                # while cv2.waitKey(1) & 0xFF != ord('q'):
                #     pass
                #
                # cv2.destroyAllWindows()


if __name__ == '__main__':
    promedio_frame_a_frame(
        matfile='videos/rgbd/scenes/desk/desk_2.mat',
        scenenamenum='desk_2',
        objname='bowl',
        objnum='3',
        param='detection_similarity_threshold',
        path='pruebas_guardadas',
    )

    promedio_frame_a_frame(
        matfile='videos/rgbd/scenes/desk/desk_2.mat',
        scenenamenum='desk_2',
        objname='bowl',
        objnum='3',
        param='detection_frame_size',
        path='pruebas_guardadas',
    )

    promedio_frame_a_frame(
        matfile='videos/rgbd/scenes/desk/desk_2.mat',
        scenenamenum='desk_2',
        objname='bowl',
        objnum='3',
        param='detection_inlier_fraction',
        path='pruebas_guardadas',
    )

    promedio_frame_a_frame(
        matfile='videos/rgbd/scenes/desk/desk_2.mat',
        scenenamenum='desk_2',
        objname='bowl',
        objnum='3',
        param='find_perc_obj_model_points',
        path='pruebas_guardadas',
    )

    # promedio_frame_a_frame(
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

    # analizar_overlapping_por_parametro(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='coffee_mug',
    #     objnum='5',
    #     param='find_fixed_search_area',
    #     path='pruebas_guardadas',
    # )



    # #
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

    # dibujar_cuadros_encontrados_y_del_ground_truth()
