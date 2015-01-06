#coding=utf-8

from __future__ import unicode_literals, division, print_function

import os
import subprocess
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

        se_solapan_filas = self.top <= other.top <= self.bottom
        se_solapan_filas |= other.top <= self.top <= other.bottom
        se_solapan_columnas = self.left <= other.left <= self.right
        se_solapan_columnas |= other.left <= self.left <= other.right

        if se_solapan_filas and se_solapan_columnas:
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


def promedio_frame_a_frame(matfile, scenenamenum, objname, objnum, param, path,
                           param_values=None):
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
    if param_values is None:
        param_values = os.listdir(
            os.path.join(path, scenenamenum, objnamenum, param)
        )
        param_values.sort()
    else:
        param_values = [unicode(p) for p in param_values]
        param_values.sort()

    # Para que funcione con las pruebas_xxx
    param_values = [pv for pv in param_values
                    if os.path.isdir(os.path.join(path, scenenamenum, objnamenum, param, pv))]
    if len(param_values) == 0:
        param_values = ['']

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

        run_nums = os.listdir(param_path)
        run_nums = [rn for rn in run_nums if os.path.isdir(os.path.join(param_path, rn))]
        if len(run_nums) == 0:
            run_nums = ['']

        for run_num in run_nums:
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

        fps_x = []
        fns_x = []
        bfs_x = []
        vns_x = []
        nns_x = []

        for i, exp in enumerate(explanation_per_frame):
            if exp == 'FP':
                fps_x.append(i+1)
            elif exp == 'FN':
                fns_x.append(i+1)
            elif exp == 'BF':
                bfs_x.append(i+1)
            elif exp == 'VN':
                vns_x.append(i+1)
            elif exp == 'NN':
                nns_x.append(i+1)

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
        ax.set_ylim(-1, 100)
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


def dibujar_cuadros_encontrados_y_del_ground_truth():

    result_path = 'pruebas_guardadas/desk_1/coffee_mug_5/DEPTH_det_inlier_thresh/4/01/'
    result_data_path = result_path + 'results.txt'

    found_cloud_path_re = 'obj_found_scenepoints_frame_{nframe:03d}.pcd'
    cloud_path_re = 'videos/rgbd/scenes/desk/desk_1/desk_1_{nframe}.pcd'
    img_path_re = 'videos/rgbd/scenes/desk/desk_1/desk_1_{nframe}.png'

    ground_truth = StaticDetector(
        matfile_path='videos/rgbd/scenes/desk/desk_1.mat',
        obj_rgbd_name='coffee_mug',
    )

    with codecs.open(result_data_path, 'r', 'utf-8') as file_:
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
                cv2.imshow(
                    'Frame {n}, {obj}, {scene}'.format(n=nframe, obj='coffee_mug', scene='desk_1'),
                    img
                )

                while cv2.waitKey(1) & 0xFF != ord('q'):
                    pass

                cv2.destroyAllWindows()

                cloud_fname = cloud_path_re.format(nframe=nframe)
                found_cloud_fname = result_path + found_cloud_path_re.format(nframe=nframe)
                subprocess.call(['pcl_viewer', cloud_fname, found_cloud_fname])



if __name__ == '__main__':
    # # Frame size
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='coffee_mug',
    #     objnum='5',
    #     param='detection_frame_size',
    #     path='pruebas_guardadas',
    # )
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='cap',
    #     objnum='4',
    #     param='detection_frame_size',
    #     path='pruebas_guardadas',
    # )
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_2.mat',
    #     scenenamenum='desk_2',
    #     objname='bowl',
    #     objnum='3',
    #     param='detection_frame_size',
    #     path='pruebas_guardadas',
    # )
    #
    # # Similarity threshold
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='coffee_mug',
    #     objnum='5',
    #     param='detection_similarity_threshold',
    #     path='pruebas_guardadas',
    # )
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='cap',
    #     objnum='4',
    #     param='detection_similarity_threshold',
    #     path='pruebas_guardadas',
    # )
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_2.mat',
    #     scenenamenum='desk_2',
    #     objname='bowl',
    #     objnum='3',
    #     param='detection_similarity_threshold',
    #     path='pruebas_guardadas',
    # )
    #
    # # Inlier fraction
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='coffee_mug',
    #     objnum='5',
    #     param='detection_inlier_fraction',
    #     path='pruebas_guardadas',
    # )
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='cap',
    #     objnum='4',
    #     param='detection_inlier_fraction',
    #     path='pruebas_guardadas',
    # )
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_2.mat',
    #     scenenamenum='desk_2',
    #     objname='bowl',
    #     objnum='3',
    #     param='detection_inlier_fraction',
    #     path='pruebas_guardadas',
    # )
    #
    # # Find percentage obj. model points
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='coffee_mug',
    #     objnum='5',
    #     param='find_perc_obj_model_points',
    #     path='pruebas_guardadas',
    # )
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='cap',
    #     objnum='4',
    #     param='find_perc_obj_model_points',
    #     path='pruebas_guardadas',
    # )
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_2.mat',
    #     scenenamenum='desk_2',
    #     objname='bowl',
    #     objnum='3',
    #     param='find_perc_obj_model_points',
    #     path='pruebas_guardadas',
    # )
    #
    # # Fixed search area
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='coffee_mug',
    #     objnum='5',
    #     param='find_fixed_search_area',
    #     path='pruebas_guardadas',
    # )
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='cap',
    #     objnum='4',
    #     param='find_fixed_search_area',
    #     path='pruebas_guardadas',
    # )
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_2.mat',
    #     scenenamenum='desk_2',
    #     objname='bowl',
    #     objnum='3',
    #     param='find_fixed_search_area',
    #     path='pruebas_guardadas',
    # )

    # dibujar_cuadros_encontrados_y_del_ground_truth()


    ####################################
    # STATIC DETECTION and RGB analisis
    ####################################

    # # BATTA GREEN
    # # Find template threshold
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='coffee_mug',
    #     objnum='5',
    #     param='batta_green_channel_find_template_threshold',
    #     param_values=[0.5],
    #     path='pruebas_guardadas',
    # )
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='cap',
    #     objnum='4',
    #     param='batta_green_channel_find_template_threshold',
    #     param_values=[0.5],
    #     path='pruebas_guardadas',
    # )
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_2.mat',
    #     scenenamenum='desk_2',
    #     objname='bowl',
    #     objnum='3',
    #     param='batta_green_channel_find_template_threshold',
    #     param_values=[0.5],
    #     path='pruebas_guardadas',
    # )
    # # Find frame threshold
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='coffee_mug',
    #     objnum='5',
    #     param='batta_green_channel_find_frame_threshold',
    #     param_values=[0.2],
    #     path='pruebas_guardadas',
    # )
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='cap',
    #     objnum='4',
    #     param='batta_green_channel_find_frame_threshold',
    #     param_values=[0.2],
    #     path='pruebas_guardadas',
    # )
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_2.mat',
    #     scenenamenum='desk_2',
    #     objname='bowl',
    #     objnum='3',
    #     param='batta_green_channel_find_frame_threshold',
    #     param_values=[0.2],
    #     path='pruebas_guardadas',
    # )

    # # CHISQUARED GREEN
    # # Find template threshold
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='coffee_mug',
    #     objnum='5',
    #     param='chisquared_green_channel_find_template_threshold',
    #     path='pruebas_guardadas',
    # )
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='cap',
    #     objnum='4',
    #     param='chisquared_green_channel_find_template_threshold',
    #     path='pruebas_guardadas',
    # )
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_2.mat',
    #     scenenamenum='desk_2',
    #     objname='bowl',
    #     objnum='3',
    #     param='chisquared_green_channel_find_template_threshold',
    #     path='pruebas_guardadas',
    # )
    #
    # # Find frame threshold
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='coffee_mug',
    #     objnum='5',
    #     param='chisquared_green_channel_find_frame_threshold',
    #     path='pruebas_guardadas',
    # )
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='cap',
    #     objnum='4',
    #     param='chisquared_green_channel_find_frame_threshold',
    #     path='pruebas_guardadas',
    # )
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_2.mat',
    #     scenenamenum='desk_2',
    #     objname='bowl',
    #     objnum='3',
    #     param='chisquared_green_channel_find_frame_threshold',
    #     path='pruebas_guardadas',
    # )

    # # CORRELATION GREEN
    # # Find template threshold
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='coffee_mug',
    #     objnum='5',
    #     param='correl_green_channel_find_template_threshold',
    #     param_values=[600],
    #     path='pruebas_guardadas',
    # )
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='cap',
    #     objnum='4',
    #     param='correl_green_channel_find_template_threshold',
    #     param_values=[600],
    #     path='pruebas_guardadas',
    # )
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_2.mat',
    #     scenenamenum='desk_2',
    #     objname='bowl',
    #     objnum='3',
    #     param='correl_green_channel_find_template_threshold',
    #     param_values=[600],
    #     path='pruebas_guardadas',
    # )
    #
    # # Find frame threshold
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='coffee_mug',
    #     objnum='5',
    #     param='correl_green_channel_find_frame_threshold',
    #     param_values=[850],
    #     path='pruebas_guardadas',
    # )
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='cap',
    #     objnum='4',
    #     param='correl_green_channel_find_frame_threshold',
    #     param_values=[850],
    #     path='pruebas_guardadas',
    # )
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_2.mat',
    #     scenenamenum='desk_2',
    #     objname='bowl',
    #     objnum='3',
    #     param='correl_green_channel_find_frame_threshold',
    #     param_values=[850],
    #     path='pruebas_guardadas',
    # )

    # # CHISQUARED HS
    # # Find template threshold
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='coffee_mug',
    #     objnum='5',
    #     param='chisquared_hs_channels_find_template_threshold_fixed',
    #     param_values=[0.03],
    #     path='pruebas_guardadas',
    # )
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='cap',
    #     objnum='4',
    #     param='chisquared_hs_channels_find_template_threshold_fixed',
    #     param_values=[0.03],
    #     path='pruebas_guardadas',
    # )
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_2.mat',
    #     scenenamenum='desk_2',
    #     objname='bowl',
    #     objnum='3',
    #     param='chisquared_hs_channels_find_template_threshold_fixed',
    #     param_values=[0.03],
    #     path='pruebas_guardadas',
    # )
    #
    # # Find frame threshold
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='coffee_mug',
    #     objnum='5',
    #     param='chisquared_hs_channels_find_frame_threshold_fixed',
    #     param_values=[0.03],
    #     path='pruebas_guardadas',
    # )
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='cap',
    #     objnum='4',
    #     param='chisquared_hs_channels_find_frame_threshold_fixed',
    #     param_values=[0.03],
    #     path='pruebas_guardadas',
    # )
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_2.mat',
    #     scenenamenum='desk_2',
    #     objname='bowl',
    #     objnum='3',
    #     param='chisquared_hs_channels_find_frame_threshold_fixed',
    #     param_values=[0.03],
    #     path='pruebas_guardadas',
    # )

    # # MI METODO BHATTA BHATTA BHATTA
    # # Find template threshold
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='coffee_mug',
    #     objnum='5',
    #     param='mi_metodo_bhatta_bhatta_bhatta_template_perc',
    #     path='pruebas_guardadas',
    # )
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='cap',
    #     objnum='4',
    #     param='mi_metodo_bhatta_bhatta_bhatta_template_perc',
    #     path='pruebas_guardadas',
    # )
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_2.mat',
    #     scenenamenum='desk_2',
    #     objname='bowl',
    #     objnum='3',
    #     param='mi_metodo_bhatta_bhatta_bhatta_template_perc',
    #     path='pruebas_guardadas',
    # )
    #
    # # Find frame threshold
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='coffee_mug',
    #     objnum='5',
    #     param='mi_metodo_bhatta_bhatta_bhatta_frame_perc',
    #     path='pruebas_guardadas',
    # )
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='cap',
    #     objnum='4',
    #     param='mi_metodo_bhatta_bhatta_bhatta_frame_perc',
    #     path='pruebas_guardadas',
    # )
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_2.mat',
    #     scenenamenum='desk_2',
    #     objname='bowl',
    #     objnum='3',
    #     param='mi_metodo_bhatta_bhatta_bhatta_frame_perc',
    #     path='pruebas_guardadas',
    # )

    ##################
    # DEFINITIVOS
    ##################
    # Bhatta verde
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='coffee_mug',
    #     objnum='5',
    #     param='definitivo_batta_green_channel',
    #     path='pruebas_guardadas',
    # )
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='cap',
    #     objnum='4',
    #     param='definitivo_batta_green_channel',
    #     path='pruebas_guardadas',
    # )
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_2.mat',
    #     scenenamenum='desk_2',
    #     objname='bowl',
    #     objnum='3',
    #     param='definitivo_batta_green_channel',
    #     path='pruebas_guardadas',
    # )
    #
    # # Correlation verde
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='coffee_mug',
    #     objnum='5',
    #     param='definitivo_correl_green_channel',
    #     path='pruebas_guardadas',
    # )
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='cap',
    #     objnum='4',
    #     param='definitivo_correl_green_channel',
    #     path='pruebas_guardadas',
    # )
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_2.mat',
    #     scenenamenum='desk_2',
    #     objname='bowl',
    #     objnum='3',
    #     param='definitivo_correl_green_channel',
    #     path='pruebas_guardadas',
    # )

    # Mi metodo tripe bhatachayyra
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='coffee_mug',
    #     objnum='5',
    #     param='definitivo_mi_metodo_bhatta_bhatta_bhatta',
    #     path='pruebas_guardadas',
    # )
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='cap',
    #     objnum='4',
    #     param='definitivo_mi_metodo_bhatta_bhatta_bhatta',
    #     path='pruebas_guardadas',
    # )
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_2.mat',
    #     scenenamenum='desk_2',
    #     objname='bowl',
    #     objnum='3',
    #     param='definitivo_mi_metodo_bhatta_bhatta_bhatta',
    #     path='/home/mbianchi/Dropbox/marianobianchi08@gmail.com/Dropbox/pruebas_guardadas/',
    # )

    # # RGB y HSV
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='coffee_mug',
    #     objnum='5',
    #     param='definitivo_RGB_staticdet',
    #     path='pruebas_guardadas',
    # )
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='cap',
    #     objnum='4',
    #     param='definitivo_RGB_staticdet',
    #     path='pruebas_guardadas',
    # )
    # promedio_frame_a_frame(
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
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='coffee_mug',
    #     objnum='5',
    #     param='DEPTH_find_euclidean_fitness',
    #     path='pruebas_guardadas',
    #     param_values=['1e-05', '1e-10'],
    # )
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='cap',
    #     objnum='4',
    #     param='DEPTH_find_euclidean_fitness',
    #     path='pruebas_guardadas',
    #     param_values=['1e-05', '1e-10'],
    # )
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_2.mat',
    #     scenenamenum='desk_2',
    #     objname='bowl',
    #     objnum='3',
    #     param='DEPTH_find_euclidean_fitness',
    #     path='pruebas_guardadas',
    #     param_values=['1e-05', '1e-10'],
    # )

    # Transformation epsilon
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='coffee_mug',
    #     objnum='5',
    #     param='DEPTH_find_transformation_epsilon',
    #     path='pruebas_guardadas',
    #     param_values=['1e-05', '1e-06'],
    # )
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='cap',
    #     objnum='4',
    #     param='DEPTH_find_transformation_epsilon',
    #     path='pruebas_guardadas',
    #     param_values=['1e-05', '1e-06'],
    # )
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_2.mat',
    #     scenenamenum='desk_2',
    #     objname='bowl',
    #     objnum='3',
    #     param='DEPTH_find_transformation_epsilon',
    #     path='pruebas_guardadas',
    #     param_values=['1e-05', '1e-06'],
    # )

    # Correspondence distance
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='coffee_mug',
    #     objnum='5',
    #     param='DEPTH_find_correspondence_distance',
    #     path='pruebas_guardadas',
    # )
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='cap',
    #     objnum='4',
    #     param='DEPTH_find_correspondence_distance',
    #     path='pruebas_guardadas',
    # )
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_2.mat',
    #     scenenamenum='desk_2',
    #     objname='bowl',
    #     objnum='3',
    #     param='DEPTH_find_correspondence_distance',
    #     path='pruebas_guardadas',
    # )

    # Percentage obj. model points
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='coffee_mug',
    #     objnum='5',
    #     param='DEPTH_find_perc_obj_model_points',
    #     path='pruebas_guardadas',
    #     param_values=['0.4', '0.3'],
    # )
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='cap',
    #     objnum='4',
    #     param='DEPTH_find_perc_obj_model_points',
    #     path='pruebas_guardadas',
    #     param_values=['0.4', '0.3'],
    # )
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_2.mat',
    #     scenenamenum='desk_2',
    #     objname='bowl',
    #     objnum='3',
    #     param='DEPTH_find_perc_obj_model_points',
    #     path='pruebas_guardadas',
    #     param_values=['0.4', '0.3'],
    # )


    # # Umbral score
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='coffee_mug',
    #     objnum='5',
    #     param='DEPTH_find_umbral_score',
    #     path='pruebas_guardadas',
    #     param_values=['0.04', '0.03', '0.01', '0.001', '0.0001'],
    # )
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='cap',
    #     objnum='4',
    #     param='DEPTH_find_umbral_score',
    #     path='pruebas_guardadas',
    #     param_values=['0.04', '0.03', '0.01', '0.001', '0.0001'],
    # )
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_2.mat',
    #     scenenamenum='desk_2',
    #     objname='bowl',
    #     objnum='3',
    #     param='DEPTH_find_umbral_score',
    #     path='pruebas_guardadas',
    #     param_values=['0.04'],
    # )


    # # Detection max iterations
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='coffee_mug',
    #     objnum='5',
    #     param='DEPTH_det_max_iter',
    #     path='pruebas_guardadas',
    #     param_values=['120'],
    # )
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='cap',
    #     objnum='4',
    #     param='DEPTH_det_max_iter',
    #     path='pruebas_guardadas',
    #     param_values=['120'],
    # )
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_2.mat',
    #     scenenamenum='desk_2',
    #     objname='bowl',
    #     objnum='3',
    #     param='DEPTH_det_max_iter',
    #     path='pruebas_guardadas',
    #     param_values=['120'],
    # )
    #
    #
    # # Detection points to sample
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='coffee_mug',
    #     objnum='5',
    #     param='DEPTH_det_points_to_sample',
    #     path='pruebas_guardadas',
    #     param_values=['3', '5'],
    # )
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='cap',
    #     objnum='4',
    #     param='DEPTH_det_points_to_sample',
    #     path='pruebas_guardadas',
    #     param_values=['3', '5'],
    # )
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_2.mat',
    #     scenenamenum='desk_2',
    #     objname='bowl',
    #     objnum='3',
    #     param='DEPTH_det_points_to_sample',
    #     path='pruebas_guardadas',
    #     param_values=['3', '5'],
    # )
    #
    #
    # # Detection nearest features
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='coffee_mug',
    #     objnum='5',
    #     param='DEPTH_det_nearest_features',
    #     path='pruebas_guardadas',
    #     param_values=['4'],
    # )
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='cap',
    #     objnum='4',
    #     param='DEPTH_det_nearest_features',
    #     path='pruebas_guardadas',
    #     param_values=['4'],
    # )
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_2.mat',
    #     scenenamenum='desk_2',
    #     objname='bowl',
    #     objnum='3',
    #     param='DEPTH_det_nearest_features',
    #     path='pruebas_guardadas',
    #     param_values=['4'],
    # )
    #
    #
    # # Detection similarity threshold
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='coffee_mug',
    #     objnum='5',
    #     param='DEPTH_det_simil_thresh',
    #     path='pruebas_guardadas',
    #     param_values=['0.6'],
    # )
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='cap',
    #     objnum='4',
    #     param='DEPTH_det_simil_thresh',
    #     path='pruebas_guardadas',
    #     param_values=['0.6'],
    # )
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_2.mat',
    #     scenenamenum='desk_2',
    #     objname='bowl',
    #     objnum='3',
    #     param='DEPTH_det_simil_thresh',
    #     path='pruebas_guardadas',
    #     param_values=['0.6'],
    # )
    #
    #
    # # detection inlier threshold
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='coffee_mug',
    #     objnum='5',
    #     param='DEPTH_det_inlier_thresh',
    #     path='pruebas_guardadas',
    #     param_values=['4'],
    # )
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='cap',
    #     objnum='4',
    #     param='DEPTH_det_inlier_thresh',
    #     path='pruebas_guardadas',
    #     param_values=['4'],
    # )
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_2.mat',
    #     scenenamenum='desk_2',
    #     objname='bowl',
    #     objnum='3',
    #     param='DEPTH_det_inlier_thresh',
    #     path='pruebas_guardadas',
    #     param_values=['4'],
    # )


    # # Detection inlier fraction
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='coffee_mug',
    #     objnum='5',
    #     param='DEPTH_det_inlier_fraction',
    #     path='pruebas_guardadas',
    #     param_values=['0.3'],
    # )
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='cap',
    #     objnum='4',
    #     param='DEPTH_det_inlier_fraction',
    #     path='pruebas_guardadas',
    #     param_values=['0.3'],
    # )
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_2.mat',
    #     scenenamenum='desk_2',
    #     objname='bowl',
    #     objnum='3',
    #     param='DEPTH_det_inlier_fraction',
    #     path='pruebas_guardadas',
    #     param_values=['0.3'],
    # )

    ###################
    # Seguimiento RGBD
    ###################
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='coffee_mug',
    #     objnum='5',
    #     param='prueba_002',
    #     path='pruebas_guardadas',
    # )
    #
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='cap',
    #     objnum='4',
    #     param='prueba_001',
    #     path='pruebas_guardadas',
    # )
    #
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_2.mat',
    #     scenenamenum='desk_2',
    #     objname='bowl',
    #     objnum='3',
    #     param='prueba_001',
    #     path='pruebas_guardadas',
    # )

    # Definitivo DEPTH
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='coffee_mug',
    #     objnum='5',
    #     param='definitivo_DEPTH',
    #     path='pruebas_guardadas',
    # )
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='cap',
    #     objnum='4',
    #     param='definitivo_DEPTH',
    #     path='pruebas_guardadas',
    # )
    # promedio_frame_a_frame(
    #     matfile='videos/rgbd/scenes/desk/desk_2.mat',
    #     scenenamenum='desk_2',
    #     objname='bowl',
    #     objnum='3',
    #     param='definitivo_DEPTH',
    #     path='pruebas_guardadas',
    # )

    # dibujar_cuadros_encontrados_y_del_ground_truth()

    # Prueba colgada hsv-rgb
    promedio_frame_a_frame(
        matfile='videos/rgbd/scenes/desk/desk_1.mat',
        scenenamenum='desk_1',
        objname='coffee_mug',
        objnum='5',
        param='prueba_003',
        path='pruebas_guardadas',
    )

    promedio_frame_a_frame(
        matfile='videos/rgbd/scenes/desk/desk_1.mat',
        scenenamenum='desk_1',
        objname='cap',
        objnum='4',
        param='prueba_002',
        path='pruebas_guardadas',
    )

    promedio_frame_a_frame(
        matfile='videos/rgbd/scenes/desk/desk_2.mat',
        scenenamenum='desk_2',
        objname='bowl',
        objnum='3',
        param='prueba_002',
        path='pruebas_guardadas',
    )