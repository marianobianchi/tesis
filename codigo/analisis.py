#coding=utf-8

from __future__ import unicode_literals, division, print_function

import codecs
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

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


def analizar_resultados(matfile, scenenamenum, objname, resultfile):
    ground_truth = StaticDetector(
        matfile,
        objname,
    )

    nframe_area = []

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
            size = fila_inf - fila_sup

            ground_truth.update({'nframe': nframe})
            gt_fue_exitoso, gt_desc = ground_truth.detect()
            gt_size = gt_desc['size']
            gt_col_izq = gt_desc['location'][1]
            gt_fila_sup = gt_desc['location'][0]

            rectangle_found = Rectangle(
                (fila_sup, col_izq),
                (fila_inf, col_der)
            )
            ground_truth_rectangle = Rectangle(
                (gt_fila_sup, gt_col_izq),
                (gt_fila_sup + gt_size, gt_col_izq + gt_size)
            )
            intersection = rectangle_found.intersection(ground_truth_rectangle)

            if intersection.area() > 0:
                found_area = rectangle_found.area()
                ground_truth_area = ground_truth_rectangle.area()
                intersection_area = intersection.area()

                # To be considered a correct detection, the area of overlap A0
                # between the predicted bounding box Bp and ground truth bounding
                # box Bgt must exceed 50% by the formula:
                # A0 = area(Bp intersection Bgt) / area(Bp union Bgt)
                union_area = found_area + ground_truth_area - intersection_area
                overlap_area = intersection_area / union_area

                nframe_area.append((nframe, overlap_area))

    # Ploteo % de solapamiento
    areas = np.array([a for nf, a in nframe_area])

    p1 = plt.bar(
        np.arange(len(nframe_area)),  # x values
        areas * 100,  # y values
        align='center',
    )
    p2 = plt.plot(
        np.arange(len(nframe_area)),  # x values
        np.ones(len(nframe_area)) * np.mean(areas * 100),
        color=(1, 0, 0)
    )

    p2 = plt.plot(
        np.arange(len(nframe_area)),  # x values
        np.ones(len(nframe_area)) * 50,
        color=(0, 1, 0)
    )

    plt.title(('Area de solapamiento para ' + objname + ' en ' + scenenamenum))
    plt.xticks(
        np.arange(len(nframe_area)),
        [nf for nf, a in nframe_area]
    )
    plt.xlabel('numero de frame')
    plt.yticks(np.arange(0, 110, 10))
    plt.ylabel('% del area solapada')
    plt.legend([p2[0]], ['% Promedio'])

    plt.autoscale(axis='x')
    plt.show()


if __name__ == '__main__':
    analizar_resultados(
        matfile='videos/rgbd/scenes/desk/desk_1.mat',
        scenenamenum='desk_1',
        objname='coffee_mug',
        resultfile='pruebas_guardadas/desk_1/coffee_mug_5/detection_frame_size/4/results.txt'
    )

    # analizar_resultados(
    #     matfile='videos/rgbd/scenes/desk/desk_1.mat',
    #     scenenamenum='desk_1',
    #     objname='cap',
    #     resultfile='pruebas_guardadas/desk_1/cap_4/prueba_001/results.txt'
    # )
