#coding=utf-8

from __future__ import unicode_literals, division, print_function

import codecs

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

        if self.top <= other.top <= self.bottom <= other.bottom:
            top = other.top
            bottom = self.bottom
        elif other.top <= self.top <= other.bottom <= self.bottom:
            top = self.top
            bottom = other.bottom

        if self.left <= other.left <= self.right <= other.right:
            left = other.left
            right = self.right
        elif other.left <= self.left <= other.right <= self.right:
            left = self.left
            right = other.right

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


def analizar_resultados():
    ground_truth = StaticDetector(
        'videos/rgbd/scenes/desk/desk_1.mat',
        'coffee_mug',
    )
    fname = 'pruebas_guardadas/desk_1/coffee_mug_5/prueba_002/results.txt'

    nframe_area = []

    with codecs.open(fname, 'r', 'utf-8') as file_:
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

    for nframe, overlap_area in nframe_area:
        print("################################")
        print("NFRAME:", nframe)
        print("overlap_area:", overlap_area)

    print("##### MAS DATOS #####")
    areas = [a for n,a in nframe_area]
    print("Min. overlap:", min(areas))
    print("Max. overlap:", max(areas))
    print("Avg. overlap:", sum(areas)/len(areas))


if __name__ == '__main__':
    analizar_resultados()
