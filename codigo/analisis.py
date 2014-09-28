#coding=utf-8

from __future__ import unicode_literals, division, print_function

import codecs

from detectores import StaticDetector

def analizar_resultados():
    ground_truth = StaticDetector(
        'videos/rgbd/scenes/desk/desk_1.mat',
        'coffee_mug',
    )
    fname = 'pruebas_guardadas/desk_1/coffee_mug_5/prueba_002/results.txt'
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

            print("################################")
            print("NFRAME:", nframe)
            print("POSTA    :", gt_fila_sup, gt_col_izq, 'TAM:', gt_size)
            print("ALGORITMO:", fila_sup, col_izq, 'TAM:', size)
            print("")
            print("")


if __name__ == '__main__':
    analizar_resultados()
