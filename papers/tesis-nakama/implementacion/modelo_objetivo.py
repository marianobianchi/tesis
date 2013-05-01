#coding=utf-8

from __future__ import unicode_literals


import math
import re
import glob

import numpy
import cv

"""
Sea Xi con i=1...n las posiciones relativas al centroide del objetivo de los n
pixeles de la región definida como el objetivo centrado en el origen.
"""


def color_to_bin(color, nbin=4):
    """
    Es la funcion b de la seccion 6.1.2. Dado un color, devuelve el bin al que
    pertenece.

    (http://en.wikipedia.org/wiki/Color_histogram)

    Se espera que color sea de 3 dimensiones, como RGB o Lab.
    nbin es el numero de bines en el que se desea dividir a los 256 valores.
    """
    ancho_por_bin = 256 / nbin
    return (color[0] / ancho_por_bin,
            color[1] / ancho_por_bin,
            color[2] / ancho_por_bin)


def delta_de_kronecher(i, j):
    return 1 if i == j else 0


def perfil_del_kernel_gaussiano(x):
    return math.exp(-1.0 / 2 * x)


def norma2(punto):
    arr = numpy.array(punto)
    return numpy.linealg.norm(arr, 2)


def constante_de_normalizacion(x_estrella, perfil_o_kernel):
    norma_por_posicion = map(
        lambda x: (norma2(x) ** 2) * perfil_o_kernel,
        x_estrella
    )
    return 1.0 / sum(norma_por_posicion)


def modelo_objetivo(lab_image, x_estrella, perfil_o_kernel, nbin=4):
    q = []
    for u in range(nbin):
        calculo_por_punto = map(
            lambda x: perfil_o_kernel *
                      norma2(x) ** 2 *
                      delta_de_kronecher(color_to_bin(lab_image[x]), u),
            x_estrella
        )
        c = constante_de_normalizacion(x_estrella, perfil_o_kernel)
        q.append(c * sum(calculo_por_punto))

    return q


def levantar_imagenes(fregex='desk/*[0-9].png', nregex='_[1-9][0-9]*\.'):
    images_filenames = glob.glob(fregex)

    def get_im_number(fname):
        r = re.compile(nregex)
        match = r.search(fname)
        if match is None:
            raise Exception("Mal el nregex")
        return int(fname[match.start() + 1:match.end() - 1])

    images_filenames = sorted(images_filenames, key=get_im_number)

    images = []

    for image_fname in images_filenames:
        images.append(cv.LoadImage(image_fname))

    return images




def seguimiento():
    pass
