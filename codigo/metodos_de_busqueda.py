#coding=utf-8

from __future__ import unicode_literals, division

import numpy as np

class BusquedaAlrededor(object):
    def get_positions_and_framesizes(self, topleft, bottomright,
                                     filas, columnas):
        top, left = topleft
        bottom, right = bottomright

        height = bottom - top
        width = right - left

        diff = 0.25

        for i in [1, 2, 3, 4]:
            actual_diff = diff * i
            actual_x_diff = height * actual_diff
            actual_y_diff = width * actual_diff
            x = top - actual_x_diff
            y = left - actual_y_diff

            for x_move in range(3):
                for y_move in range(3):
                    next_x = x + x_move * actual_x_diff
                    next_y = y + y_move * actual_y_diff
                    if (0 <= next_x < next_x + height <= filas and
                            0 <= next_y < next_y + width <= columnas):
                        yield ((next_x, next_y),
                               (next_x + height, next_y + width))


class BusquedaAlrededorCambiandoFrameSize(object):
    def get_positions_and_framesizes(self, topleft, bottomright,
                                     filas, columnas):
        top, left = topleft
        bottom, right = bottomright

        original_height = bottom - top
        original_width = right - left

        for tam_diff in 0.75, 1, 1.25:

            height = max(int(original_height * tam_diff), 1)
            width = max(int(original_width * tam_diff), 1)

            diff = 0.25

            for i in [1, 2, 3, 4]:
                actual_diff = diff * i
                actual_x_diff = height * actual_diff
                actual_y_diff = width * actual_diff
                x = top - actual_x_diff
                y = left - actual_y_diff

                for x_move in range(3):
                    for y_move in range(3):
                        next_x = x + x_move * actual_x_diff
                        next_y = y + y_move * actual_y_diff
                        if (0 <= next_x < next_x + height <= filas and
                                0 <= next_y < next_y + width <= columnas):
                            yield ((next_x, next_y),
                                   (next_x + height, next_y + width))


class BusquedaPorFramesSolapados(object):
    @staticmethod
    def iterate_frame_boxes(obj_limits, scene_limits, obj_mult=2, step=2):
        """
        La idea de la segmentación es tomar un cuadrante de cierto tamaño e
        ir recortando la imagen de manera que cada recorte se solape con el
        anterior en la mitad del tamaño del cuadrante.
        """
        obj_width = (obj_limits.max_x - obj_limits.min_x) * obj_mult
        obj_height = (obj_limits.max_y - obj_limits.min_y) * obj_mult

        scene_min_col = scene_limits.min_x
        scene_max_col = scene_limits.max_x
        scene_min_row = scene_limits.min_y
        scene_max_row = scene_limits.max_y

        ###################################################################
        # Armo una lista de tuplas con los limites del filtro para el alto
        ###################################################################
        paso_alto_frame = obj_height / step
        cant_pasos_alto = max(
            int((scene_max_row - scene_min_row) / paso_alto_frame),
            step
        )

        # armo una lista con intervalos de tamaño ancho_objeto / step
        alto_linspace = np.linspace(
            scene_min_row,
            scene_max_row,
            cant_pasos_alto,
        )

        # Armo una lista de tuplas que marquen los puntos a "barrer", en donde
        # la diferencia entre los valores de cada tupla sea del tamaño del
        # objeto y la diferencia entre el primer valor de una tupla y el
        # primero de la siguiente sea la mitad del tamaño del objeto
        if len(alto_linspace) == step:
            alto_limites_inferiores = alto_linspace[:-1 * (step-1)]
            alto_limites_superiores = alto_linspace[step-1:]
        else:  # len(alto_linspace) > 2
            alto_limites_inferiores = alto_linspace[:-1*step]
            alto_limites_superiores = alto_linspace[step:]
        alto_limites = zip(alto_limites_inferiores, alto_limites_superiores)

        ####################################################################
        # Armo una lista de tuplas con los limites del filtro para el ancho
        ####################################################################
        paso_ancho_frame = obj_width / step
        cant_pasos_ancho = max(
            int((scene_max_col - scene_min_col) / paso_ancho_frame),
            step
        )

        # armo una lista con intervalos de tamaño ancho_objeto / step
        ancho_linspace = np.linspace(
            scene_min_col,
            scene_max_col,
            cant_pasos_ancho,
        )

        # Armo una lista de tuplas que marquen los puntos a "barrer", en donde
        # la diferencia entre los valores de cada tupla sea del tamaño del
        # objeto y la diferencia entre el primer valor de una tupla y el
        # primero de la siguiente sea la mitad del tamaño del objeto
        if len(ancho_linspace) == step:
            ancho_limites_inferiores = ancho_linspace[:-1 * (step-1)]
            ancho_limites_superiores = ancho_linspace[step-1:]
        else:
            ancho_limites_inferiores = ancho_linspace[:-1 * step]
            ancho_limites_superiores = ancho_linspace[step:]
        ancho_limites = zip(ancho_limites_inferiores, ancho_limites_superiores)

        for row_low, row_up in alto_limites:
            for col_low, col_up in ancho_limites:
                yield {
                    'min_x': col_low,
                    'max_x': col_up,
                    'min_y': row_low,
                    'max_y': row_up,
                }