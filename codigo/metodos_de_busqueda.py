#coding=utf-8

from __future__ import unicode_literals, division

import numpy as np

class BusquedaAlrededor(object):
    def get_positions_and_framesizes(self, topleft, bottomright,
                                     template_filas, template_columnas,
                                     filas, columnas):
        top, left = topleft
        bottom, right = bottomright

        height = template_filas
        width = template_columnas

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
                                     template_filas, template_columnas,
                                     filas, columnas):
        # Como manejo porcentajes, puede ser que repita frames. Los acumulo para
        # descartar los repetidos
        puntos_ya_visitados = set()

        top, left = topleft

        original_height = template_filas
        original_width = template_columnas

        for tam_diff in [0.75, 1, 1.25]:

            height = max(int(original_height * tam_diff), 2)
            width = max(int(original_width * tam_diff), 2)

            # Voy a ir desplazando de a diff pixeles
            diff = max(int(0.25 * min(height, width)), 1)

            for i in [1, 2, 3, 4]:
                # Diferencia entre la esquina superior izquierda inicial y la
                # que estoy explorando ahora
                starting_diff = diff * i

                # Nueva coordenada para la esquina superior izquierda en la
                # busqueda
                x = top - starting_diff
                y = left - starting_diff

                # Calculando cuantas veces entra el cuadrante en la zona de
                # busqueda si me desplazo de a diff pixeles
                x_most_bottom = top + height + starting_diff
                y_most_right = left + width + starting_diff
                cant_repes_x = int((x_most_bottom - (x + height) + 1) / diff)
                cant_repes_y = int((y_most_right - (y + width) + 1) / diff)

                for x_move in range(cant_repes_x):
                    for y_move in range(cant_repes_y):
                        next_x = int(x + x_move * diff)
                        next_y = int(y + y_move * diff)
                        if (0 <= next_x < next_x + height <= filas and
                                0 <= next_y < next_y + width <= columnas):
                            new_topleft = (next_x, next_y)
                            new_bottomright = (next_x + height, next_y + width)

                            if not (new_topleft, new_bottomright) in puntos_ya_visitados:
                                puntos_ya_visitados.add((new_topleft, new_bottomright))
                                yield (new_topleft, new_bottomright)


class BusquedaCambiandoSizePeroMismoCentro(object):
    def get_positions_and_framesizes(self, topleft, bottomright,
                                     template_filas, template_columnas,
                                     filas, columnas):
        original_height = template_filas
        original_width = template_columnas

        for tam_diff in [0.5, 1, 1.5, 2]:

            height = max(int(original_height * tam_diff), 2)
            diff_height = int((height - original_height) / 2)
            width = max(int(original_width * tam_diff), 2)
            diff_width = int((width - original_width) / 2)

            new_top = topleft[0] - diff_height
            new_bottom = topleft[0] + template_filas + diff_height

            new_left = topleft[1] - diff_width
            new_right = topleft[1] + template_columnas + diff_width

            if (0 <= new_top < new_bottom <= filas and
                    0 <= new_left < new_right <= columnas):
                yield (new_top, new_left), (new_bottom, new_right)


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