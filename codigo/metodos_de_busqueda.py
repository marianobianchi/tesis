#coding=utf-8

from __future__ import unicode_literals, division

import numpy as np

class BusquedaEnEspiral(object):
    def get_positions_and_framesizes(self, ultima_ubicacion, tam_region, filas, columnas):
        top, left = ultima_ubicacion
        bottom, right = top + tam_region, left + tam_region

        height = bottom - top
        width = right - left

        diff = 0.25
        cant_moves = int(1 / diff) + 1

        # Voy a ir moviendo el cuadrante de a "diff" para cada eje y aumentaré
        # hasta 4 * diff veces la zona de busqueda
        for i in range(1, 3):
            actual_diff = diff * i
            actual_x_diff = height * actual_diff
            actual_y_diff = width * actual_diff
            x = top - actual_x_diff
            y = left - actual_y_diff

            for x_move in range(cant_moves):
                for y_move in range(cant_moves):
                    next_x = x + x_move * actual_x_diff
                    next_y = y + y_move * actual_y_diff
                    if (0 <= next_x < next_x + tam_region <= filas and
                            0 <= next_y < next_y + tam_region <= columnas):
                        yield (next_x, next_y, tam_region)


class BusquedaEnEspiralCambiandoFrameSize(object):
    def get_positions_and_framesizes(self, ultima_ubicacion, tam_region_inicial, filas, columnas):
        mitad_region = tam_region_inicial / 2
        cuarto_de_region = tam_region_inicial / 4
        for tam_region in [(mitad_region + (i*cuarto_de_region)) for i in range(5)]:
            x, y = ultima_ubicacion

            sum_x = 2
            sum_y = 2
            for j in range(10):
                # Hago 2 busquedas por cada nuevo X
                x += sum_x/2
                if (0 <= x <= (x+tam_region) <= filas) and (0 <= y <= (y+tam_region) <= columnas):
                    yield (x, y, tam_region)

                x += sum_x/2
                if (0 <= x <= (x+tam_region) <= filas) and (0 <= y <= (y+tam_region) <= columnas):
                    yield (x, y, tam_region)

                sum_x *= -2

                # Hago 2 busquedas por cada nuevo Y
                y += sum_y/2
                if (0 <= x <= (x+tam_region) <= filas) and (0 <= y <= (y+tam_region) <= columnas):
                    yield (x, y, tam_region)

                y += sum_y/2
                if (0 <= x <= (x+tam_region) <= filas) and (0 <= y <= (y+tam_region) <= columnas):
                    yield (x, y, tam_region)

                sum_y *= -2


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