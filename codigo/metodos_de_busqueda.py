#coding=utf-8

from __future__ import unicode_literals, division

import numpy as np
import math

class BusquedaEnEspiral(object):
    def get_positions_and_framesizes(self, ultima_ubicacion, tam_region, filas, columnas):
        x, y = ultima_ubicacion

        sum_x = 2
        sum_y = 2
        for j in range(30):
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
    def iterate_frame_boxes(scene_min_col, scene_max_col, scene_min_row,
                            scene_max_row, obj_width, obj_height, step=2):
        """
        La idea de la segmentación es tomar un cuadrante de cierto tamaño e
        ir recortando la imagen de manera que cada recorte se solape con el
        anterior en la mitad del tamaño del cuadrante.
        """
        ###################################################################
        # Armo una lista de tuplas con los limites del filtro para el alto
        ###################################################################
        paso_alto_frame = obj_height / step
        cant_pasos_alto = (scene_max_row - scene_min_row) / paso_alto_frame

        # armo una lista con intervalos de tamaño ancho_objeto / step
        alto_linspace = np.linspace(
            scene_min_row,
            scene_max_row,
            int(cant_pasos_alto),
        )

        # armo 2 listas y las reparto de manera tal que tengan el mismo tamaño
        # pero que al hacer zip, la diferencia entre lista_1[i] y lista_2[i]
        # sea de ancho_objeto / step
        int_obj_height = math.ceil(obj_height/paso_alto_frame)
        alto_limites_inferiores = alto_linspace[:-1*int_obj_height]
        alto_limites_superiores = alto_linspace[int_obj_height:]
        alto_limites = zip(alto_limites_inferiores, alto_limites_superiores)

        ####################################################################
        # Armo una lista de tuplas con los limites del filtro para el ancho
        ####################################################################
        paso_ancho_frame = obj_width / step
        cant_pasos_ancho = (scene_max_row - scene_min_row) / paso_ancho_frame

        # armo una lista con intervalos de tamaño ancho_objeto / step
        ancho_linspace = np.linspace(
            scene_min_col,
            scene_max_col,
            int(cant_pasos_ancho),
        )

        # armo 2 listas y las reparto de manera tal que tengan el mismo tamaño
        # pero que al hacer zip, la diferencia entre lista_1[i] y lista_2[i]
        # sea de ancho_objeto / step
        int_obj_width = math.ceil(obj_width/paso_ancho_frame)
        ancho_limites_inferiores = ancho_linspace[:-1 * int_obj_width]
        ancho_limites_superiores = ancho_linspace[int_obj_width:]
        ancho_limites = zip(ancho_limites_inferiores, ancho_limites_superiores)

        # #######################################################
        # Algunos calculos a mano para corroborar que ande bien
        ########################################################
        obj_height = alto_limites[0][1] - alto_limites[0][0]
        obj_width = ancho_limites[0][1] - ancho_limites[0][0]
        scene_width = scene_max_col - scene_min_col
        frames_ancho = (scene_width / obj_width) * 2 - 1
        print "Frames a lo ancho POSTA:", round(frames_ancho)

        scene_height = scene_max_row - scene_min_row
        frames_alto = (scene_height / obj_height) * 2 - 1
        print "Frames a lo alto POSTA:", round(frames_alto)

        print "Frames totales supuestos POSTA:", frames_ancho * frames_alto
        ########################################################


        for row_low, row_up in alto_limites:
            for col_low, col_up in ancho_limites:
                yield {
                    'min_x': col_low,
                    'max_x': col_up,
                    'min_y': row_low,
                    'max_y': row_up,
                }