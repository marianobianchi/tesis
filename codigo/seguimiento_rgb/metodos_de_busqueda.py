#!/usr/bin/python
# -*- coding: utf-8 -*-
#Con esto, todos los strings literales son unicode (no hace falta poner u'algo')
from __future__ import unicode_literals


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