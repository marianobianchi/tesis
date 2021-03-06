#coding=utf-8

from __future__ import (unicode_literals, division)

import time

import numpy as np
import cv2
import scipy.io

from cpp.common import get_point, points, get_min_max, compute_centroid


class GroundTruthFrame(object):
    def __init__(self, nframe, obj_data_list):
        data = {}
        for obj_data in obj_data_list:
            obj_name = obj_data[0].flatten()[0]
            obj_num = obj_data[1].flatten()[0]
            obj_top = obj_data[2].flatten()[0] - 1
            obj_bottom = obj_data[3].flatten()[0] - 1
            obj_left = obj_data[4].flatten()[0] - 1
            obj_right = obj_data[5].flatten()[0] - 1

            key = '{objname}_{objnum}'.format(objname=obj_name, objnum=obj_num)
            value = ((obj_top, obj_left), (obj_bottom, obj_right))
            data[key] = value

        self._data = data
        self._nframe = nframe

    def _obj_name_num(self, objname_or_namenum, objnum=None):
        if objnum is None:
            namenum = objname_or_namenum
        else:
            namenum = '{na}_{nu}'.format(na=objname_or_namenum, nu=objnum)

        return namenum

    def appear(self, objname_or_namenum, objnum=None):
        key = self._obj_name_num(objname_or_namenum, objnum)
        return key in self._data

    def nframe(self):
        return self._nframe

    def bounding_box(self, objname_or_namenum, objnum=None):
        key = self._obj_name_num(objname_or_namenum, objnum)
        return self._data[key]

    def __iter__(self):
        """
        Returns a tuple with objname_objnum, topleft, bottomright
        """
        for key, value in self._data.items():
            yield key, value[0], value[1]


class GroundTruthScene(object):
    def __init__(self, matfile_path):
        self._frames_data = scipy.io.loadmat(matfile_path)['bboxes'][0]

    def frames(self):
        """
        Returns the number of frames in the scene
        """
        return len(self._frames_data)

    def __iter__(self):
        """
        Iterate through frames of the scene
        """
        for i in range(self.frames()):
            yield GroundTruthFrame(i + 1, self._frames_data[i][0])


def list_to_interval_list(l):
    assert len(l) % 2 == 0, "La lista debe tener longitud par"
    intervals = []
    while l:
        t = tuple(l[:2])
        intervals.append(t)
        l = l[2:]
    return intervals


def scene_summary(matfile_path):
    gts = GroundTruthScene(matfile_path)
    obj_frames = {}
    for frame in gts:
        # Llevo la cuenta de los objetos que estan en el frame
        objs_in_frame = set()
        for obj, topleft, bottomright in frame:
            objs_in_frame.add(obj)

            if obj not in obj_frames:
                obj_frames[obj] = (False, [])  # False = no estaba en el frame anterior

            if not obj_frames[obj][0]:  # Si no estaba en el frame anterior
                l = obj_frames[obj][1]
                l.append(frame.nframe())
                obj_frames[obj] = (True, l)

        objs_in_other_frames = set(obj_frames.keys()) - objs_in_frame
        for obj in objs_in_other_frames:
            if obj_frames[obj][0]:  # Si estaba en el frame anterior
                l = obj_frames[obj][1]
                l.append(frame.nframe() - 1)
                assert len(l) % 2 == 0, "Hubo un error al armar los intervalos"
                obj_frames[obj] = (False, l)

    for obj in obj_frames:
        if obj_frames[obj][0]:  # Si estaba en el ultimo frame anterior
            l = obj_frames[obj][1]
            l.append(gts.frames())
            assert len(l) % 2 == 0, "Hubo un error al armar los intervalos finales"
            obj_frames[obj] = (False, l)

    # Imprimo valores
    cant_frames = gts.frames()
    print 'Frames =', cant_frames
    print ('#A sacar: Negativo == sacar frames donde no aparece. '
           'Positivo == sacar frames donde aparece')
    print ''.join(
        ['Objeto'.center(15),
         'Intervalos'.center(35),
         '#Aparicion'.center(15),
         '#A sacar'.center(15)]
    )

    for obj, (b, l) in obj_frames.items():
        l = list_to_interval_list(l)
        cant_apar = sum([j - i + 1 for i, j in l])
        a_sacar = 2 * cant_apar - cant_frames

        print ''.join(
            [obj.rjust(15),
             '{l}'.format(l=l).rjust(35),
             '{n}'.format(n=cant_apar).rjust(15),
             '{p}'.format(p=a_sacar).rjust(15)]
        )


def dibujar_cuadrado(img, topleft, bottomright, color=(0, 0, 0)):
    topleft = (int(topleft[0]), int(topleft[1]))
    bottomright = (int(bottomright[0]), int(bottomright[1]))
    cv2.rectangle(
        img,
        tuple(reversed(topleft)),
        tuple(reversed(bottomright)),
        color,
        3
    )
    return img


def pegar_fotos(foto1, foto2):
    """
    Pega dos fotos, una a la izquierda y la otra a la derecha y devuelve la
    foto resultante.

    Las fotos deben tener el mismo alto.
    """
    alto = len(foto1)
    ancho = len(foto1[0])

    foto3 = np.zeros((alto, ancho * 2, 3), dtype=foto1.dtype)
    foto3[:, :ancho] = foto1
    foto3[:, ancho:] = foto2
    return foto3


def dibujar_matching(train_frame, query_frame, kp_train, kp_query, good):

    good_train_kps_idx = [m[0].trainIdx for m in good]
    good_query_kps_idx = [m[0].queryIdx for m in good]

    good_train_kp = [kp_train[i] for i in good_train_kps_idx]
    good_query_kp = [kp_query[i] for i in good_query_kps_idx]

    train_with_kp = cv2.drawKeypoints(train_frame, good_train_kp)
    query_with_kp = cv2.drawKeypoints(query_frame, good_query_kp)

    frames_pegados = pegar_fotos(train_with_kp, query_with_kp)

    ancho = len(train_frame[0])

    for train_kp, query_kp in zip(good_train_kp, good_query_kp):
        pt1 = (int(train_kp.pt[0]+0.5), int(train_kp.pt[1]+0.5))
        pt2 = (int(train_kp.pt[0]+0.5) + ancho, int(query_kp.pt[1]+0.5))
        cv2.line(frames_pegados, pt1, pt2, (0,0,0))

    cv2.imshow('Frames pegados', frames_pegados)
    cv2.resizeWindow('Frames pegados', 200, 100)
    cv2.waitKey()


def keypoint_matching_entre_dos_imagenes():
    """
    ¿Que hago con los descriptores?
    """
    train_frame = cv2.imread('videos/pelotita_naranja_webcam/pelota_frame1.jpg')
    query_frame = cv2.imread('videos/pelotita_naranja_webcam/pelota_frame2.jpg')

    surf = cv2.SURF()

    # Deteccion de keypoints y generacion de descriptores
    kp_train, des_train = surf.detectAndCompute(train_frame, None)
    kp_query, des_query = surf.detectAndCompute(query_frame, None)


    # Calcular matching entre descriptores
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des_train,des_query, k=2)

    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])


    dibujar_matching(train_frame, query_frame, kp_train, kp_query, good)


def from_flat_to_cloud(imR, imC, depth):

    # return something invalid if depth is zero
    if depth == 0:
        return (-10000.0, -10000.0)

    # cloud coordinates
    cloud_row = float(imR)
    cloud_col = float(imC)

    # images size is 640 COLS x 480 ROWS
    rows_center = 240
    cols_center = 320

    # focal distance
    constant = 570.3

    # move the coordinate (0,0) from the top-left corner to the center of the plane
    cloud_row = cloud_row - rows_center
    cloud_col = cloud_col - cols_center

    # calculate cloud
    cloud_row = cloud_row * depth / constant / 1000
    cloud_col = cloud_col * depth / constant / 1000

    return (cloud_row, cloud_col)


def from_cloud_to_flat(cloud_row, cloud_col, float_depth):
    # images size is 640 COLS x 480 ROWS
    rows_center = 240
    cols_center = 320

    # focal distance
    constant = 570.3

    # inverse of cloud calculation
    im_row = int(cloud_row / float_depth * constant)
    im_col = int(cloud_col / float_depth * constant)

    im_row += rows_center
    im_col += cols_center

    return im_row, im_col


def from_cloud_to_flat_limits(cloud):
    top = 1e10
    left = 1e10
    bottom = 0
    right = 0
    for i in range(points(cloud)):
        point_xyz = get_point(cloud, i)
        if point_xyz.z > 0:
            point_flat = from_cloud_to_flat(point_xyz.y, point_xyz.x, point_xyz.z)

            top = min(point_flat[0], top)
            bottom = max(point_flat[0], bottom)

            left = min(point_flat[1], left)
            right = max(point_flat[1], right)

    #### NUEVO Y MAS CORTO METODO
    # minmax = get_min_max(cloud)
    # topleft_1 = from_cloud_to_flat(minmax.min_y, minmax.min_x, minmax.min_z)
    # topleft_2 = from_cloud_to_flat(minmax.min_y, minmax.min_x, minmax.max_z)
    # topright_1 = from_cloud_to_flat(minmax.min_y, minmax.max_x, minmax.min_z)
    # topright_2 = from_cloud_to_flat(minmax.min_y, minmax.max_x, minmax.max_z)
    #
    # bottomleft_1 = from_cloud_to_flat(minmax.max_y, minmax.min_x, minmax.min_z)
    # bottomleft_2 = from_cloud_to_flat(minmax.max_y, minmax.min_x, minmax.max_z)
    # bottomright_1 = from_cloud_to_flat(minmax.max_y, minmax.max_x, minmax.min_z)
    # bottomright_2 = from_cloud_to_flat(minmax.max_y, minmax.max_x, minmax.max_z)
    #
    # top = min(topleft_1[0], topleft_2[0], topright_1[0], topright_2[0])
    # bottom = max(bottomleft_1[0], bottomleft_2[0],
    #              bottomright_1[0], bottomright_2[0])
    #
    # left = min(topleft_1[1], topleft_2[1], bottomleft_1[1], bottomleft_2[1])
    # right = max(topright_1[1], topright_2[1],
    #             bottomright_1[1], bottomright_2[1])
    ####

    return (top, left), (bottom, right)


def from_flat_to_cloud_limits(topleft, bottomright, depth_img):
    """
     * Dada una imagen en profundidad y la ubicación de un objeto
     * en coordenadas sobre la imagen en profundidad, obtengo la nube
     * de puntos correspondiente a esas coordenadas y me quedo
     * unicamente con los valores máximos y mínimos de las coordenadas
     * "x" e "y" de dicha nube
    """

    # Inicializo los limites. Los inicializo al revés para que funcione bien al comparar
    r_top_limit    =  10.0
    r_bottom_limit = -10.0
    c_left_limit   =  10.0
    c_right_limit  = -10.0

    # TODO: paralelizar estos "for" usando funciones de OpenCV o PCL o lo que sea
    # Hint: que "from_flat_to_cloud" reciba una matriz (la imagen) directamente

    for r in range(topleft[0], bottomright[0] + 1):
        for c in range(topleft[1], bottomright[1] + 1):
            depth = depth_img[r][c]

            cloudRC = from_flat_to_cloud(r, c, depth)

            if cloudRC[0] != -10000 and cloudRC[0] < r_top_limit:
                r_top_limit = cloudRC[0]

            if cloudRC[0] != -10000 and cloudRC[0] > r_bottom_limit:
                r_bottom_limit = cloudRC[0]

            if cloudRC[1] != -10000 and cloudRC[1] < c_left_limit:
                c_left_limit = cloudRC[1]

            if cloudRC[1] != -10000 and cloudRC[1] > c_right_limit:
                c_right_limit = cloudRC[1]

    top_bottom = (r_top_limit, r_bottom_limit)
    left_right = (c_left_limit, c_right_limit)
    res = (top_bottom, left_right)
    return res


#########
# TESTS #
#########
def test_flat_and_cloud_conversion():
    depth = 1300

    for i in range(0, 480):
        for j in range(0, 640):
            cloudXY = from_flat_to_cloud(i, j, depth)
            flatXY = from_cloud_to_flat(cloudXY[0], cloudXY[1], depth / 1000)

            if not ((flatXY[0] -1) <= i <= (flatXY[0] +1)):
                print flatXY[0] -1
                print i
                print flatXY[0] +1
                raise Exception('Falla la fila')
            if not ((flatXY[1] -1) <= j <= (flatXY[1] +1)):
                print flatXY[1] -1
                print j
                print flatXY[1] +1
                raise Exception('Falla la columna')


def measure_time(func):
    """
    Decorador que imprime en pantalla el tiempo que tarda en ejecutarse
    una funcion
    """
    def inner(*args, **kwargs):
        start_time = time.time()
        val = func(*args, **kwargs)
        end_time = time.time()
        print func.__name__, "took", end_time - start_time, "to finish"
        return val
    return inner


class Timer(object):
    def __init__(self, func_name):
        self.func_name = func_name

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        print self.func_name, "took", self.end - self.start, "to finish"


class AdaptSearchArea(object):
    """
    Adapta el area de busqueda dependiendo de la velocidad del objeto. Las
    primeras MUESTRAS veces devuelve la mitad del tamaño del objeto para cada
    eje. (que termina siendo un volumen 8 veces mayor)
    Una vez que tiene estadística suficiente calcula un movimiento ponderado.

    Lo que devuelve es cuanto hay que "estirar" cada borde del bounding-box.
    """
    def __init__(self, muestras=4):
        self.centroids = []
        self.default_x_distance = None
        self.default_y_distance = None
        self.default_z_distance = None
        self.muestras = muestras

    def set_default_distances(self, obj_model):
        if self.default_x_distance is None:
            minmax = get_min_max(obj_model)
            self.default_x_distance = (minmax.max_x - minmax.min_x) / 2
            self.default_y_distance = (minmax.max_y - minmax.min_y) / 2
            self.default_z_distance = (minmax.max_z - minmax.min_z) / 2

    def save_centroid(self, x_min, y_min, z_min, x_max, y_max, z_max):
        centroid = compute_centroid(x_min, y_min, z_min, x_max, y_max, z_max)
        self.centroids.append(centroid)
        return centroid

    def estimate_distance(self, axe):
        if len(self.centroids) > self.muestras:
            centroids = self.centroids[-1 * self.muestras:]
            if axe == 'x':
                centroids = [c.x for c in centroids]
            elif axe == 'y':
                centroids = [c.y for c in centroids]
            elif axe == 'z':
                centroids = [c.z for c in centroids]
            else:
                raise Exception('No existe el eje indicado')

            cent_a = centroids[:-1]
            cent_b = centroids[1:]
            cent = zip(cent_a, cent_b)
            distances = [abs(p1 - p2) for p1, p2 in cent]

            # TODO: adaptar los porcentajes al valor de self.muestras
            porcentajes = [0.1, 0.2, 0.7]
            # [0.25, 0.25, 0.5]


            return sum([d*p for d, p in zip(distances, porcentajes)])
        else:
            if axe == 'x':
                val = self.default_x_distance
            elif axe == 'y':
                val = self.default_y_distance
            elif axe == 'z':
                val = self.default_z_distance
            else:
                raise Exception('No existe el eje indicado')

            return val


class FixedSearchArea(AdaptSearchArea):
    def __init__(self, obj_size_times=2):
        self.centroids = []
        self.default_x_distance = None
        self.default_y_distance = None
        self.default_z_distance = None

        # Multiplo del tamaño del objeto que tiene cada eje del area de busqueda
        self.obj_size_times = max(obj_size_times, 1) - 1

    def estimate_distance(self, axe):
        if axe == 'x':
            val = self.default_x_distance
        elif axe == 'y':
            val = self.default_y_distance
        elif axe == 'z':
            val = self.default_z_distance
        else:
            raise Exception('No existe el eje indicado')

        obj_tam = val * 2
        eje_tam = obj_tam + obj_tam * self.obj_size_times
        # print "    tam. objeto en el eje:", obj_tam
        # print "    tam. busqueda en el eje:", eje_tam
        # print "    relacion-tamaño:", eje_tam * 1.0 / obj_tam

        return val * self.obj_size_times


class AdaptLeafRatio(object):
    """
    Adapts the leaf radio used to take object points from scene
    """
    def __init__(self, first_leaf=0.005, min_ratio=0.002):
        self.found_points = []
        self.ratios = [first_leaf]
        self.min_ratio = min_ratio

    def was_started(self):
        return len(self.found_points) > 0

    def reset(self):
        self.found_points = self.found_points[:1]
        self.ratios = self.ratios[:1]

    def set_first_values(self, model_points):
        self.found_points = [model_points]

    def set_found_points(self, points):
        self.found_points.append(points)
        self._set_leaf_ratio()

    def _set_leaf_ratio(self):
        last_ratio = self.ratios[-1]
        new_ratio = last_ratio
        if len(self.ratios) > 1:
            model_points = self.found_points[0]
            last_points = self.found_points[-1]

            if 70 <= (last_points / model_points * 100) < 100:
                # Mantengo si se perdieron puntos, bajo si se ganaron
                diff_points = self.found_points[-1] - self.found_points[-2]
                if diff_points > 0:
                    new_ratio -= 0.001
            elif (last_points / model_points * 100) < 70:
                diff_ratio = self.ratios[-1] - self.ratios[-2]
                diff_points = self.found_points[-1] - self.found_points[-2]

                # Si subieron el radio y los puntos
                if diff_ratio > 0 and diff_points > 0:
                    new_ratio = last_ratio  # Mantengo
                # Si subio el radio y bajaron los puntos
                elif diff_ratio > 0 and diff_points <= 0:
                    new_ratio = last_ratio + 0.001
                # Si bajo el radio y subieron los puntos
                elif diff_ratio < 0 and diff_points > 0:
                    new_ratio = last_ratio  # Mantengo
                # Si bajaron el radio y los puntos
                elif diff_ratio < 0 and diff_points <= 0:
                    new_ratio = last_ratio + 0.001
                # Si se mantuvo el radio y subieron los puntos
                elif diff_ratio == 0 and diff_points > 0:
                    new_ratio = last_ratio  # Mantengo
                # Si se mantuvo el radio y bajaron los puntos
                else:
                    new_ratio = last_ratio + 0.001

            else:  # (last_points / model_points * 100) >= 100
                new_ratio = last_ratio - 0.002

        new_ratio = max(self.min_ratio, new_ratio)
        self.ratios.append(new_ratio)

    def leaf_ratio(self):
        return self.ratios[-1]
