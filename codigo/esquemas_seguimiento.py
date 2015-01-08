#coding=utf-8

from __future__ import unicode_literals, print_function

import os
import re

from cpp.common import save_pcd


class FollowingScheme(object):

    def __init__(self, img_provider, obj_follower, show_following):
        self.img_provider = img_provider
        self.obj_follower = obj_follower
        self.show_following = show_following

    def run(self):
        #########################
        # Etapa de entrenamiento
        #########################
        self.obj_follower.train()

        ######################
        # Etapa de detección
        ######################
        fue_exitoso, topleft, bottomright = (
            self.obj_follower.detect()
        )

        # Muestro el seguimiento para hacer pruebas
        self.show_following.run(
            img_provider=self.img_provider,
            topleft=topleft,
            bottomright=bottomright,
            fue_exitoso=fue_exitoso,
            es_deteccion=True,
            frenar=True,
        )

        #######################
        # Etapa de seguimiento
        #######################

        # Adelanto un frame
        self.img_provider.next()

        es_deteccion = True

        while self.img_provider.have_images():

            if fue_exitoso:
                fue_exitoso, topleft, bottomright = (
                    self.obj_follower.follow(es_deteccion)
                )
                if fue_exitoso:
                    es_deteccion = False

            if not fue_exitoso:
                es_deteccion = True
                fue_exitoso, topleft, bottomright = (
                    self.obj_follower.detect()
                )
            # Muestro el seguimiento
            self.show_following.run(
                img_provider=self.img_provider,
                topleft=topleft,
                bottomright=bottomright,
                fue_exitoso=fue_exitoso,
                es_deteccion=es_deteccion,
                frenar=True,
            )

            # Adelanto un frame
            self.img_provider.next()

        self.show_following.close()


class FollowingSchemeSavingDataPCD(FollowingScheme):
    """
    Guarda las pruebas en carpetas consecutivas llamadas prueba_###
    """
    def __init__(self, img_provider, obj_follower, path):
        super(FollowingSchemeSavingDataPCD, self).__init__(
            img_provider,
            obj_follower,
            None,
        )

        # para no pisar nunca los resultados voy a ir creando carpetas sucesivas
        # llamadas prueba_###
        self.results_path = os.path.join(path, '{s}_{sn}/{o}_{on}/')
        self.results_path = self.results_path.format(
            s=self.img_provider.scene,
            sn=self.img_provider.scene_number,
            o=self.img_provider.obj,
            on=self.img_provider.obj_number,
        )

        os.listdir(self.results_path)
        rc = re.compile('prueba_(?P<number>\d{3})')
        pruebas_dirs = [l for l in os.listdir(self.results_path) if rc.match(l)]
        pruebas_dirs = pruebas_dirs if pruebas_dirs else ['prueba_000']
        pruebas_dirs.sort()
        last_test_number = int(rc.match(pruebas_dirs[-1]).groupdict()['number'])
        new_folder_name = 'prueba_{n:03d}'.format(n=last_test_number+1)
        self.results_path = os.path.join(self.results_path, new_folder_name)

        if not os.path.isdir(self.results_path):
            os.makedirs(self.results_path)
        self.file = open(os.path.join(self.results_path, 'results.txt'), 'w')

        self.write_parameter_values()


    def write_parameter_values(self):
        # Guardo los valores de los parametros
        ap_defaults = self.obj_follower.detector._ap_defaults
        self.file.write(b'ap_leaf={v}\n'.format(v=ap_defaults.leaf))
        self.file.write(b'ap_max_ransac_iters={v}\n'.format(v=ap_defaults.max_ransac_iters))
        self.file.write(b'ap_points_to_sample={v}\n'.format(v=ap_defaults.points_to_sample))
        self.file.write(b'ap_nearest_features_used={v}\n'.format(v=ap_defaults.nearest_features_used))
        self.file.write(b'ap_simil_threshold={v}\n'.format(v=ap_defaults.simil_threshold))
        self.file.write(b'ap_inlier_threshold={v}\n'.format(v=ap_defaults.inlier_threshold))
        self.file.write(b'ap_inlier_fraction={v}\n'.format(v=ap_defaults.inlier_fraction))

        icp_defaults = self.obj_follower.detector._icp_defaults
        self.file.write(b'det_euc_fit={v}\n'.format(v=icp_defaults.euc_fit))
        self.file.write(b'det_max_corr_dist={v}\n'.format(v=icp_defaults.max_corr_dist))
        self.file.write(b'det_max_iter={v}\n'.format(v=icp_defaults.max_iter))
        self.file.write(b'det_transf_epsilon={v}\n'.format(v=icp_defaults.transf_epsilon))

        icp_defaults = self.obj_follower.finder._icp_defaults
        self.file.write(b'seg_euc_fit={v}\n'.format(v=icp_defaults.euc_fit))
        self.file.write(b'seg_max_corr_dist={v}\n'.format(v=icp_defaults.max_corr_dist))
        self.file.write(b'seg_max_iter={v}\n'.format(v=icp_defaults.max_iter))
        self.file.write(b'seg_transf_epsilon={v}\n'.format(v=icp_defaults.transf_epsilon))
        self.file.write(b'RESULTS_SECTION\n')
        self.file.flush()

    def __del__(self):
        self.file.close()

    def run(self):
        # ########################
        # Etapa de entrenamiento
        #########################
        self.obj_follower.train()

        ######################
        # Etapa de detección
        ######################
        print("Detectando en imagen {i}".format(
            i=self.img_provider.next_frame_number
        ))
        es_deteccion = True

        fue_exitoso, topleft, bottomright = (
            self.obj_follower.detect()
        )

        self.save_result(0, fue_exitoso, topleft, bottomright)

        #######################
        # Etapa de seguimiento
        #######################

        # Adelanto un frame
        self.img_provider.next()

        es_deteccion = True

        while self.img_provider.have_images():

            if fue_exitoso:
                print(
                    "Buscando en imagen {i}".format(
                        i=self.img_provider.next_frame_number
                    ),
                    ''
                )
                fue_exitoso, topleft, bottomright = (
                    self.obj_follower.follow(es_deteccion)
                )
                if fue_exitoso:
                    es_deteccion = False

            if not fue_exitoso:

                print("{p}Detectando en imagen {i}".format(
                    p='...MISS... ' if not es_deteccion else '',
                    i=self.img_provider.next_frame_number,
                ))
                fue_exitoso, topleft, bottomright = (
                    self.obj_follower.detect()
                )
                es_deteccion = True

            self.save_result(
                0 if es_deteccion else 1,
                fue_exitoso,
                topleft,
                bottomright
            )

            # Adelanto un frame
            self.img_provider.next()

    def save_result(self, method, fue_exitoso, topleft, bottomright):
        """
        Formato para guardar:

        frame_number;exito;metodo;fila_sup;col_izq;fila_inf;col_der

        donde:
        metodo = 0 si deteccion, 1 si seguimiento
        exito = 0 si fallo, 1 si funciono
        """
        nframe = self.img_provider.next_frame_number
        exito = 1 if fue_exitoso else 0

        values = [nframe, exito, method, int(topleft[0]), int(topleft[1]),
                  int(bottomright[0]), int(bottomright[1])]

        self.file.write(b';'.join([str(o) for o in values]))
        self.file.write(b'\n')
        self.file.flush()

        # Guardo las nubes de puntos
        if fue_exitoso and 'object_cloud' in self.obj_follower.descriptors():
            # Guardo el objeto cuyos puntos pertenecen a la escena
            pcd = self.obj_follower.descriptors()['object_cloud']
            pcd_filename = 'obj_found_scenepoints_frame_{i:03}.pcd'.format(i=nframe)
            filename = os.path.join(self.results_path, pcd_filename)
            save_pcd(pcd, str(filename))

            # Guardo el objeto alineado, cuyos puntos son los del frame anterior
            # pero alineado
            pcd = self.obj_follower.descriptors()['detected_cloud']
            pcd_filename = 'obj_found_alignedpoints_frame_{i:03}.pcd'.format(i=nframe)
            filename = os.path.join(self.results_path, pcd_filename)
            save_pcd(pcd, str(filename))


class FollowingSquemaExploringParameterPCD(FollowingSchemeSavingDataPCD):
    """
    Guarda las pruebas en carpetas con el nombre del parametro y el valor
    que se estan explorando
    """

    def __init__(self, img_provider, obj_follower, path, param_name, param_val):
        self.img_provider = img_provider
        self.obj_follower = obj_follower
        self.show_following = None

        self.results_path = os.path.join(path, '{s}_{sn}/{o}_{on}/{p}/{v}')
        self.results_path = self.results_path.format(
            s=self.img_provider.scene,
            sn=self.img_provider.scene_number,
            o=self.img_provider.obj,
            on=self.img_provider.obj_number,
            p=param_name,
            v=param_val,
        )

        # Creo la carpeta para ese parametro en la escena y objeto
        # correspondientes
        if not os.path.isdir(self.results_path):
            os.makedirs(self.results_path)

        rc = re.compile('(?P<number>\d{2})')
        pruebas_dirs = [l for l in os.listdir(self.results_path) if rc.match(l)]
        pruebas_dirs = pruebas_dirs if pruebas_dirs else ['00']
        pruebas_dirs.sort()
        last_test_number = int(rc.match(pruebas_dirs[-1]).groupdict()['number'])
        new_folder_name = '{n:02d}'.format(n=last_test_number + 1)
        self.results_path = os.path.join(self.results_path, new_folder_name)

        # Creo una carpeta distinta por numero de corrida
        if not os.path.isdir(self.results_path):
            os.makedirs(self.results_path)
        else:
            raise Exception('Ojo. Vas a pisar resultados!')

        self.file = open(os.path.join(self.results_path, 'results.txt'), 'w')

        self.write_parameter_values()

    def write_parameter_values(self):
        # Guardo los valores de los parametros
        ap_defaults = self.obj_follower.detector._ap_defaults
        self.file.write(b'ap_leaf={v}\n'.format(v=ap_defaults.leaf))
        self.file.write(b'ap_max_ransac_iters={v}\n'.format(
            v=ap_defaults.max_ransac_iters))
        self.file.write(b'ap_points_to_sample={v}\n'.format(
            v=ap_defaults.points_to_sample))
        self.file.write(b'ap_nearest_features_used={v}\n'.format(
            v=ap_defaults.nearest_features_used))
        self.file.write(b'ap_simil_threshold={v}\n'.format(
            v=ap_defaults.simil_threshold))
        self.file.write(b'ap_inlier_threshold={v}\n'.format(
            v=ap_defaults.inlier_threshold))
        self.file.write(b'ap_inlier_fraction={v}\n'.format(
            v=ap_defaults.inlier_fraction))

        icp_defaults = self.obj_follower.detector._icp_defaults
        self.file.write(b'det_euc_fit={v}\n'.format(v=icp_defaults.euc_fit))
        self.file.write(
            b'det_max_corr_dist={v}\n'.format(v=icp_defaults.max_corr_dist))
        self.file.write(
            b'det_max_iter={v}\n'.format(v=icp_defaults.max_iter))
        self.file.write(b'det_transf_epsilon={v}\n'.format(
            v=icp_defaults.transf_epsilon))

        icp_defaults = self.obj_follower.finder._icp_defaults
        self.file.write(b'seg_euc_fit={v}\n'.format(v=icp_defaults.euc_fit))
        self.file.write(
            b'seg_max_corr_dist={v}\n'.format(v=icp_defaults.max_corr_dist))
        self.file.write(
            b'seg_max_iter={v}\n'.format(v=icp_defaults.max_iter))
        self.file.write(b'seg_transf_epsilon={v}\n'.format(
            v=icp_defaults.transf_epsilon))
        self.file.write(b'RESULTS_SECTION\n')
        self.file.flush()


class FollowingSchemeSavingDataRGB(FollowingScheme):
    """
    Guarda las pruebas en carpetas consecutivas llamadas prueba_###
    """
    def __init__(self, img_provider, obj_follower, path):
        (super(FollowingSchemeSavingDataRGB, self)
         .__init__(img_provider, obj_follower, None))

        # para no pisar nunca los resultados voy a ir creando carpetas sucesivas
        # llamadas prueba_###
        self.results_path = os.path.join(path, '{s}_{sn}/{o}_{on}/')
        self.results_path = self.results_path.format(
            s=self.img_provider.scene,
            sn=self.img_provider.scene_number,
            o=self.img_provider.obj,
            on=self.img_provider.obj_number,
        )

        os.listdir(self.results_path)
        rc = re.compile('prueba_(?P<number>\d{3})')
        pruebas_dirs = [l for l in os.listdir(self.results_path) if rc.match(l)]
        pruebas_dirs = pruebas_dirs if pruebas_dirs else ['prueba_000']
        pruebas_dirs.sort()
        last_test_number = int(rc.match(pruebas_dirs[-1]).groupdict()['number'])
        new_folder_name = 'prueba_{n:03d}'.format(n=last_test_number+1)
        self.results_path = os.path.join(self.results_path, new_folder_name)

        if not os.path.isdir(self.results_path):
            os.makedirs(self.results_path)
        self.file = open(os.path.join(self.results_path, 'results.txt'), 'w')

        # Guardo los valores de los parametros
        self.write_parameter_values()

    def __del__(self):
        self.file.close()

    def write_parameter_values(self):
        self.file.write(b'RESULTS_SECTION\n')
        self.file.flush()

    def run(self):
        # ########################
        # Etapa de entrenamiento
        #########################
        self.obj_follower.train()

        ######################
        # Etapa de detección
        ######################
        print("Detectando en imagen {i}".format(
            i=self.img_provider.next_frame_number
        ))
        es_deteccion = True

        fue_exitoso, topleft, bottomright = (
            self.obj_follower.detect()
        )

        self.save_result(0, fue_exitoso, topleft, bottomright)

        #######################
        # Etapa de seguimiento
        #######################

        # Adelanto un frame
        self.img_provider.next()

        es_deteccion = True

        while self.img_provider.have_images():

            if fue_exitoso:
                print(
                    "Buscando en imagen {i}".format(
                        i=self.img_provider.next_frame_number
                    ),
                    ''
                )
                fue_exitoso, topleft, bottomright = (
                    self.obj_follower.follow(es_deteccion)
                )
                if fue_exitoso:
                    es_deteccion = False

            if not fue_exitoso:

                print("{p}Detectando en imagen {i}".format(
                    p='...MISS... ' if not es_deteccion else '',
                    i=self.img_provider.next_frame_number,
                ))
                fue_exitoso, topleft, bottomright = (
                    self.obj_follower.detect()
                )
                es_deteccion = True

            self.save_result(
                0 if es_deteccion else 1,
                fue_exitoso,
                topleft,
                bottomright
            )

            # Adelanto un frame
            self.img_provider.next()

    def save_result(self, method, fue_exitoso, topleft, bottomright):
        """
        Formato para guardar:

        frame_number;exito;metodo;fila_sup;col_izq;fila_inf;col_der

        donde:
        metodo = 0 si deteccion, 1 si seguimiento
        exito = 0 si fallo, 1 si funciono
        """
        nframe = self.img_provider.next_frame_number
        exito = 1 if fue_exitoso else 0

        values = [nframe, exito, method, int(topleft[0]), int(topleft[1]),
                  int(bottomright[0]), int(bottomright[1])]

        self.file.write(b';'.join([str(o) for o in values]))
        self.file.write(b'\n')
        self.file.flush()


class FollowingSquemaExploringParameterRGB(FollowingSchemeSavingDataRGB):
    """
    Guarda las pruebas en carpetas con el nombre del parametro y el valor
    que se estan explorando
    """

    def __init__(self, img_provider, obj_follower, path, param_name, param_val):
        self.img_provider = img_provider
        self.obj_follower = obj_follower
        self.show_following = None

        self.results_path = os.path.join(path, '{s}_{sn}/{o}_{on}/{p}/{v}')
        self.results_path = self.results_path.format(
            s=self.img_provider.scene,
            sn=self.img_provider.scene_number,
            o=self.img_provider.obj,
            on=self.img_provider.obj_number,
            p=param_name,
            v=param_val,
        )

        # Creo la carpeta para ese parametro en la escena y objeto
        # correspondientes
        if not os.path.isdir(self.results_path):
            os.makedirs(self.results_path)

        rc = re.compile('(?P<number>\d{2})')
        pruebas_dirs = [l for l in os.listdir(self.results_path) if rc.match(l)]
        pruebas_dirs = pruebas_dirs if pruebas_dirs else ['00']
        pruebas_dirs.sort()
        last_test_number = int(rc.match(pruebas_dirs[-1]).groupdict()['number'])
        new_folder_name = '{n:02d}'.format(n=last_test_number + 1)
        self.results_path = os.path.join(self.results_path, new_folder_name)

        # Creo una carpeta distinta por numero de corrida
        if not os.path.isdir(self.results_path):
            os.makedirs(self.results_path)
        else:
            raise Exception('Ojo. Vas a pisar resultados!')

        self.file = open(os.path.join(self.results_path, 'results.txt'), 'w')

        # Guardo los valores de los parametros
        self.write_parameter_values()


class FollowingSchemeSavingDataRGBD(FollowingSchemeSavingDataPCD):
    def write_parameter_values(self):
        self.file.write(b'RESULTS_SECTION\n')
        self.file.flush()


class FollowingSchemeExploringParameterRGBD(FollowingSquemaExploringParameterPCD):
    def write_parameter_values(self):
        self.file.write(b'RESULTS_SECTION\n')
        self.file.flush()