#!/usr/bin/python
#coding=utf-8

from __future__ import (unicode_literals, division)

from cpp.my_pcl import (points, filter_cloud, icp, ICPDefaults, save_pcd)

from esquemas_seguimiento import FollowingScheme
from metodos_comunes import (measure_time, Timer)
from observar_seguimiento import MuestraSeguimientoEnVivo
from proveedores_de_imagenes import FrameNamesAndImageProviderPreCharged
from seguidores_rgbd import (Follower, Finder)
from detector_estatico_seguimiento_icp import (
    FollowerWithStaticDetectionAndPCD, StaticDetectorWithPCDFiltering,
    ICPFinder,
)
from cpp.alignment_prerejective import (align, APDefaults)


class FollowerStaticICPAndObjectModel(FollowerWithStaticDetectionAndPCD):
    def train(self):
        obj_model = self.img_provider.obj_pcd()
        self._obj_descriptors.update({'obj_model': obj_model})


class StaticDetectorWithModelAlignment(StaticDetectorWithPCDFiltering):
    def calculate_descriptors(self, detected_descriptors):
        """
        Obtengo la nube de puntos correspondiente a la ubicacion y region
        pasadas por parametro.
        """
        detected_descriptors = (
            super(StaticDetectorWithModelAlignment, self)
            .calculate_descriptors(detected_descriptors)
        )

        model_cloud = self._descriptors['obj_model']
        detected_cloud = detected_descriptors['object_cloud']

        # Calculate ICP
        icp_defaults = ICPDefaults()
        icp_defaults.euc_fit = 1e-15
        icp_defaults.max_corr_dist = 3
        icp_defaults.max_iter = 50
        icp_defaults.transf_epsilon = 1e-15
        icp_result = icp(model_cloud, detected_cloud, icp_defaults)

        if icp_result.has_converged:
            ap_defaults = APDefaults()
            ap_result = align(model_cloud, icp_result.cloud, ap_defaults)
            ap_defaults.show_values = True
            ap_result = align(ap_result.cloud, detected_cloud, ap_defaults)

            if ap_result.has_converged:
                path = 'pruebas_guardadas/detector_con_modelo/'
                save_pcd(ap_result.cloud, str(path + 'deteccion_ap.pcd'))

                detected_descriptors['object_cloud'] = ap_result.cloud
                detected_descriptors['obj_model'] = ap_result.cloud

        return detected_descriptors


class ICPFinderWithModel(ICPFinder):
    def simple_follow(self,
                      r_top_limit,
                      r_bottom_limit,
                      c_left_limit,
                      c_right_limit,
                      object_cloud,
                      target_cloud):
        """
        Tomando como centro el centro del cuadrado que contiene al objeto
        en el frame anterior, busco el mismo objeto en una zona 4 veces mayor
        a la original.
        """
        # Define row and column limits for the zone to search the object
        # In this case, we look on a box N times the size of the original
        n = 2
        r_top_limit = r_top_limit - ( (r_bottom_limit - r_top_limit) * n)
        r_bottom_limit = r_bottom_limit + ( (r_bottom_limit - r_top_limit) * n)
        c_left_limit = c_left_limit - ( (c_right_limit - c_left_limit) * n)
        c_right_limit = c_right_limit + ( (c_right_limit - c_left_limit) * n)

        # Filter points corresponding to the zone where the object being
        # followed is supposed to be
        filter_cloud(target_cloud, str("y"), float(r_top_limit), float(r_bottom_limit))
        filter_cloud(target_cloud, str("x"), float(c_left_limit), float(c_right_limit))

        model_cloud = self._descriptors['obj_model']


        nframe = self._descriptors['nframe']
        path = 'pruebas_guardadas/detector_con_modelo/'
        save_pcd(model_cloud, str(path + 'modelo_{i}.pcd'.format(i=nframe)))
        save_pcd(object_cloud, str(path + 'source_{i}.pcd'.format(i=nframe)))
        save_pcd(target_cloud, str(path + 'target_{i}.pcd'.format(i=nframe)))

        # Calculate ICP
        icp_defaults = ICPDefaults()
        icp_result = icp(object_cloud, target_cloud, icp_defaults)

        save_pcd(icp_result.cloud, str(path + 'icp_{i}.pcd'.format(i=nframe)))

        if icp_result.has_converged:
            ap_defaults = APDefaults()
            #ap_defaults.show_values = True
            aligned_prerejective_result = align(model_cloud, icp_result.cloud, ap_defaults)

            if aligned_prerejective_result.has_converged:
                save_pcd(aligned_prerejective_result.cloud, str(path + 'ap_{i}.pcd'.format(i=nframe)))
                return aligned_prerejective_result

        return icp_result


def prueba_seguimiento_ICP_con_modelo():
    img_provider = FrameNamesAndImageProviderPreCharged(
        'videos/rgbd/scenes/', 'desk', '1', 'videos/rgbd/objs/', 'coffee_mug', '5',
    )  # path, objname, number

    detector = StaticDetectorWithModelAlignment(
        'videos/rgbd/scenes/desk/desk_1.mat',
        'coffee_mug'
    )

    finder = ICPFinderWithModel()

    follower = FollowerStaticICPAndObjectModel(img_provider, detector, finder)

    show_following = MuestraSeguimientoEnVivo('Seguidor ICP')

    FollowingScheme(
        img_provider,
        follower,
        show_following,
    ).run()


if __name__ == '__main__':
    prueba_seguimiento_ICP_con_modelo()