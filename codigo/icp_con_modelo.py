#!/usr/bin/python
#coding=utf-8

from __future__ import (unicode_literals, division)

from cpp.my_pcl import (points, filter_cloud, icp, ICPDefaults, save_pcd)
from cpp.alignment_prerejective import (align, APDefaults)

from esquemas_seguimiento import FollowingScheme
from metodos_comunes import (measure_time, Timer)
from observar_seguimiento import MuestraSeguimientoEnVivo
from proveedores_de_imagenes import FrameNamesAndImageProviderPreCharged
from icp_sin_modelo import (
    FollowerWithStaticDetectionAndPCD, StaticDetectorWithPCDFiltering,
    ICPFinder,
)


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
            ap_defaults.simil_threshold = 0.1
            ap_defaults.inlier_fraction = 0.7
            ap_defaults.show_values = True
            ap_result = align(icp_result.cloud, detected_cloud, ap_defaults)

            if ap_result.has_converged:
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
        en el frame anterior, busco el mismo objeto en una zona N veces mayor
        a la original.
        """
        # Define row and column limits for the zone to search the object
        # In this case, we look on a box N times the size of the original
        # i.e: if height is 1 and i want a box 2 times bigger and centered
        # on the center of the original box, i have to substract 0.5 times the
        # height to the top of the box and add the same amount to the bottom
        n = 2

        factor = 0.5 * (n - 1)
        height = r_bottom_limit - r_top_limit
        width = c_right_limit - c_left_limit

        r_top_limit -= height * factor
        r_bottom_limit += height * factor
        c_left_limit -= width * factor
        c_right_limit += width * factor

        # Filter points corresponding to the zone where the object being
        # followed is supposed to be
        filter_cloud(target_cloud, str("y"), float(r_top_limit), float(r_bottom_limit))
        filter_cloud(target_cloud, str("x"), float(c_left_limit), float(c_right_limit))

        model_cloud = self._descriptors['obj_model']

        nframe = self._descriptors['nframe']
        path = 'pruebas_guardadas/detector_con_modelo/'
        save_pcd(target_cloud, str(path + 'target_{i}.pcd'.format(i=nframe)))

        # Calculate ICP
        icp_defaults = ICPDefaults()
        icp_result = icp(object_cloud, target_cloud, icp_defaults)
        save_pcd(icp_result.cloud, str(path + 'icp_{i}.pcd'.format(i=nframe)))

        #if icp_result.has_converged:
        #    ap_defaults = APDefaults()
        #    ap_defaults.simil_threshold = 0.1
        #    ap_defaults.inlier_fraction = 0.7
        #    #ap_defaults.show_values = True
        #    aligned_prerejective_result = align(model_cloud, icp_result.cloud, ap_defaults)
        #
        #    if aligned_prerejective_result.has_converged:
        #        return aligned_prerejective_result

        return icp_result

    def calculate_descriptors(self, detected_descriptors):
        detected_descriptors = (super(ICPFinderWithModel, self)
                                .calculate_descriptors(detected_descriptors))
        detected_descriptors['obj_model'] = detected_descriptors['object_cloud']
        return detected_descriptors



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