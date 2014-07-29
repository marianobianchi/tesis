#!/usr/bin/python
#coding=utf-8

from __future__ import (unicode_literals, division)

from cpp.my_pcl import (points, filter_cloud, icp, ICPDefaults, save_pcd)

from esquemas_seguimiento import FollowingScheme
from metodos_comunes import (from_flat_to_cloud_limits, from_cloud_to_flat,
                             measure_time, Timer)
from observar_seguimiento import MuestraSeguimientoEnVivo
from proveedores_de_imagenes import FrameNamesAndImageProviderPreCharged
from seguidores_rgbd import (Follower, Finder)
from detector_estatico_seguimiento_icp import (
    FollowerWithStaticDetectionAndPCD, StaticDetectorWithPCDFiltering,
    ICPFinder,
)


class FollowerStaticICPAndObjectModel(FollowerWithStaticDetectionAndPCD):
    def train(self):
        obj_model = self.img_provider.obj_pcd()
        self._obj_descriptors.update({'obj_model': obj_model})


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

        # Calculate ICP
        icp_defaults = ICPDefaults()
        icp_result = icp(object_cloud, target_cloud, icp_defaults)

        print "Convergió el primero?", icp_result.has_converged, "(",icp_result.score,")"
        save_pcd(object_cloud, str('object_cloud.pcd'))
        save_pcd(target_cloud, str('target_cloud.pcd'))
        save_pcd(icp_result.cloud, str('transformed_object.pcd'))

        if icp_result.has_converged:
            icp_defaults.max_corr_dist = 4
            icp_defaults.max_iter = 50
            icp_defaults.transf_epsilon = 1e-12
            icp_defaults.euc_fit = 1e-12
            #icp_defaults.ran_iter = 1000000000
            #icp_defaults.ran_out_rej = 0.5e-100
            #icp_defaults.show_values = True

            found_model_cloud = model_cloud
            last_converged_icp = None
            for i in range(30):
                new_icp_result = icp(found_model_cloud, icp_result.cloud, icp_defaults)
                print "Convergió {i}?".format(i=i), new_icp_result.has_converged, "(",new_icp_result.score,")"
                if new_icp_result.has_converged:
                    if last_converged_icp is None:
                        last_converged_icp = new_icp_result
                    found_model_cloud = new_icp_result.cloud
                    if new_icp_result.score <= last_converged_icp.score:
                        last_converged_icp = new_icp_result

                if icp_defaults.max_corr_dist > 0.04:
                    icp_defaults.max_corr_dist /= 2.0

            print "Score de la ultima que convergió:", last_converged_icp.score
            save_pcd(model_cloud, str('model.pcd'))
            save_pcd(last_converged_icp.cloud, str('posta.pcd'))

            return last_converged_icp

        return first_icp_result


def prueba_seguimiento_ICP_con_modelo():
    img_provider = FrameNamesAndImageProviderPreCharged(
        'videos/rgbd/scenes/', 'desk', '1', 'videos/rgbd/objs/', 'coffee_mug', '5',
    )  # path, objname, number

    detector = StaticDetectorWithPCDFiltering(
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