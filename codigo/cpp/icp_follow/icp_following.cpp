#include "icp_following.h"

ICPResult icp(PointCloud3D::Ptr source_cloud,
              PointCloud3D::Ptr target_cloud,
              ICPDefaults &icp_defaults)
{
    /**
     * Calculate ICP transformation
     **/
    pcl::IterativeClosestPoint<Point3D, Point3D> icp;

    // Set the max correspondence distance (in meters). Correspondences with higher distances will be ignored
    icp.setMaxCorrespondenceDistance(icp_defaults.max_corr_dist);

    // Set the maximum number of iterations (termination critera 1) Ejemplo: 50
    icp.setMaximumIterations(icp_defaults.max_iter);

    // Set the transformation epsilon (termination critera 2) Ejemplo: 1e-8
    icp.setTransformationEpsilon(icp_defaults.transf_epsilon);

    // Set the euclidean distance difference epsilon (termination critera 3) Ejemplo: 1
    icp.setEuclideanFitnessEpsilon(icp_defaults.euc_fit);

    icp.setRANSACIterations(icp_defaults.ran_iter);
    icp.setRANSACOutlierRejectionThreshold(icp_defaults.ran_out_rej);

    icp.setInputSource(source_cloud);
    icp.setInputTarget(target_cloud);
    PointCloud3D::Ptr final (new PointCloud3D);
    icp.align(*final);

    ICPResult res;
    res.has_converged = icp.hasConverged();
    res.score = icp.getFitnessScore();
    res.cloud = final;
    res.transformation = mat_to_vector(icp.getFinalTransformation());

    if(icp_defaults.show_values){

        // SET DIFFERENT PARAMETERS
        std::cout << "ICP EPSILON DEFAULT = ";
        std::cout << icp.getEuclideanFitnessEpsilon() << std::endl;

        std::cout << "MAX CORR DISTANCE DEFAULT = ";
        std::cout << icp.getMaxCorrespondenceDistance() << std::endl;

        std::cout << "MAX ITERATIONS DEFAULT = ";
        std::cout << icp.getMaximumIterations() << std::endl;

        std::cout << "TRANSF EPSILON DEFAULT = ";
        std::cout << icp.getTransformationEpsilon() << std::endl;

        std::cout << "RANSAC ITERATIONS DEFAULT = ";
        std::cout << icp.getRANSACIterations() << std::endl;

        std::cout << "RANSAC OUTLIER REJECTION THRESHOLD DEFAULT = ";
        std::cout << icp.getRANSACOutlierRejectionThreshold() << std::endl;

        pcl::visualization::PCLVisualizer visu("ICP");
        pcl::console::print_info("Verde = escena\n");
        pcl::console::print_info("Azul = modelo alineado\n");
        visu.addPointCloud (target_cloud, ColorHandler3D (target_cloud, 0.0, 255.0, 0.0), "target");
        visu.addPointCloud (final, ColorHandler3D (final, 0.0, 0.0, 255.0), "object_aligned");
        visu.spin ();
    }

    return res;
}





/*
 * Exporto todo a python
 * */

BOOST_PYTHON_MODULE(icp)
{
    using namespace boost::python;

    class_<ICPDefaults>("ICPDefaults")
        .def(init<>())
        .def_readwrite("euc_fit", &ICPDefaults::euc_fit)
        .def_readwrite("max_corr_dist", &ICPDefaults::max_corr_dist)
        .def_readwrite("max_iter", &ICPDefaults::max_iter)
        .def_readwrite("transf_epsilon", &ICPDefaults::transf_epsilon)
        .def_readwrite("ran_iter", &ICPDefaults::ran_iter)
        .def_readwrite("ran_out_rej", &ICPDefaults::ran_out_rej)
        .def_readwrite("show_values", &ICPDefaults::show_values);

    class_<ICPResult>("ICPResult")
        .def_readwrite("has_converged", &ICPResult::has_converged)
        .def_readwrite("score", &ICPResult::score)
        .def_readwrite("cloud", &ICPResult::cloud)
        .def_readwrite("transformation", &ICPResult::transformation);

    def("icp", icp);
}
