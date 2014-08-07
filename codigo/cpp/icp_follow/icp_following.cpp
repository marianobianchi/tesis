#include "icp_following.h"

ICPResult icp(pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud,
              pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud,
              ICPDefaults &icp_defaults)
{
    /**
     * Calculate ICP transformation
     **/
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;


    // Set the max correspondence distance (in meters)
    icp.setMaxCorrespondenceDistance(icp_defaults.max_corr_dist);

    // Ejemplo: 50
    icp.setMaximumIterations(icp_defaults.max_iter);

    // Ejemplo: 1e-8
    icp.setTransformationEpsilon(icp_defaults.transf_epsilon);

    // Ejemplo: 1
    ic(icp_defaults.euc_fit);



    icp.setRANSACIterations(icp_defaults.ran_iter);
    icp.setRANSACOutlierRejectionThreshold(icp_defaults.ran_out_rej);


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

        // setTransformationEstimation (SVD, point to plane, etc)
    }

    icp.setInputSource(source_cloud);
    icp.setInputTarget(target_cloud);
    pcl::PointCloud<pcl::PointXYZ>::Ptr final (new pcl::PointCloud<pcl::PointXYZ>);
    icp.align(*final);

    ICPResult res;
    res.has_converged = icp.hasConverged();
    res.score = icp.getFitnessScore();
    res.cloud = final;

    return res;
}


pcl::PointCloud<pcl::PointXYZ>::Ptr read_pcd(std::string pcd_filename)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    if (pcl::io::loadPCDFile<pcl::PointXYZ> (pcd_filename, *cloud) == -1) //* load the file
    {
        PCL_ERROR ("Couldn't read file\n");
        throw;
    }

    return cloud;
}

void save_pcd(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, std::string fname){
    pcl::io::savePCDFileBinary(fname, *cloud);
}


void filter_cloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                  const std::string & field_name,
                  const float & lower_limit,
                  const float & upper_limit)
{
    // Create the filtering object
    pcl::PassThrough<pcl::PointXYZ> pass;

    pass.setInputCloud(cloud);
    pass.setFilterFieldName(field_name);
    pass.setFilterLimits(lower_limit, upper_limit);

    // filter
    pass.filter(*cloud);
}

int points(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
    return cloud->points.size();
}

pcl::PointXYZ get_point(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, int i)
{
    return cloud->points[i];
}

/*
 * Exporto todo a python
 * */

BOOST_PYTHON_MODULE(my_pcl)
{

    using namespace boost::python;

    /*
     * Comparto para python lo minimo indispensable para usar
     * PointCloud's de manera razonable
     * */

    // Comparto el shared_ptr
    boost::python::register_ptr_to_python< boost::shared_ptr<pcl::PointCloud<pcl::PointXYZ> > >();

    // Comparto la clase point cloud
    class_<pcl::PointCloud<pcl::PointXYZ> >("PointCloudXYZ");

    class_<pcl::PointXYZ>("PointXYZ")
        .def_readonly("x", &pcl::PointXYZ::x)
        .def_readonly("y", &pcl::PointXYZ::y)
        .def_readonly("z", &pcl::PointXYZ::z);

    class_<ICPDefaults>("ICPDefaults")
        .def(init<>())
        .def_readwrite("euc_fit", &ICPDefaults::euc_fit)
        .def_readwrite("max_corr_dist", &ICPDefaults::max_corr_dist)
        .def_readwrite("max_iter", &ICPDefaults::max_iter)
        .def_readwrite("transf_epsilon", &ICPDefaults::transf_epsilon)
        .def_readwrite("ran_iter", &ICPDefaults::ran_iter)
        .def_readwrite("ran_out_rej", &ICPDefaults::ran_out_rej)
        .def_readwrite("show_values", &ICPDefaults::show_values);

    // Funciones basicas
    def("read_pcd", read_pcd);
    def("save_pcd", save_pcd);
    def("filter_cloud", filter_cloud);

    // Funciones que son de la clase PointCloud pero las manejo sueltas
    def("points", points);
    def("get_point", get_point);

    class_<ICPResult>("ICPResult")
        .def_readonly("has_converged", &ICPResult::has_converged)
        .def_readonly("score", &ICPResult::score)
        .def_readwrite("cloud", &ICPResult::cloud);

    def("icp", icp);

}
