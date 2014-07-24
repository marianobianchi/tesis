#include "icp_following.h"

ICPResult icp(pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud,
              pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud)
{
    /**
     * Calculate ICP transformation
     **/
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    icp.setInputSource(source_cloud);
    icp.setInputTarget(target_cloud);
    pcl::PointCloud<pcl::PointXYZ>::Ptr final (new pcl::PointCloud<pcl::PointXYZ>);
    icp.align(*final);

    ICPResult res;//(icp.hasConverged(), icp.getFitnessScore(), final);
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

    // Funciones basicas
    def("read_pcd", read_pcd);
    def("filter_cloud", filter_cloud);

    // Funciones que son de la clase PointCloud pero las manejo sueltas
    def("points", points);
    def("get_point", get_point);


    class_<ICPResult>("ICPResult")
        .def_readonly("has_converged", &ICPResult::has_converged)
        .def_readonly("score", &ICPResult::score)
        .def_readonly("cloud", &ICPResult::cloud);

    def("icp", icp);

}
