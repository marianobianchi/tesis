
#include <vector>

#include <pcl/io/pcd_io.h>
#include <pcl/filters/passthrough.h>

#include <boost/python.hpp>


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


BOOST_PYTHON_MODULE(cpp_main)
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
    
    // Funciones basicas
    def("read_pcd", read_pcd);
    def("filter_cloud", filter_cloud);
    def("points", points);
    
    
}
