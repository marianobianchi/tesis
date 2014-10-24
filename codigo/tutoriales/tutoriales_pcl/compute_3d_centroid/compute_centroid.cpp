#include <Eigen/Core>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/centroid.h>


// Types
typedef pcl::PointXYZ Point3D;
typedef pcl::PointCloud<Point3D> PointCloud3D;


// Align a rigid object to a scene with clutter and occlusions
Point3D compute_centroid(float x1, float y1, float z1, float x2, float y2, float z2){
    
    // Point cloud
    PointCloud3D::Ptr cloud(new PointCloud3D);
    
    cloud->width = 8;
    cloud->height = 1;
    cloud->points.resize(cloud->width * cloud->height);
    
    // Tapa de arriba (de un prisma/cubo)
    cloud->points[0].x = x1;
    cloud->points[0].y = y1;
    cloud->points[0].z = z1;
    
    cloud->points[1].x = x1;
    cloud->points[1].y = y2;
    cloud->points[1].z = z1;
    
    cloud->points[2].x = x1;
    cloud->points[2].y = y1;
    cloud->points[2].z = z2;
    
    cloud->points[3].x = x1;
    cloud->points[3].y = y2;
    cloud->points[3].z = z2;
    
    // Tapa de abajo
    cloud->points[4].x = x2;
    cloud->points[4].y = y1;
    cloud->points[4].z = z1;
    
    cloud->points[5].x = x2;
    cloud->points[5].y = y2;
    cloud->points[5].z = z1;
    
    cloud->points[6].x = x2;
    cloud->points[6].y = y1;
    cloud->points[6].z = z2;
    
    cloud->points[7].x = x2;
    cloud->points[7].y = y2;
    cloud->points[7].z = z2;
    
    // centroid
    Eigen::Matrix<float,4,1> centroid;
    
    // Compute centroid
    pcl::compute3DCentroid(*cloud, centroid);
    
    Point3D res;
    res.x = centroid(0,0);
    res.y = centroid(1,0);
    res.z = centroid(2,0);
    
    return res;

}


/*
 * Includes para exportar a python
 * */
#include <boost/python.hpp>
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>



/*
 * Exporto todo a python
 * */

BOOST_PYTHON_MODULE(compute_centroid)
{

    using namespace boost::python;

    /*
     * Comparto para python lo minimo indispensable para usar
     * PointCloud's de manera razonable
     * */
    
    class_<Point3D>("Point3D")
        .def_readonly("x", &Point3D::x)
        .def_readonly("y", &Point3D::y)
        .def_readonly("z", &Point3D::z);
    
    def("compute_centroid", compute_centroid);

}
