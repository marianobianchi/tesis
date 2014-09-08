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

    // Set the euclidean distance difference epsilon (termination critera 2) Ejemplo: 1
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



PointCloud3D::Ptr filter_object_from_scene_cloud(PointCloud3D::Ptr object_cloud,
                                                 PointCloud3D::Ptr scene_cloud,
                                                 float radius, // 0.001 = 1 mm
                                                 bool show_values)
{

    PointCloud3D::Ptr matching_points_cloud (new PointCloud3D);
    
    // Tomo a la nube de puntos del objeto como area de busqueda
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(object_cloud);
    
    // Busco los puntos de la escena que tienen "correspondencias" con el objeto
    Point3D search_point;
    std::vector<int> pointIdxRadiusSearch;
    std::vector<float> pointRadiusSquaredDistance;
    
    int neighbors_count[10] = {0,0,0,0,0,0,0,0,0,0};
    int neighbors;
    
    for(int i=0; i<=scene_cloud->size(); ++i){
        search_point = scene_cloud->points[i];
        pointIdxRadiusSearch.clear();
        pointRadiusSquaredDistance.clear();
        
        // Neighbors within radius search
        neighbors = kdtree.radiusSearch (search_point, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance);
        if ( neighbors > 0 ){
            if(neighbors >= 10){
                neighbors_count[9] += 1;
            }
            else{
                neighbors_count[neighbors - 1] += 1;
            }
            matching_points_cloud->push_back(search_point);
        }
        
    }

    if ( show_values ){
        std::cout << "Radio de busqueda: " << radius << std::endl;
        std::cout << "Puntos totales de la escena: " << scene_cloud->size() << std::endl;
        std::cout << "Puntos de la escena con correspondencias en el objeto: " << matching_points_cloud->size() << std::endl;
        
        for (int i = 0; i < 10; ++i)
            std::cout << "Puntos de la escena con " << i + 1 << " vecinos:" << neighbors_count[i] << std::endl;

    }

    return matching_points_cloud;

}



PointCloud3D::Ptr read_pcd(std::string pcd_filename)
{
    PointCloud3D::Ptr cloud(new PointCloud3D);

    if (pcl::io::loadPCDFile<Point3D> (pcd_filename, *cloud) == -1) //* load the file
    {
        PCL_ERROR ("Couldn't read file\n");
        throw;
    }

    return cloud;
}

void save_pcd(PointCloud3D::Ptr cloud, std::string fname){
    pcl::io::savePCDFileBinary(fname, *cloud);
}


void filter_cloud(PointCloud3D::Ptr cloud,
                  const std::string & field_name,
                  const float & lower_limit,
                  const float & upper_limit)
{
    // Create the filtering object
    pcl::PassThrough<Point3D> pass;

    pass.setInputCloud(cloud);
    pass.setFilterFieldName(field_name);
    pass.setFilterLimits(lower_limit, upper_limit);

    // filter
    pass.filter(*cloud);
}

int points(PointCloud3D::Ptr cloud)
{
    return cloud->points.size();
}

Point3D get_point(PointCloud3D::Ptr cloud, int i)
{
    return cloud->points[i];
}

void show_clouds(std::string title, PointCloud3D::Ptr first_cloud, PointCloud3D::Ptr second_cloud){
    if(title == "") title = "Showing clouds";
    pcl::visualization::PCLVisualizer visu(title);
    pcl::console::print_info("Verde = first_cloud\n");
    pcl::console::print_info("Azul = second_cloud\n");
    visu.addPointCloud (first_cloud, ColorHandler3D (first_cloud, 0.0, 255.0, 0.0), "first_cloud");
    visu.addPointCloud (second_cloud, ColorHandler3D (second_cloud, 0.0, 0.0, 255.0), "second_cloud");
    visu.spin ();
}

MinMax3D get_min_max3D(PointCloud3D::Ptr cloud){
    Point3D min;
    Point3D max;
    pcl::getMinMax3D(*cloud, min, max);
    
    MinMax3D minmax;
    minmax.min_x = min.x;
    minmax.min_y = min.y;
    minmax.min_z = min.z;
    minmax.max_x = max.x;
    minmax.max_y = max.y;
    minmax.max_z = max.z;
    
    return minmax;
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
    boost::python::register_ptr_to_python< boost::shared_ptr<PointCloud3D > >();

    // Comparto la clase point cloud
    class_<PointCloud3D >("PointCloudXYZ");

    class_<Point3D>("PointXYZ")
        .def_readonly("x", &Point3D::x)
        .def_readonly("y", &Point3D::y)
        .def_readonly("z", &Point3D::z);

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
    def("show_clouds", show_clouds);

    class_<ICPResult>("ICPResult")
        .def_readonly("has_converged", &ICPResult::has_converged)
        .def_readonly("score", &ICPResult::score)
        .def_readwrite("cloud", &ICPResult::cloud);

    def("icp", icp);
    
    class_<MinMax3D>("MinMax3D")
        .def_readonly("min_x", &MinMax3D::min_x)
        .def_readonly("min_y", &MinMax3D::min_y)
        .def_readonly("min_z", &MinMax3D::min_z)
        .def_readonly("max_x", &MinMax3D::max_x)
        .def_readonly("max_y", &MinMax3D::max_y)
        .def_readonly("max_z", &MinMax3D::max_z);

    def("get_min_max", get_min_max3D);

}
