#include "common.h"



PointCloud3D::Ptr voxel_grid_downsample(PointCloud3D::Ptr cloud, float leaf){

    PointCloud3D::Ptr downsampled_cloud(new PointCloud3D);

    pcl::VoxelGrid<Point3D> grid;
    grid.setLeafSize (leaf, leaf, leaf);
    grid.setInputCloud (cloud);
    grid.filter (*downsampled_cloud);

    return downsampled_cloud;

}


PointCloud3D::Ptr filter_object_from_scene_cloud(PointCloud3D::Ptr object_cloud,
                                                 PointCloud3D::Ptr scene_cloud,
                                                 float radius, // 0.001 = 1 mm
                                                 bool show_values)
{

    PointCloud3D::Ptr matching_points_cloud (new PointCloud3D);

    // Tomo a la nube de puntos del objeto como area de busqueda
    pcl::KdTreeFLANN<Point3D> kdtree;
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


PointCloud3D::Ptr filter_cloud(PointCloud3D::Ptr const_cloud,
                  const std::string & field_name,
                  const float & lower_limit,
                  const float & upper_limit)
{
    PointCloud3D::Ptr cloud(new PointCloud3D);

    // Create the filtering object
    pcl::PassThrough<Point3D> pass;

    pass.setInputCloud(const_cloud);
    pass.setFilterFieldName(field_name);
    pass.setFilterLimits(lower_limit, upper_limit);

    // filter
    pass.filter(*cloud);

    return cloud;
}


VectorMat mat_to_vector(const Mat& m){
    VectorMat s(4, std::vector<float>(4, 0.0));
    s[0][0] = m(0,0);
    s[0][1] = m(0,1);
    s[0][2] = m(0,2);
    s[0][3] = m(0,3);

    s[1][0] = m(1,0);
    s[1][1] = m(1,1);
    s[1][2] = m(1,2);
    s[1][3] = m(1,3);

    s[2][0] = m(2,0);
    s[2][1] = m(2,1);
    s[2][2] = m(2,2);
    s[2][3] = m(2,3);

    return s;
}

Mat vector_to_mat(const VectorMat& s){
    Mat m;
    m(0,0) = s[0][0];
    m(0,1) = s[0][1];
    m(0,2) = s[0][2];
    m(0,3) = s[0][3];

    m(1,0) = s[1][0];
    m(1,1) = s[1][1];
    m(1,2) = s[1][2];
    m(1,3) = s[1][3];

    m(2,0) = s[2][0];
    m(2,1) = s[2][1];
    m(2,2) = s[2][2];
    m(2,3) = s[2][3];

    return m;
}

PointCloud3D::Ptr transform_cloud(PointCloud3D::Ptr cloud, const VectorMat& transformation){
    Mat m = vector_to_mat(transformation);
    PointCloud3D::Ptr transformed_cloud(new PointCloud3D);
    pcl::transformPointCloud(*cloud, *transformed_cloud, m);
    return transformed_cloud;
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
    visu.close();
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
