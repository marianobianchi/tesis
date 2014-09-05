#include "kdtree.h"

void showHelp(char * program_name)
{
    std::cout << std::endl;
    std::cout << "Usage: " << program_name << " object.pcd" << std::endl;
    std::cout << "-h or --help:  Show this help." << std::endl;
    std::cout << "-r: Radius (in meters)" << std::endl;
}


int main (int argc, char** argv)
{
    PointCloud3D::Ptr object_cloud (new PointCloud3D);
    PointCloud3D::Ptr scene_cloud (new PointCloud3D);
    PointCloud3D::Ptr matching_points_cloud (new PointCloud3D);
    
    std::vector<int> filenames;
    filenames = pcl::console::parse_file_extension_argument (argc, argv, ".pcd");
    
    if (filenames.size () != 2 || pcl::console::find_switch (argc, argv, "-h") || pcl::console::find_switch (argc, argv, "--help")){
        showHelp (argv[0]);
        return (1);
    }
    
    
    float radius = 0.001;
    if(pcl::console::find_argument(argc, argv, "-r") != -1){
        radius = atof(argv[pcl::console::find_argument(argc, argv, "-r") + 1]);
    }
  
    // Load object and scene
    if (pcl::io::loadPCDFile<Point3D> (argv[1], *object_cloud) < 0 || 
        pcl::io::loadPCDFile<Point3D> (argv[2], *scene_cloud) < 0)
    {
        pcl::console::print_error ("Error loading object/scene file!\n");
        return (1);
    }

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
    
    std::cout << "Puntos totales de la escena: " << scene_cloud->size() << std::endl;
    std::cout << "Puntos de la escena con correspondencias en el objeto: " << matching_points_cloud->size() << std::endl;
    
    for (int i = 0; i < 10; ++i)
        std::cout << "Puntos de la escena con " << i + 1 << " vecinos:" << neighbors_count[i] << std::endl;


    return 0;
}
