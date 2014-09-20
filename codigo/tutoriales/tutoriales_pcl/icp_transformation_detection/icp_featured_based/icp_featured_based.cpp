/**
 * Sacado de http://pointclouds.org/documentation/tutorials/registration_api.php
 * 
 * Feature based registration
 * - Use SIFT Keypoints (pcl::SIFT...something)
 * 
 * - Use FPFH descriptors (pcl::FPFHEstimation) at the keypoints (see 
 *   our tutorials for that, like http://www.pointclouds.org/media/rss2011.html)
 * 
 * - Get the FPFH descriptors and estimate correspondences using pcl::CorrespondenceEstimation
 * 
 * - Reject bad correspondences using one or many of the pcl::CorrespondenceRejectionXXX methods
 * 
 * - Estimate a transformation using the good correspondences (ICP)
 * 
 * */
 
#include "icp_featured_based.h"

void showHelp(char * program_name)
{
    std::cout << std::endl;
    std::cout << "Usage: " << program_name << " object.pcd scene.pcd" << std::endl;
    std::cout << "-h:  Show this help." << std::endl;
    std::cout << "-leaf:  Downsample leaf [0.005]" << std::endl;
    std::cout << "-fr:  Featrue estimation radius search [0.03]" << std::endl;
    std::cout << "-sac_it: Set inlier threshold: maximum distance between corresponding points (in meters) [2]" << std::endl;
    std::cout << "-sac_mi: Set maximum iterations [10000]" << std::endl;
}


void convert_to_normal(PointCloud3D::Ptr source_cloud, PointCloudN::Ptr target_cloud){
    
    pcl::NormalEstimationOMP<Point3D, PointN> ne;
    ne.setInputCloud (source_cloud);

    // Create an empty kdtree representation, and pass it to the normal estimation object.
    // Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
    KdTree3D::Ptr tree (new KdTree3D ());
    ne.setSearchMethod (tree);

    // Use all neighbors in a sphere of radius 3cm
    ne.setRadiusSearch (0.03);

    // Compute the features
    ne.compute (*target_cloud);
}


int main (int argc, char** argv)
{
    // Fetch point cloud filename in arguments | Works with PCD and PLY files
    std::vector<int> filenames;
    filenames = pcl::console::parse_file_extension_argument (argc, argv, ".pcd");
    
    if(pcl::console::find_argument(argc, argv, "-h") != -1){
        showHelp(argv[0]);
        return 0;
    }

    if (filenames.size () != 2) {
        showHelp(argv[0]);
        return -1;
    }
    
    /** Juntando valores de parametros **/
    float downsample_leaf = 0.005;
    if(pcl::console::find_argument(argc, argv, "-leaf") != -1){
        downsample_leaf = atof(argv[pcl::console::find_argument(argc, argv, "-leaf") + 1]);
    }
    
    float feature_radius = 0.03;
    if(pcl::console::find_argument(argc, argv, "-fr") != -1){
        feature_radius = atof(argv[pcl::console::find_argument(argc, argv, "-fr") + 1]);
    }
    
    
    double epsilon_sac = 2.0; // 2m
    if(pcl::console::find_argument(argc, argv, "-sac_it") != -1){
        epsilon_sac = atof(argv[pcl::console::find_argument(argc, argv, "-sac_it") + 1]);
    }
    
    int iter_sac = 10000;
    if(pcl::console::find_argument(argc, argv, "-sac_mi") != -1){
        iter_sac = atof(argv[pcl::console::find_argument(argc, argv, "-sac_mi") + 1]);
    }
    
    /** Define point clouds **/
    PointCloud3D::Ptr source_cloud (new PointCloud3D ());
    PointCloudN::Ptr source_normal_cloud (new PointCloudN ());
    FeatureCloudT::Ptr source_features (new FeatureCloudT);
    
    PointCloud3D::Ptr target_cloud (new PointCloud3D ());
    PointCloudN::Ptr target_normal_cloud (new PointCloudN ());
    FeatureCloudT::Ptr target_features (new FeatureCloudT);
    

    /** Load files **/
    if (pcl::io::loadPCDFile (argv[filenames[0]], *source_cloud) < 0)  {
        std::cout << "Error loading source point cloud " << argv[filenames[0]] << std::endl << std::endl;
        return -1;
    }
    if (pcl::io::loadPCDFile (argv[filenames[1]], *target_cloud) < 0)  {
        std::cout << "Error loading target point cloud " << argv[filenames[1]] << std::endl << std::endl;
        return -1;
    }
    
    
    /** Obtain keypoints (http://docs.pointclouds.org/trunk/classpcl_1_1_s_i_f_t_keypoint.html) **/
    // NECESITO OTRO TIPO DE ARCHIVO PCD
    //~ SIFTKeypoint sift;
    //~ sift.setKSearch(5);
    //~ sift.setRadiusSearch(0.03); // Use all neighbors in a sphere of radius 3cm
    //~ sift.setInputCloud(source_cloud);
    //~ sift.compute(*source_keypoint_cloud);
    //~ 
    //~ sift.setInputCloud(target_cloud);
    //~ sift.compute(*target_keypoint_cloud);
    
    // En vez de keypoints, hago un donwsample de la nube de puntos 
    // target que suele ser la más grande
    // Compute alignment
    std::cout << "Puntos en target = " << target_cloud->size() << std::endl;
    {
        pcl::ScopeTime t("Downsample...");
        pcl::VoxelGrid<Point3D> grid;
        grid.setLeafSize (downsample_leaf, downsample_leaf, downsample_leaf);
        grid.setInputCloud (target_cloud);
        grid.filter (*target_cloud);
    }
    std::cout << "Puntos en target downsampleado = " << target_cloud->size() << std::endl;
    
    
    /** Convert clouds to PointNormalCloud to use FPFHEstimation features **/
    {
        pcl::ScopeTime t("Normalizing...");
        convert_to_normal(source_cloud, source_normal_cloud);
        convert_to_normal(target_cloud, target_normal_cloud);
    }
    
    /** Obtain features **/
    {
        pcl::ScopeTime t("Features...");
        FeatureEstimationT fest;
        fest.setRadiusSearch (feature_radius);
        
        fest.setInputCloud (source_normal_cloud);
        fest.setInputNormals (source_normal_cloud);
        fest.compute (*source_features);
        
        fest.setInputCloud (target_normal_cloud);
        fest.setInputNormals (target_normal_cloud);
        fest.compute (*target_features);
    }
    std::cout << "Source features = " << source_features->size() << std::endl;
    std::cout << "Target features = " << target_features->size() << std::endl;
    
    /** Estimate correspondences **/
    boost::shared_ptr<pcl::Correspondences> all_correspondences (new pcl::Correspondences);
    {
        pcl::ScopeTime t("Estimate correspondences...");
        pcl::registration::CorrespondenceEstimation<FeatureT, FeatureT> est;
        est.setInputSource (source_features);
        est.setInputTarget (target_features);
        
        // Determine all reciprocal correspondences
        
        est.determineReciprocalCorrespondences (*all_correspondences);
    }
    
    /** Reject bad correspondences **/
    Eigen::Matrix4f transformation;
    boost::shared_ptr<pcl::Correspondences> inlier_correspondences (new pcl::Correspondences);
    {
        pcl::ScopeTime t("Reject bad correspondences...");
        
        
        pcl::registration::CorrespondenceRejectorSampleConsensus<Point3D> sac;
        sac.setInputSource (source_cloud);
        sac.setInputTarget (target_cloud);
        sac.setInlierThreshold (epsilon_sac);
        sac.setMaximumIterations (iter_sac);
        sac.setInputCorrespondences (all_correspondences);

        
        sac.getCorrespondences (*inlier_correspondences);
        PCL_INFO (" RANSAC: %d Correspondences Remaining\n", inlier_correspondences->size ());

        transformation = sac.getBestTransformation(); 
    }
}
