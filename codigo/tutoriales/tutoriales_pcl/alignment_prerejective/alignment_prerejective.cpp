// http://pointclouds.org/documentation/tutorials/alignment_prerejective.php
#include <Eigen/Core>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/time.h>
#include <pcl/console/print.h>
#include <pcl/console/parse.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/sample_consensus_prerejective.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/visualization/pcl_visualizer.h>

// Types
typedef pcl::PointXYZ Point3D;
typedef pcl::PointCloud<Point3D> PointCloud3D;
typedef pcl::PointNormal PointNT;
typedef pcl::PointCloud<PointNT> PointCloudT;
typedef pcl::FPFHSignature33 FeatureT;
typedef pcl::FPFHEstimationOMP<PointNT,PointNT,FeatureT> FeatureEstimationT;
typedef pcl::PointCloud<FeatureT> FeatureCloudT;
typedef pcl::visualization::PointCloudColorHandlerCustom<Point3D> ColorHandler3D;
typedef pcl::visualization::PointCloudColorHandlerCustom<PointNT> ColorHandlerT;


void show_normal_cloud(std::string title, PointCloudT::Ptr scene){
    pcl::visualization::PCLVisualizer visu(title);
    visu.addPointCloud (scene, ColorHandlerT (scene, 0.0, 255.0, 0.0), "scene");
    visu.spin ();
}

void show_cloud(std::string title, PointCloud3D::Ptr scene){
    pcl::visualization::PCLVisualizer visu(title);
    visu.addPointCloud (scene, ColorHandler3D (scene, 0.0, 255.0, 0.0), "scene");
    visu.spin ();
}


void showHelp(char * program_name)
{
    std::cout << std::endl;
    std::cout << "Usage: " << program_name << " object.pcd scene.pcd" << std::endl;
    std::cout << "-h:  Show this help." << std::endl;
    std::cout << "-leaf:  Downsample leaf" << std::endl;
    std::cout << "-mi:  Max. RANSAC iterations" << std::endl;
    std::cout << "-sn:  Number of points to sample for generating/prerejecting a pose (setNumberOfSamples)" << std::endl;
    std::cout << "-fn:  Number of nearest features to use (setCorrespondenceRandomness)" << std::endl;
    std::cout << "-st:  Polygonal edge length similarity threshold (setSimilarityThreshold)" << std::endl;
    std::cout << "-it:  Inlier threshold" << std::endl;
    std::cout << "-if:  Required inlier fraction for accepting a pose hypothesis" << std::endl;
    
    /*
     * 
     * Number of samples - setNumberOfSamples (): The number of point correspondences to sample 
     *                     between the object and the scene. At minimum, 3 points are required 
     *                     to calculate a pose.
     * Correspondence randomness - setCorrespondenceRandomness (): Instead of matching each object 
     *                             FPFH descriptor to its nearest matching feature in the scene, we 
     *                             can choose between the N best matches at random. This increases 
     *                             the iterations necessary, but also makes the algorithm robust 
     *                             towards outlier matches.
     * Polygonal similarity threshold - setSimilarityThreshold (): The alignment class uses the 
     *                                  CorrespondenceRejectorPoly class for early elimination of bad 
     *                                  poses based on pose-invariant geometric consistencies of the 
     *                                  inter-distances between sampled points on the object and the 
     *                                  scene. The closer this value is set to 1, the more greedy and 
     *                                  thereby fast the algorithm becomes. However, this also 
     *                                  increases the risk of eliminating good poses when noise is 
     *                                  present.
     * Inlier threshold - setMaxCorrespondenceDistance (): This is the Euclidean distance threshold 
     *                    used for determining whether a transformed object point is correctly aligned
     *                    to the nearest scene point or not. In this example, we have used a heuristic 
     *                    value of 1.5 times the point cloud resolution.
     * Inlier fraction - setInlierFraction (): In many practical scenarios, large parts of the observed 
     *                   object in the scene are not visible, either due to clutter, occlusions or both. 
     *                   In such cases, we need to allow for pose hypotheses that do not align all 
     *                   object points to the scene. The absolute number of correctly aligned points is 
     *                   determined using the inlier threshold, and if the ratio of this number to the 
     *                   total number of points in the object is higher than the specified inlier 
     *                   fraction, we accept a pose hypothesis as valid.

     * */
}


// Align a rigid object to a scene with clutter and occlusions
int main (int argc, char **argv)
{
    // Point clouds
    PointCloud3D::Ptr loaded_object (new PointCloud3D);
    PointCloud3D::Ptr loaded_scene (new PointCloud3D);
    PointCloud3D::Ptr filtered_object (new PointCloud3D);
    PointCloud3D::Ptr filtered_scene (new PointCloud3D);
    
    PointCloudT::Ptr object (new PointCloudT);
    PointCloudT::Ptr object_aligned (new PointCloudT);
    PointCloudT::Ptr scene (new PointCloudT);
    FeatureCloudT::Ptr object_features (new FeatureCloudT);
    FeatureCloudT::Ptr scene_features (new FeatureCloudT);
  
    std::vector<int> filenames;
    filenames = pcl::console::parse_file_extension_argument (argc, argv, ".pcd");
    
    if (filenames.size () != 2 || pcl::console::find_switch (argc, argv, "-h") || pcl::console::find_switch (argc, argv, "--help")){
        showHelp (argv[0]);
        return (1);
    }
  
    // Load object and scene
    pcl::console::print_highlight ("Loading point clouds...\n");
    if (pcl::io::loadPCDFile<Point3D> (argv[1], *loaded_object) < 0 ||
        pcl::io::loadPCDFile<Point3D> (argv[2], *loaded_scene) < 0)
    {
        pcl::console::print_error ("Error loading object/scene file!\n");
        return (1);
    }
    
    //~ show_cloud("escena cargada", loaded_scene);
    
    // Downsample
    pcl::console::print_highlight ("Downsampling...\n");
    pcl::VoxelGrid<Point3D> grid;
    
    float leaf = 0.005;
    if(pcl::console::find_argument(argc, argv, "-leaf") != -1){
        leaf = atof(argv[pcl::console::find_argument(argc, argv, "-leaf") + 1]);
    }
    std::cout << "leaf = " << leaf << std::endl;
    
    grid.setLeafSize (leaf, leaf, leaf);
    grid.setInputCloud (loaded_object);
    grid.filter (*filtered_object);
    grid.setInputCloud (loaded_scene);
    grid.filter (*filtered_scene);
    
    show_cloud("escena filtrada", filtered_scene);
    
    // Converting from PointXYZ to PointNormal
     pcl::console::print_highlight ("Converting...\n");
    
    object->points.resize(filtered_object->size());
    for (size_t i = 0; i < filtered_object->points.size(); i++) {
        object->points[i].x = filtered_object->points[i].x;
        object->points[i].y = filtered_object->points[i].y;
        object->points[i].z = filtered_object->points[i].z;
    }
    
    scene->points.resize(filtered_scene->size());
    for (size_t i = 0; i < filtered_scene->points.size(); i++) {
        scene->points[i].x = filtered_scene->points[i].x;
        scene->points[i].y = filtered_scene->points[i].y;
        scene->points[i].z = filtered_scene->points[i].z;
    }
    
    show_normal_cloud("escena normalizada", scene);

    // Estimate normals for object
    pcl::console::print_highlight ("Estimating object normals...\n");
    pcl::NormalEstimationOMP<PointNT,PointNT> nest;
    nest.setRadiusSearch (0.01);
    nest.setInputCloud (object);
    nest.compute (*object);

    // Estimate normals for scene
    pcl::console::print_highlight ("Estimating scene normals...\n");
    pcl::NormalEstimationOMP<PointNT,PointNT> nest2;
    nest2.setRadiusSearch (0.01);
    nest2.setInputCloud (scene);
    nest2.compute (*scene);

    // Estimate features
    pcl::console::print_highlight ("Estimating features...\n");
    FeatureEstimationT fest;
    fest.setRadiusSearch (0.025);
    fest.setInputCloud (object);
    fest.setInputNormals (object);
    fest.compute (*object_features);
    fest.setInputCloud (scene);
    fest.setInputNormals (scene);
    fest.compute (*scene_features);
  
    // Perform alignment
    pcl::console::print_highlight ("Starting alignment...\n");
    pcl::SampleConsensusPrerejective<PointNT,PointNT,FeatureT> align;
    align.setInputSource (object);
    align.setSourceFeatures (object_features);
    align.setInputTarget (scene);
    align.setTargetFeatures (scene_features);

    // Number of RANSAC iterations
    int max_it = 1000;
    if(pcl::console::find_argument(argc, argv, "-mi") != -1){
        max_it = atoi(argv[pcl::console::find_argument(argc, argv, "-mi") + 1]);
    }
    std::cout << "max. iter. = " << max_it << std::endl;
    align.setMaximumIterations (max_it);
  
  
    // Number of points to sample for generating/prerejecting a pose
    int samp_num = 3;
    if(pcl::console::find_argument(argc, argv, "-sn") != -1){
        samp_num = atoi(argv[pcl::console::find_argument(argc, argv, "-sn") + 1]);
    }
    std::cout << "number of samples = " << samp_num << std::endl;
    align.setNumberOfSamples (samp_num);//(3); 
    
    
    // Number of nearest features to use
    int feat_num = 2;
    if(pcl::console::find_argument(argc, argv, "-fn") != -1){
        feat_num = atoi(argv[pcl::console::find_argument(argc, argv, "-fn") + 1]);
    }
    std::cout << "number of features = " << feat_num << std::endl;
    align.setCorrespondenceRandomness (feat_num);
    
    
    // Polygonal edge length similarity threshold
    float sim_thr = 0.6;
    if(pcl::console::find_argument(argc, argv, "-st") != -1){
        sim_thr = atof(argv[pcl::console::find_argument(argc, argv, "-st") + 1]);
    }
    std::cout << "simil. threshold = " << sim_thr << std::endl;
    align.setSimilarityThreshold (sim_thr); 
    
    
    // Inlier threshold
    float in_thr = 1.5;
    if(pcl::console::find_argument(argc, argv, "-it") != -1){
        in_thr = atof(argv[pcl::console::find_argument(argc, argv, "-it") + 1]);
    }
    std::cout << "it = " << in_thr << std::endl;
    std::cout << "inlier threshold (it * leaf) = " << in_thr * leaf << std::endl;
    align.setMaxCorrespondenceDistance (in_thr * leaf);
    
    
    // Required inlier fraction for accepting a pose hypothesis
    float in_frac = 0.25;
    if(pcl::console::find_argument(argc, argv, "-if") != -1){
        in_frac = atof(argv[pcl::console::find_argument(argc, argv, "-if") + 1]);
    }
    std::cout << "inlier fraction = " << in_frac << std::endl;
    align.setInlierFraction (in_frac);
    
    // Compute alignment
    {
        pcl::ScopeTime t("Alignment");
        align.align (*object_aligned);
    }
  
    if (align.hasConverged ()){
        // Print results
        printf ("\n");
        Eigen::Matrix4f transformation = align.getFinalTransformation ();
        pcl::console::print_info ("    | %6.3f %6.3f %6.3f | \n", transformation (0,0), transformation (0,1), transformation (0,2));
        pcl::console::print_info ("R = | %6.3f %6.3f %6.3f | \n", transformation (1,0), transformation (1,1), transformation (1,2));
        pcl::console::print_info ("    | %6.3f %6.3f %6.3f | \n", transformation (2,0), transformation (2,1), transformation (2,2));
        pcl::console::print_info ("\n");
        pcl::console::print_info ("t = < %0.3f, %0.3f, %0.3f >\n", transformation (0,3), transformation (1,3), transformation (2,3));
        pcl::console::print_info ("\n");
        pcl::console::print_info ("Inliers: %i/%i\n", align.getInliers ().size (), object->size ());

        // Show alignment
        pcl::visualization::PCLVisualizer visu("Alignment");
        visu.addPointCloud (scene, ColorHandlerT (scene, 0.0, 255.0, 0.0), "scene");
        visu.addPointCloud (object_aligned, ColorHandlerT (object_aligned, 0.0, 0.0, 255.0), "object_aligned");
        visu.spin ();
    }
    else {
        pcl::console::print_error ("Alignment failed!\n");
        return (1);
    }

    return (0);
}

