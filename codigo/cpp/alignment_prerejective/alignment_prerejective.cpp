
#include "alignment_prerejective.h"


// Align a rigid object to a scene with clutter and occlusions
APResult alignment_prerejective(PointCloud3D::Ptr const_source_cloud,
                                PointCloud3D::Ptr target_cloud,
                                APDefaults &ap_defaults)
{   
    // Point clouds
    PointCloud3D::Ptr source_cloud (new PointCloud3D);
    *source_cloud = *const_source_cloud; // Copy original cloud to avoid unwanted changes
    
    PointCloudT::Ptr normalized_source (new PointCloudT);
    PointCloudT::Ptr normalized_target (new PointCloudT);
    
    FeatureCloudT::Ptr source_features (new FeatureCloudT);
    FeatureCloudT::Ptr target_features (new FeatureCloudT);
    
    PointCloudT::Ptr object_aligned (new PointCloudT);
    
    PointCloud3D::Ptr source_cloud_transformed (new PointCloud3D);
    *source_cloud_transformed = *source_cloud; // Copy original cloud to apply transformation later
  
    // Downsample
    if( ap_defaults.show_values ) pcl::console::print_highlight ("Downsampling...\n");
    
    pcl::VoxelGrid<Point3D> grid;
    grid.setLeafSize (ap_defaults.leaf, ap_defaults.leaf, ap_defaults.leaf);
    grid.setInputCloud (source_cloud);
    grid.filter (*source_cloud);
    grid.setInputCloud (target_cloud);
    grid.filter (*target_cloud);
    
    // Converting from PointXYZ to PointNormal
    if( ap_defaults.show_values ) pcl::console::print_highlight ("Converting...\n");
    
    normalized_source->points.resize(source_cloud->size());
    for (size_t i = 0; i < source_cloud->points.size(); i++) {
        normalized_source->points[i].x = source_cloud->points[i].x;
        normalized_source->points[i].y = source_cloud->points[i].y;
        normalized_source->points[i].z = source_cloud->points[i].z;
    }
    
    normalized_target->points.resize(target_cloud->size());
    for (size_t i = 0; i < target_cloud->points.size(); i++) {
        normalized_target->points[i].x = target_cloud->points[i].x;
        normalized_target->points[i].y = target_cloud->points[i].y;
        normalized_target->points[i].z = target_cloud->points[i].z;
    }

    // Estimate normals for source
    if( ap_defaults.show_values ) pcl::console::print_highlight ("Estimating object normals...\n");
    pcl::NormalEstimationOMP<PointNT,PointNT> nest;
    nest.setRadiusSearch (0.01);
    nest.setInputCloud (normalized_source);
    nest.compute (*normalized_source);

    // Estimate normals for target
    if( ap_defaults.show_values ) pcl::console::print_highlight ("Estimating target normals...\n");
    pcl::NormalEstimationOMP<PointNT,PointNT> nest2;
    nest2.setRadiusSearch (0.01);
    nest2.setInputCloud (normalized_target);
    nest2.compute (*normalized_target);

    // Estimate features
    if( ap_defaults.show_values ) pcl::console::print_highlight ("Estimating features...\n");
    FeatureEstimationT fest;
    fest.setRadiusSearch (0.025);
    fest.setInputCloud (normalized_source);
    fest.setInputNormals (normalized_source);
    fest.compute (*source_features);
    fest.setInputCloud (normalized_target);
    fest.setInputNormals (normalized_target);
    fest.compute (*target_features);
  
    // Perform alignment
    if( ap_defaults.show_values ) pcl::console::print_highlight ("Starting alignment...\n");
    pcl::SampleConsensusPrerejective<PointNT,PointNT,FeatureT> align;
    align.setInputSource (normalized_source);
    align.setSourceFeatures (source_features);
    align.setInputTarget (normalized_target);
    align.setTargetFeatures (target_features);

    // Number of RANSAC iterations
    align.setMaximumIterations (ap_defaults.max_ransac_iters);
  
    // Number of points to sample for generating/prerejecting a pose
    align.setNumberOfSamples (ap_defaults.points_to_sample);
    
    // Number of nearest features to use
    align.setCorrespondenceRandomness (ap_defaults.nearest_features_used);
    
    // Polygonal edge length similarity threshold
    align.setSimilarityThreshold (ap_defaults.simil_threshold); 
    
    // Inlier threshold
    align.setMaxCorrespondenceDistance (ap_defaults.inlier_threshold * ap_defaults.leaf);
    
    // Required inlier fraction for accepting a pose hypothesis
    align.setInlierFraction (ap_defaults.inlier_fraction);
    
    // Compute alignment
    align.align (*object_aligned);
    
    APResult ap_result;
  
    if (align.hasConverged ()){
        Eigen::Matrix4f transformation = align.getFinalTransformation ();
        pcl::transformPointCloud (*source_cloud_transformed, *source_cloud_transformed, transformation);
        ap_result.cloud = source_cloud_transformed;
    }
    ap_result.has_converged = align.hasConverged();
    
    if(ap_defaults.show_values){
        Eigen::Matrix4f transformation = align.getFinalTransformation ();
        std::cout << "leaf = " << ap_defaults.leaf << std::endl;
        std::cout << "max. iter. = " << ap_defaults.max_ransac_iters << std::endl;
        std::cout << "number of samples = " << ap_defaults.points_to_sample << std::endl;
        std::cout << "number of features = " << ap_defaults.nearest_features_used << std::endl;
        std::cout << "simil. threshold = " << ap_defaults.simil_threshold << std::endl;
        std::cout << "it = " << ap_defaults.inlier_threshold << std::endl;
        std::cout << "inlier threshold (it * leaf) = " << ap_defaults.inlier_threshold * ap_defaults.leaf << std::endl;
        std::cout << "inlier fraction = " << ap_defaults.inlier_fraction << std::endl;
        
        // Print results
        printf ("\n");
        pcl::console::print_info ("    | %6.3f %6.3f %6.3f | \n", transformation (0,0), transformation (0,1), transformation (0,2));
        pcl::console::print_info ("R = | %6.3f %6.3f %6.3f | \n", transformation (1,0), transformation (1,1), transformation (1,2));
        pcl::console::print_info ("    | %6.3f %6.3f %6.3f | \n", transformation (2,0), transformation (2,1), transformation (2,2));
        pcl::console::print_info ("\n");
        pcl::console::print_info ("t = < %0.3f, %0.3f, %0.3f >\n", transformation (0,3), transformation (1,3), transformation (2,3));
        pcl::console::print_info ("\n");
        pcl::console::print_info ("Inliers: %i/%i\n", align.getInliers ().size (), normalized_source->size ());
        
        // Show alignment
        if(ap_result.has_converged){
            pcl::visualization::PCLVisualizer visu("Alignment");
            pcl::console::print_info("Verde = escena\n");
            pcl::console::print_info("Azul = modelo alineado\n");
            visu.addPointCloud (normalized_target, ColorHandlerT (normalized_target, 0.0, 255.0, 0.0), "scene");
            visu.addPointCloud (ap_result.cloud, ColorHandler3D (ap_result.cloud, 0.0, 0.0, 255.0), "object_aligned");
            visu.spin ();
        }
    }

    return ap_result;
}
