#include "alignment_prerejective.h"


// Align a rigid object to a scene with clutter and occlusions
APResult alignment_prerejective(PointCloud3D::Ptr const_source_cloud,
                                PointCloud3D::Ptr const_target_cloud,
                                APDefaults &ap_defaults)
{
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
     *                    to the nearest scene point or not.
     * Inlier fraction - setInlierFraction (): In many practical scenarios, large parts of the observed
     *                   object in the scene are not visible, either due to clutter, occlusions or both.
     *                   In such cases, we need to allow for pose hypotheses that do not align all
     *                   object points to the scene. The absolute number of correctly aligned points is
     *                   determined using the inlier threshold, and if the ratio of this number to the
     *                   total number of points in the object is higher than the specified inlier
     *                   fraction, we accept a pose hypothesis as valid.

     * */


    // Point clouds
    PointCloud3D::Ptr source_cloud (new PointCloud3D);
    *source_cloud = *const_source_cloud; // Copy original cloud to avoid unwanted changes

    PointCloud3D::Ptr target_cloud (new PointCloud3D);
    *target_cloud = *const_target_cloud; // Copy original cloud to avoid unwanted changes

    PointCloudT::Ptr normalized_source (new PointCloudT);
    PointCloudT::Ptr normalized_target (new PointCloudT);
    
    PointCloudT::Ptr filtered_source (new PointCloudT);
    PointCloudT::Ptr filtered_target (new PointCloudT);

    FeatureCloudT::Ptr source_features (new FeatureCloudT);
    FeatureCloudT::Ptr target_features (new FeatureCloudT);

    PointCloudT::Ptr object_aligned (new PointCloudT);
    PointCloud3D::Ptr source_cloud_transformed (new PointCloud3D);
    
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

    // Downsample
    if( ap_defaults.show_values ) pcl::console::print_highlight ("Downsampling...\n");

    pcl::VoxelGrid<PointNT> grid;
    grid.setLeafSize (ap_defaults.leaf, ap_defaults.leaf, ap_defaults.leaf);
    grid.setInputCloud (normalized_source);
    grid.filter (*filtered_source);
    grid.setInputCloud (normalized_target);
    grid.filter (*filtered_target);

    // Estimate features
    if( ap_defaults.show_values ) pcl::console::print_highlight ("Estimating features...\n");
    FeatureEstimationT fest;
    fest.setRadiusSearch (0.025);
    fest.setInputCloud (filtered_source);
    fest.setInputNormals (filtered_source);
    fest.compute (*source_features);
    fest.setInputCloud (filtered_target);
    fest.setInputNormals (filtered_target);
    fest.compute (*target_features);

    // Perform alignment
    if( ap_defaults.show_values ) pcl::console::print_highlight ("Starting alignment...\n");

    APResult ap_result;
    ap_result.has_converged = false;
    ap_result.score = 1e10;

    pcl::SampleConsensusPrerejective<PointNT,PointNT,FeatureT> align;
    align.setInputSource (filtered_source);
    align.setSourceFeatures (source_features);
    align.setInputTarget (filtered_target);
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
    if(ap_defaults.show_values){
        // Count time
        pcl::ScopeTime t("Alignment");
        align.align (*object_aligned);
    }
    else{
        align.align (*object_aligned);
    }
    
    
    // Set result values
    if (align.hasConverged ()){
        Eigen::Matrix4f transformation = align.getFinalTransformation ();
        pcl::transformPointCloud (*const_source_cloud, *source_cloud_transformed, transformation);
        ap_result.cloud = source_cloud_transformed;
    }
    ap_result.has_converged = align.hasConverged();
    ap_result.score = align.getFitnessScore();
    

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
        pcl::console::print_info ("Inliers: %i/%i\n", align.getInliers ().size (), filtered_source->size ());
        pcl::console::print_info ("Score: %f\n", align.getFitnessScore());

        // Show alignment
        if(ap_result.has_converged){
            pcl::visualization::PCLVisualizer visu("Alignment");
            pcl::console::print_info("Verde = escena\n");
            pcl::console::print_info("Azul = modelo alineado\n");
            visu.addPointCloud (filtered_target, ColorHandlerT (filtered_target, 0.0, 255.0, 0.0), "scene");
            visu.addPointCloud (ap_result.cloud, ColorHandler3D (ap_result.cloud, 0.0, 0.0, 255.0), "object_aligned");
            visu.spin ();
        }
    }

    return ap_result;
}
