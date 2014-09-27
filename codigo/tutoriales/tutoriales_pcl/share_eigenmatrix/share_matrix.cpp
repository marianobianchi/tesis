#include "share_matrix.h"


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


VectorMat get_transformation(){
    Mat m = Mat::Identity();
    m(0,3) = 2.0; // aplico traslacion en el eje .. x?
    return mat_to_vector(m);
}

void transform(VectorMat s){
    
    Mat m = vector_to_mat(s);
    
    PointCloud3D::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ> ());

    if (pcl::io::loadPCDFile("cloud.pcd", *cloud) < 0){
        std::cout << "Error al cargar el pcd" << std::endl;
        return;
    }
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud (new pcl::PointCloud<pcl::PointXYZ> ());
	pcl::transformPointCloud (*cloud, *transformed_cloud, m);
    
    pcl::io::savePCDFileBinary("transformed_cloud.pcd", *transformed_cloud);

}
