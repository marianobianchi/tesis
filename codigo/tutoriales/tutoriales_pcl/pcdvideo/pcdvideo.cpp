//============================================================================
// Name        : pcdvideo.cpp
// Author      : Pachi
//============================================================================

//#include <pcl/io/openni_grabber.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/pcd_io.h>
//#include <boost/thread/mutex.hpp>
#include <iostream>
#include <sstream>
#include <vector>
#include <ctime>
#include <utility>

using namespace std;

#define QCLOUD_SIZE 10



std::string cloud_path;


class SimpleOpenNIViewer {
public:

	typedef pcl::PointXYZRGB PointType;
	typedef pcl::PointCloud<PointType>::Ptr CloudType;
    typedef pcl::visualization::PointCloudColorHandlerCustom<PointType> ColorHandlerType;


	SimpleOpenNIViewer() {
		pviewer = NULL;
	}
	~SimpleOpenNIViewer() {
		delete pviewer;
	}


	void run() {

        pviewer = new pcl::visualization::PCLVisualizer("PCL Cloud Viewer");
        

		struct timespec slptm;
		slptm.tv_sec = 0;
		slptm.tv_nsec = 30000000L;

		vector<pair<CloudType,CloudType> >::iterator itb = qcloud.begin();
		vector<pair<CloudType,CloudType> >::iterator ite = qcloud.end();
		vector<pair<CloudType,CloudType> >::iterator it = itb;
        
        pviewer->setBackgroundColor(255.0, 255.0, 255.0);
		while (!pviewer->wasStopped() && !qcloud.empty())
		{
			//CloudType cl = *it++;
            pair<CloudType,CloudType> cls = *it++;
            
            pviewer->addPointCloud (cls.first, ColorHandlerType (cls.first, 0.0, 0.0, 0.0), "pc");
            
            if(!cls.second->empty()){
                pviewer->addPointCloud (cls.second, ColorHandlerType (cls.second, 255.0, 0.0, 0.0), "pc2");
            }
            
            pviewer->spinOnce(10);
            
            pviewer->removePointCloud("pc");
            
            if(!cls.second->empty()){
                pviewer->removePointCloud("pc2");
            }
            
			if(it == ite) it = itb;

//			nanosleep(&slptm,NULL);
		}

	}

	void loadCloud(const string& pcd_file, pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcloud)
	{
	  if (pcl::io::loadPCDFile<pcl::PointXYZRGB> (pcd_file, *pcloud) == -1) //* load the file
	  {
		PCL_ERROR ("Couldn't read pcd file.\n");
		return;
	  }
//	  std::cout << "Loaded "
//				<< pcloud->width * pcloud->height
//				<< " data points from "
//				<< pcd_file
//				<< std::endl;
	}

	void loadClouds(const string& dir, size_t first, size_t last)
	{
		cout << "Loading " << (last - first +1) << " clouds ..." << endl;

		time_t start,end;
		double ttot = 0;
        for(size_t i=first; i <= last; i++)
		{
            

            time(&start);
            
            // Cargo primer nube
            stringstream pcd_file;
            pcd_file << dir << "/desk_1_" << i << ".pcd";
			CloudType cl(new pcl::PointCloud<pcl::PointXYZRGB>);
			loadCloud(pcd_file.str(), cl);
            
            // Cargo segunda nube si hay
            stringstream pcd_file_2;
            pcd_file_2 << dir;
            char d[100];
            sprintf(d, "/obj_found_scenepoints_frame_%.3d", i);
            pcd_file_2 << d << ".pcd";
            CloudType cl2(new pcl::PointCloud<pcl::PointXYZRGB>);
            loadCloud(pcd_file_2.str(), cl2);
            
			time(&end);
			ttot += difftime (end,start);

			cout << "Clouds left: " << (last-i) << " Time left: " << (ttot/(i-first+1)*(last-i)/60) << " min" << endl;

			qcloud.push_back(make_pair(cl, cl2));
		}
	}

  private:
	//pcl::visualization::CloudViewer* pviewer;
    pcl::visualization::PCLVisualizer* pviewer;
    
	vector<pair<CloudType,CloudType> > qcloud;
};


int main(int argc, char** argv) {

	if(argc == 2)
	{
		std::cout << "Input params:" << std::endl;
		std::cout << argv[1] << std::endl;
	}
	else
	{
        std::cerr << "Not enough params" << std::endl;
		return -1;
	}

	cout << "Starting viewer ..." << endl << endl;

	SimpleOpenNIViewer v;

	cloud_path = argv[1];
	v.loadClouds(cloud_path,1,98);

	cout << "Showing clouds ..." << endl << endl;

	v.run();

	return 0;
}

