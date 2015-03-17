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

using namespace std;

#define QCLOUD_SIZE 10

typedef pcl::PointXYZRGB Point3D;
typedef pcl::visualization::PointCloudColorHandlerCustom<Point3D> ColorHandler3D;

std::string cloud_path;


class SimpleOpenNIViewer {
public:

	typedef pcl::PointXYZRGB PointType;
	typedef pcl::PointCloud<PointType>::Ptr CloudType;


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

		vector<CloudType>::iterator itb = qcloud.begin();
		vector<CloudType>::iterator ite = qcloud.end();
		vector<CloudType>::iterator it = itb;
        
        pviewer->setBackgroundColor(255.0, 255.0, 255.0);
		while (!pviewer->wasStopped() && !qcloud.empty())
		{
			CloudType cl = *it++;
            
            pviewer->addPointCloud (cl, ColorHandler3D (cl, 0.0, 0.0, 0.0), "pc");
            pviewer->spinOnce(10);
            
            pviewer->removePointCloud("pc");
            
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
            stringstream pcd_file;
            pcd_file << dir << "/desk_1_" << i << ".pcd";

            time(&start);
			CloudType cl(new pcl::PointCloud<pcl::PointXYZRGB>);
			loadCloud(pcd_file.str(), cl);
			time(&end);
			ttot += difftime (end,start);

			cout << "Clouds left: " << (last-i) << " Time left: " << (ttot/(i-first+1)*(last-i)/60) << " min" << endl;

			qcloud.push_back(cl);
		}
	}

  private:
	//pcl::visualization::CloudViewer* pviewer;
    pcl::visualization::PCLVisualizer* pviewer;
    
	vector<CloudType> qcloud;
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
	v.loadClouds(cloud_path,1,5);

	cout << "Showing clouds ..." << endl << endl;

	v.run();

	return 0;
}

