#include "cv_define.h"

#ifdef MAIN_WIL

#include <sstream>
#include <string>

#include "cv_utilities.h"
#include "cv_homography.h"
#include "cv_camera.h"
#include "cv_harris.h"
#include "cv_pcabase.h"
#include "cv_gepard.h"
#include "cv_hyper.h"
#include "cv_esm.h"

#include "ippi.h"
#include "ippcv.h"


void  draw_patch( IplImage * ap_image, CvMat * ap_image_points )
{
	CvPoint l_pt0;
	CvPoint l_pt1;
	CvPoint l_pt2;
	CvPoint l_pt3;

	l_pt0.x = CV_MAT_ELEM(*ap_image_points,float,0,0);
	l_pt0.y = CV_MAT_ELEM(*ap_image_points,float,1,0);
	l_pt1.x = CV_MAT_ELEM(*ap_image_points,float,0,1);
	l_pt1.y = CV_MAT_ELEM(*ap_image_points,float,1,1);
	l_pt2.x = CV_MAT_ELEM(*ap_image_points,float,0,2);
	l_pt2.y = CV_MAT_ELEM(*ap_image_points,float,1,2);
	l_pt3.x = CV_MAT_ELEM(*ap_image_points,float,0,3);
	l_pt3.y = CV_MAT_ELEM(*ap_image_points,float,1,3);
	
	CvPoint l_point_vector[5] = {l_pt0,l_pt1,l_pt2,l_pt3,l_pt0};
	cvFillConvexPoly(ap_image,l_point_vector,5,CV_RGB(170,170,0));

	cv::cv_draw_poly(ap_image,ap_image_points,4,0,0,0);
	
	cvDrawLine(ap_image,l_pt0,l_pt1,cvScalar(255,255,255,255),2);
	cvDrawLine(ap_image,l_pt1,l_pt2,cvScalar(0,0,255,255),2);
	cvDrawLine(ap_image,l_pt2,l_pt3,cvScalar(0,255,0,255),2);
	cvDrawLine(ap_image,l_pt3,l_pt0,cvScalar(255,0,0,255),2);

	return;
}

int main( int argc, char * argv[] )
{
	std::string l_base_name    = "./base/basis";

	CvMat * lp_k = cvCreateMat(3,3,CV_32FC1);

	CV_MAT_ELEM(*lp_k,float,0,0) = 1073.20; 
	CV_MAT_ELEM(*lp_k,float,0,1) = 0; 
	CV_MAT_ELEM(*lp_k,float,0,2) = 320; 
	CV_MAT_ELEM(*lp_k,float,1,0) = 0; 
	CV_MAT_ELEM(*lp_k,float,1,1) = 1075.92;
	CV_MAT_ELEM(*lp_k,float,1,2) = 240; 
	CV_MAT_ELEM(*lp_k,float,2,0) = 0; 
	CV_MAT_ELEM(*lp_k,float,2,1) = 0; 
	CV_MAT_ELEM(*lp_k,float,2,2) = 1; 
	
	cv::cv_pcabase l_pcabase;

	int l_num_of_gep_trains = 300;
	int l_num_of_hyp_levels = 4;
	int l_gep_nx			= 11;
	int l_gep_ny			= 11;
	int l_hyp_nx			= 10;
	int l_hyp_ny			= 10;
	int l_num_of_pcas		= 600;
	int l_sup_size			= 120;
	int l_pat_size			= 71;
	int l_max_motion		= 23;

	std::cerr << "try to load the pca basis:";

	if( l_pcabase.load(l_base_name) == false )
	{
		std::cerr << " failed! " << std::endl;
		std::cerr << "- a pca base is not available yet..." << std::endl;
		std::cerr << "- we will learn one now for once and for all..." << std::endl;

		l_pcabase.set_parameters(lp_k,l_pat_size,l_sup_size,l_gep_nx,l_gep_ny,l_num_of_gep_trains);	
			
		int l_learn_counter = 0;

		cv::cv_harris::set_radius(4);
		cv::cv_harris::set_num_of_points(300);
		
		for( int l_i=1; l_i<=10; ++l_i )
		{
			std::stringstream l_string;

			l_string << "./base/img" << l_i << ".jpg";

			IplImage * lp_color_image = cv::cv_load_image(l_string.str());
			IplImage * lp_gray_image = cv::cv_convert_color_to_gray(lp_color_image);
			IplImage * lp_image = cv::cv_smooth(lp_gray_image,5);
			cvReleaseImage(&lp_color_image);
			cvReleaseImage(&lp_gray_image);
			cv::cv_show_image(lp_image);

			CvMat * lp_points = cv::cv_harris::get_points(lp_image);

			if( lp_points == NULL )
			{
				cvReleaseImage(&lp_image);
					
				printf("error: lp_points is NULL!");
				return 0;
			}
			IplImage * lp_mask = cvCreateImage(cvGetSize(lp_image),IPL_DEPTH_8U,1);

			cvSet(lp_mask,cvRealScalar(1));

			for( int l_j=0; l_j<lp_points->cols; ++l_j )
			{
				int l_col = CV_MAT_ELEM(*lp_points,float,0,l_j);
				int l_row = CV_MAT_ELEM(*lp_points,float,1,l_j);
				
				if( l_col - l_pcabase.get_support_size()  > 0 &&
					l_row - l_pcabase.get_support_size()  > 0 &&
					l_col + l_pcabase.get_support_size()  < lp_image->width &&
					l_row + l_pcabase.get_support_size()  < lp_image->height )
				{
					++l_learn_counter;
					l_pcabase.add( lp_image, lp_mask, l_row, l_col );
				}
			}
			cvReleaseImage(&lp_image);
			cvReleaseImage(&lp_mask);

			std::cerr << l_learn_counter << " patches collected..." << std::endl;
		}
		std::cerr << "- start learning the pca basis (takes some minutes)..." << std::endl;

		if( l_pcabase.learn_base(l_num_of_pcas) == false )
		{
			std::cerr << "error: learning the base!" << std::endl;
			return 0;
		}
		std::cerr << "- finished learning the base... "  << std::endl;
		std::cerr << "try to save the base: ";
		
		if( l_pcabase.save(l_base_name) == false )
		{
			std::cerr << "error: pca base could not be saved... " << std::endl;
		}
		else
		{
			std::cerr << " successfully saved..." << std::endl;
		}
	}	
	else
	{	
		std::cerr << "successfull!" << std::endl;
	}
	std::vector<cv::cv_gepard*> l_gepard; 
	std::vector<cv::cv_hyper*>	l_hyper;
	std::vector<cv::cv_esm*>    l_esm;

	bool l_one_break = false;
	bool l_esm_break = true;
	bool l_pat_break = true;
	bool l_fps_break = true;
	
	float l_fps = 15;

	cv::cv_create_window("runtime");
	cv::cv_camera l_camera;
	cv::cv_timer l_timer1;
	cv::cv_mouse l_mouse;
	
	l_camera.set_cam(cv::usb);
	
	if( l_camera.start_capture_from_cam() == false )
	{
		printf("main_wil: the camera could not be initalized!");
		cvReleaseMat(&lp_k);
		return 0;
	}
	l_mouse.start("runtime");

	int l_tra_col1=0;
	int l_tra_row1=0;

	bool l_learn_flag1 = false;
	bool l_learn_flag2 = false;

	int l_counter=0;

	while(true)
	{
		l_timer1.start(true);

		IplImage * lp_color = NULL;
		IplImage * lp_gray  = NULL;
		IplImage * lp_mean  = NULL;

		lp_color = l_camera.get_image();
		lp_gray  = cv::cv_convert_color_to_gray(lp_color);
		lp_mean	 = cv::cv_smooth(lp_gray,5);
		
		int l_inlier=0;
		int l_num_of_points=0;
		int l_point_num=100;

		CvMat * lp_points = NULL;

		cv::cv_harris::set_radius(4);
		cv::cv_harris::set_num_of_points(l_point_num);
		lp_points = cv::cv_harris::get_points(lp_mean);

		if( lp_points == NULL )
		{
			cv::cv_show_image(lp_color,"runtime");

			cvWaitKey(1);
			cvReleaseImage(&lp_mean);
			cvReleaseImage(&lp_gray);
			cvReleaseImage(&lp_color);
			continue;
		}
		l_num_of_points = lp_points->cols;

		cv::cv_draw_points(lp_color,lp_points,3,0,0,255,l_point_num);
		cv::cv_draw_points(lp_color,lp_points,3,255,0,0,l_point_num/2);
		
		int l_x=l_mouse.get_x();
		int l_y=l_mouse.get_y();
		int l_e=l_mouse.get_event();

		if( l_x != -1 && l_y != -1 )
		{
			l_tra_col1 = l_x;
			l_tra_row1 = l_y;
		}
		for( int l_i=0; l_i<lp_points->cols; ++l_i )
		{
			int l_col = CV_MAT_ELEM(*lp_points,float,0,l_i);
			int l_row = CV_MAT_ELEM(*lp_points,float,1,l_i);

			if( l_col-l_pcabase.get_support_size()/2 < 0 ||
				l_row-l_pcabase.get_support_size()/2 < 0 ||
				l_col+l_pcabase.get_support_size()/2 > lp_mean->width-1 ||
				l_row+l_pcabase.get_support_size()/2 > lp_mean->height-1 )
			{
				continue;
			}
			if( sqrt(static_cast<double>(SQR(l_col-l_tra_col1)+SQR(l_row-l_tra_row1))) < 10 )
			{
				CvPoint l_pt1;
				CvPoint l_pt2;
				CvPoint l_pt3;
				CvPoint l_pt4;

				l_pt1.x = l_tra_col1-l_pcabase.get_patch_size()/2;
				l_pt1.y = l_tra_row1-l_pcabase.get_patch_size()/2;
				l_pt2.x = l_tra_col1+l_pcabase.get_patch_size()/2;
				l_pt2.y = l_tra_row1-l_pcabase.get_patch_size()/2;
				l_pt3.x = l_tra_col1+l_pcabase.get_patch_size()/2;
				l_pt3.y = l_tra_row1+l_pcabase.get_patch_size()/2;
				l_pt4.x = l_tra_col1-l_pcabase.get_patch_size()/2;
				l_pt4.y = l_tra_row1+l_pcabase.get_patch_size()/2;

				cvLine(lp_color,l_pt1,l_pt2,CV_RGB(0,0,0),3);
				cvLine(lp_color,l_pt2,l_pt3,CV_RGB(0,0,0),3);
				cvLine(lp_color,l_pt3,l_pt4,CV_RGB(0,0,0),3);
				cvLine(lp_color,l_pt4,l_pt1,CV_RGB(0,0,0),3);

				cvLine(lp_color,l_pt1,l_pt2,CV_RGB(255,0,0),1);
				cvLine(lp_color,l_pt2,l_pt3,CV_RGB(255,0,0),1);
				cvLine(lp_color,l_pt3,l_pt4,CV_RGB(255,0,0),1);
				cvLine(lp_color,l_pt4,l_pt1,CV_RGB(255,0,0),1);
			}
		}			
		if(  l_x != -1 && l_y != -1 && l_e == 2 )
		{
			float	l_best_distance = 10e10;
			int		l_best_col=-1;
			int		l_best_row=-1;

			for( int l_i=0; l_i<lp_points->cols; ++l_i )
			{
				int l_col = CV_MAT_ELEM(*lp_points,float,0,l_i);
				int l_row = CV_MAT_ELEM(*lp_points,float,1,l_i);

				if( l_col-l_pcabase.get_support_size()/2 < 0 ||
					l_row-l_pcabase.get_support_size()/2 < 0 ||
					l_col+l_pcabase.get_support_size()/2 > lp_mean->width-1 ||
					l_row+l_pcabase.get_support_size()/2 > lp_mean->height-1 )
				{
					continue;
				}
				float l_distance = sqrt(static_cast<double>(SQR(l_col-l_x)+SQR(l_row-l_y)));

				if( l_distance < 10 &&
					l_distance < l_best_distance )
				{
					l_best_distance = l_distance;
					l_best_row = l_row;
					l_best_col = l_col;
				}
			}
			if( l_best_col != -1 && l_best_row != -1 )
			{
				//cv::cv_timer l_timer1;
				//l_timer1.start();
				
				CvMat * lp_rec = cvCreateMat(3,4,CV_32FC1);
				cvSet(lp_rec,cvRealScalar(1));

				CV_MAT_ELEM(*lp_rec,float,0,0) = l_best_col-l_pat_size/2;
				CV_MAT_ELEM(*lp_rec,float,1,0) = l_best_row-l_pat_size/2;
				CV_MAT_ELEM(*lp_rec,float,0,1) = l_best_col+l_pat_size/2;
				CV_MAT_ELEM(*lp_rec,float,1,1) = l_best_row-l_pat_size/2;
				CV_MAT_ELEM(*lp_rec,float,0,2) = l_best_col+l_pat_size/2;
				CV_MAT_ELEM(*lp_rec,float,1,2) = l_best_row+l_pat_size/2;
				CV_MAT_ELEM(*lp_rec,float,0,3) = l_best_col-l_pat_size/2;
				CV_MAT_ELEM(*lp_rec,float,1,3) = l_best_row+l_pat_size/2;

				cv::cv_gepard * lp_gepard = l_pcabase.get_tracker(lp_mean,599,l_best_row,l_best_col);
				cv::cv_hyper * lp_hyper = new cv::cv_hyper;
				cv::cv_esm * lp_esm = new cv::cv_esm;
			
				lp_hyper->set_parameters(10,10,4,l_max_motion,600,0);//23,300
				lp_hyper->learn(lp_mean,lp_rec);
	
				lp_esm->learn(lp_mean,l_best_row-l_pat_size/2,l_best_col-l_pat_size/2,l_pat_size,l_pat_size);
	
				cvReleaseMat(&lp_rec);

				if( lp_gepard != NULL && lp_hyper != NULL && lp_esm != NULL )
				{
					l_gepard.push_back(lp_gepard);
					l_hyper.push_back(lp_hyper);
					l_esm.push_back(lp_esm);
				}
				else
				{
					delete lp_gepard;
					delete lp_hyper;
					delete lp_esm;
				}
				//l_timer1.stop();
				
				l_learn_flag1 = true;
				l_learn_flag2 = true;
			}
		}
		if( l_learn_flag1 == true )
		{
			cvReleaseImage(&lp_mean);
			cvReleaseImage(&lp_gray);
			cvReleaseImage(&lp_color);
			cvReleaseMat(&lp_points);
			l_learn_flag1 = false;
			continue;
		}
		l_num_of_points = lp_points->cols;

		CvMat * lp_best = cvCreateMat(3,4,CV_32FC1);
		float l_best_ncc = -1.0f;
		int l_best_ind = -1;

		std::stringstream l_data;

		for( int l_i=0; l_i<l_gepard.size(); ++l_i )
		{
			for( int l_j=0; l_j<lp_points->cols; ++l_j )
			{
				if( CV_MAT_ELEM(*lp_points,float,1,l_j)-l_pat_size/2-1 < 0 ||
					CV_MAT_ELEM(*lp_points,float,0,l_j)-l_pat_size/2-1 < 0 ||
					CV_MAT_ELEM(*lp_points,float,1,l_j)+l_pat_size/2+1 >= lp_mean->height ||
					CV_MAT_ELEM(*lp_points,float,0,l_j)+l_pat_size/2+1 >= lp_mean->width  )
				{
					continue;
				}
				float l_hyp_ncc = 0.80;
				float l_esm_ncc = 0.90;

				CvMat * lp_quad = NULL;
	
				lp_quad = l_gepard[l_i]->recognize(	lp_mean,
													CV_MAT_ELEM(*lp_points,float,1,l_j),
													CV_MAT_ELEM(*lp_points,float,0,l_j) );
				if( lp_quad == NULL )
					continue;
				
				CvMat * lp_result = l_hyper[l_i]->track(lp_mean,lp_quad,3);

				if( lp_result == NULL )
				{
					cvReleaseMat(&lp_quad);				
					continue;
				}
				if( l_hyper[l_i]->get_ncc() < l_hyp_ncc  )
				{
					cvReleaseMat(&lp_quad);
					continue;
				}
				if( cv::cv_homography_heuristic(lp_result,0.7,5) == false )
				{
					cvReleaseMat(&lp_result);
					cvReleaseMat(&lp_quad);				
					continue;
				}	
				CvMat * lp_result1 = l_esm[l_i]->track(lp_mean,lp_result,10,1);
				cvCopy(lp_result1,lp_result);
				cvReleaseMat(&lp_result1);
			
				if( l_esm[l_i]->get_ncc() < l_esm_ncc )
				{
					cvReleaseMat(&lp_result);
					cvReleaseMat(&lp_quad);				
					continue;
				}
				if( cv::cv_homography_heuristic(lp_result,0.7,5) == false )
				{
					cvReleaseMat(&lp_result);
					cvReleaseMat(&lp_quad);				
					continue;
				}
				++l_inlier;

				draw_patch(lp_color,lp_result);

				cvReleaseMat(&lp_result);
				cvReleaseMat(&lp_quad);

				break; 
			}
			if( l_one_break == true && l_inlier == 1 ) 
				break;
		}
		l_timer1.stop();

		if( l_learn_flag2 == true )
		{
			CvFont l_font;
			std::stringstream l_stringstream;
			cvInitFont(&l_font,CV_FONT_HERSHEY_DUPLEX,0.7,0.7);
			l_stringstream << "PATCH LEARNED" << std::endl;
			cvPutText(lp_color,l_stringstream.str().c_str(),cvPoint(30,lp_color->height-60),&l_font,CV_RGB(255,0,0));
			l_learn_flag2 = false;
			cvAddS(lp_color,cvScalar(0,0,255,0),lp_color);
		}
		CvFont l_font;
		std::stringstream l_stringstream;
		cvInitFont(&l_font,CV_FONT_HERSHEY_DUPLEX,0.7,0.7);
		l_stringstream << l_timer1.get_fps() << " fps";
		cvPutText(lp_color,l_stringstream.str().c_str(),cvPoint(30,lp_color->height-30),&l_font,CV_RGB(0,255,0));
					
		cv::cv_show_image(lp_color,"runtime");
		
		std::cout << "fps: " << l_timer1.get_fps() << " num_of_harris: " << l_num_of_points << " inlier: " << l_inlier << "/" << l_gepard.size() << char(13) << std::flush;

		int l_key = cvWaitKey(1);
		
		if( l_key == 111 )
		{
			l_one_break = !l_one_break;
		}	
		if( l_key == 27 )
		{
			break;
		}
		if( l_key == 100 )
		{
			for( int l_i=0; l_i<l_gepard.size(); ++l_i )
			{
				delete l_gepard[l_i];
				delete l_hyper[l_i];
				delete l_esm[l_i];
			}
			l_gepard.clear();
			l_hyper.clear();
			l_esm.clear();
		}
		cvReleaseImage(&lp_mean);
		cvReleaseImage(&lp_gray);
		cvReleaseImage(&lp_color);
		cvReleaseMat(&lp_points);
	}
	l_camera.stop_capture_from_cam();

	for( int l_i=0; l_i<l_gepard.size(); ++l_i )
	{
		delete l_gepard[l_i];
		delete l_hyper[l_i];
		delete l_esm[l_i];
	}
	return 0;
}

#endif


