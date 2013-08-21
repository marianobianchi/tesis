#include "cv_define.h"

#ifdef MAIN_LEO

#include <sstream>
#include <string>

#include "cv_utilities.h"
#include "cv_homography.h"
#include "cv_camera.h"
#include "cv_harris.h"
#include "cv_leopar.h"
#include "cv_hyper.h"
#include "cv_esm.h"

#define LEARN

int main( int argc, char * argv[] )
{
	std::string l_image_name  = "./pics/ICCV.png";
	std::string l_track_name  = "./data/leopar";
	std::string l_leopar_name = "./data/leopar"; 

	int l_size = 71;

	int l_pt_roi_height = 92;
	int l_pt_roi_width = 162;
	int l_pt_roi_col = 220;
	int l_pt_roi_row = 200;

	CvMat * lp_roi_pt = cvCreateMat(3,4,CV_32FC1);
	CV_MAT_ELEM(*lp_roi_pt,float,0,0) = l_pt_roi_col;
	CV_MAT_ELEM(*lp_roi_pt,float,1,0) = l_pt_roi_row;
	CV_MAT_ELEM(*lp_roi_pt,float,0,1) = l_pt_roi_col+l_pt_roi_width;
	CV_MAT_ELEM(*lp_roi_pt,float,1,1) = l_pt_roi_row;
	CV_MAT_ELEM(*lp_roi_pt,float,0,2) = l_pt_roi_col+l_pt_roi_width;
	CV_MAT_ELEM(*lp_roi_pt,float,1,2) = l_pt_roi_row+l_pt_roi_height;
	CV_MAT_ELEM(*lp_roi_pt,float,0,3) = l_pt_roi_col;
	CV_MAT_ELEM(*lp_roi_pt,float,1,3) = l_pt_roi_row+l_pt_roi_height;
	CV_MAT_ELEM(*lp_roi_pt,float,2,0) = 1;
	CV_MAT_ELEM(*lp_roi_pt,float,2,1) = 1;
	CV_MAT_ELEM(*lp_roi_pt,float,2,2) = 1;
	CV_MAT_ELEM(*lp_roi_pt,float,2,3) = 1;

	int l_re_roi_col = 180;
	int l_re_roi_row = 60;
	int l_re_roi_height = 390;
	int l_re_roi_width  = 260-30;

	CvMat * lp_roi_re = cvCreateMat(3,4,CV_32FC1);
	CV_MAT_ELEM(*lp_roi_re,float,0,0) = l_re_roi_col;
	CV_MAT_ELEM(*lp_roi_re,float,1,0) = l_re_roi_row;
	CV_MAT_ELEM(*lp_roi_re,float,0,1) = l_re_roi_col+l_re_roi_width;
	CV_MAT_ELEM(*lp_roi_re,float,1,1) = l_re_roi_row;
	CV_MAT_ELEM(*lp_roi_re,float,0,2) = l_re_roi_col+l_re_roi_width;
	CV_MAT_ELEM(*lp_roi_re,float,1,2) = l_re_roi_row+l_re_roi_height;
	CV_MAT_ELEM(*lp_roi_re,float,0,3) = l_re_roi_col;
	CV_MAT_ELEM(*lp_roi_re,float,1,3) = l_re_roi_row+l_re_roi_height;
	CV_MAT_ELEM(*lp_roi_re,float,2,0) = 1;
	CV_MAT_ELEM(*lp_roi_re,float,2,1) = 1;
	CV_MAT_ELEM(*lp_roi_re,float,2,2) = 1;
	CV_MAT_ELEM(*lp_roi_re,float,2,3) = 1;

	IplImage * lp_imag = cv::cv_load_image(l_image_name);
	cv::cv_show_image(lp_imag);
	IplImage * lp_pt_mask = cv::cv_roi_mask(lp_imag,lp_roi_pt);
	cv::cv_show_image(lp_pt_mask);
	IplImage * lp_re_mask = cv::cv_roi_mask(lp_imag,lp_roi_re);
	cv::cv_show_image(lp_re_mask);
	IplImage * lp_gray = cv::cv_convert_color_to_gray(lp_imag);
	cv::cv_show_image(lp_gray);
	IplImage * lp_mean = cv::cv_smooth(lp_gray,5);
	cv::cv_show_image(lp_mean);

	cv::cv_harris::set_num_of_points(6);
	cv::cv_harris::set_radius(10);
	CvMat * lp_rob = cv::cv_harris::get_most_robust_points(lp_mean,lp_pt_mask);

	if( lp_rob == NULL )
	{
		cvReleaseImage(&lp_gray);
		cvReleaseImage(&lp_imag);
		cvReleaseImage(&lp_mean);
		cvReleaseImage(&lp_re_mask);
		cvReleaseImage(&lp_pt_mask);
		cvReleaseMat(&lp_roi_pt);
		cvReleaseMat(&lp_roi_re);

		printf("main_leo: lp_rob is NULL!");
		return 0;
	}
	IplImage * lp_exp = cvCreateImage(cvGetSize(lp_gray),IPL_DEPTH_32F,1);
	cvCopy(lp_gray,lp_exp);

	for( int l_i=0; l_i<lp_rob->cols; ++l_i )
	{
		std::stringstream l_stringstream;
		CvFont l_font;

		cvInitFont(	&l_font, 
					CV_FONT_HERSHEY_DUPLEX,
					1.0,1.0);

		l_stringstream << l_i;

		cvPutText(	lp_exp,
					l_stringstream.str().c_str(),
					cvPoint(CV_MAT_ELEM(*lp_rob,float,0,l_i),
							CV_MAT_ELEM(*lp_rob,float,1,l_i) ),
					&l_font,
					cvScalar(100,100,100,100));
	}
	cv::cv_show_points(lp_exp,lp_rob,3,0,0,0);
	cvReleaseImage(&lp_exp);

	cv::cv_leopar * lp_leopar = new cv::cv_leopar[lp_rob->cols];
	cv::cv_hyper * lp_hyper = new cv::cv_hyper[lp_rob->cols];
	cv::cv_esm l_esm;

	int l_esm_col = CV_MAT_ELEM(*lp_roi_pt,float,0,0);
	int l_esm_row = CV_MAT_ELEM(*lp_roi_pt,float,1,0);
	int l_esm_height = CV_MAT_ELEM(*lp_roi_pt,float,1,2)-CV_MAT_ELEM(*lp_roi_pt,float,1,1);
	int l_esm_width  = CV_MAT_ELEM(*lp_roi_pt,float,0,1)-CV_MAT_ELEM(*lp_roi_pt,float,0,0);

	if( l_esm.learn(	lp_gray, 
						l_esm_row,
						l_esm_col,
						l_esm_height,
						l_esm_width ) == false )
	{
		cvReleaseImage(&lp_gray);
		cvReleaseImage(&lp_imag);
		cvReleaseImage(&lp_mean);
		cvReleaseImage(&lp_re_mask);
		cvReleaseImage(&lp_pt_mask);
		cvReleaseMat(&lp_roi_pt);
		cvReleaseMat(&lp_roi_re);
		cvReleaseMat(&lp_rob);
			
		printf("leo: esm could not be learned!");
		return 0;
	}

#ifdef LEARN

	for( int l_i=0; l_i<lp_rob->cols; ++l_i )
	{
		std::stringstream l_stringstream1;
		std::stringstream l_stringstream2;

		l_stringstream1 << l_leopar_name << l_i; 
		lp_leopar[l_i].set_parameters(l_size,10,30,18,20000);
		lp_leopar[l_i].learn(lp_mean,
							 lp_re_mask,
							 CV_MAT_ELEM(*lp_rob,float,1,l_i),
							 CV_MAT_ELEM(*lp_rob,float,0,l_i) );
	
		lp_leopar[l_i].save(l_stringstream1.str()+std::string(".leopar"));
		std::cerr << "save " << l_i << " leopar... " << std::endl;

		l_stringstream2 << l_track_name << l_i; 
		lp_hyper[l_i].set_parameters(11,11,5,21,10000,0);
		
		CvMat * lp_rec = cvCreateMat(3,4,CV_32FC1);
		cvSet(lp_rec,cvRealScalar(1));

		CV_MAT_ELEM(*lp_rec,float,0,0) = CV_MAT_ELEM(*lp_rob,float,0,l_i)-l_size/2;
		CV_MAT_ELEM(*lp_rec,float,1,0) = CV_MAT_ELEM(*lp_rob,float,1,l_i)-l_size/2;
		CV_MAT_ELEM(*lp_rec,float,0,1) = CV_MAT_ELEM(*lp_rob,float,0,l_i)+l_size/2;
		CV_MAT_ELEM(*lp_rec,float,1,1) = CV_MAT_ELEM(*lp_rob,float,1,l_i)-l_size/2;
		CV_MAT_ELEM(*lp_rec,float,0,2) = CV_MAT_ELEM(*lp_rob,float,0,l_i)+l_size/2;
		CV_MAT_ELEM(*lp_rec,float,1,2) = CV_MAT_ELEM(*lp_rob,float,1,l_i)+l_size/2;
		CV_MAT_ELEM(*lp_rec,float,0,3) = CV_MAT_ELEM(*lp_rob,float,0,l_i)-l_size/2;
		CV_MAT_ELEM(*lp_rec,float,1,3) = CV_MAT_ELEM(*lp_rob,float,1,l_i)+l_size/2;

		lp_hyper[l_i].learn(lp_mean,lp_rec);
		cvReleaseMat(&lp_rec);

		lp_hyper[l_i].save(l_stringstream2.str()+std::string(".hyper"));
		std::cerr << "save " << l_i << " hyper... " << std::endl;
	}
#else
	for( int l_i=0; l_i<lp_rob->cols; ++l_i )
	{
		std::stringstream l_stringstream1;
		std::stringstream l_stringstream2;

		l_stringstream1 << l_leopar_name << l_i; 
		lp_leopar[l_i].load(l_stringstream1.str()+std::string(".leopar"));
		std::cerr << "load " << l_i << " leopar... " << std::endl;

		l_stringstream2 << l_track_name << l_i; 
		lp_hyper[l_i].load(l_stringstream2.str()+std::string(".hyper"));
		std::cerr << "load " << l_i << " hyper... " << std::endl;
	}
#endif

	cvReleaseImage(&lp_gray);
	cvReleaseImage(&lp_imag);
	cvReleaseImage(&lp_mean);
	cvReleaseImage(&lp_re_mask);
	cvReleaseImage(&lp_pt_mask);
		
	bool l_one_break = false;
	bool l_esm_break = false;
	bool l_pat_break = true;
	
	cv::cv_create_window("hallo");
	cv::cv_camera l_camera;
	cv::cv_timer l_timer1;
	
	l_camera.set_cam(cv::usb);
	
	if( l_camera.start_capture_from_cam() == false )
	{
		printf("main_leo: web camera could not be initialized!");
		cvReleaseMat(&lp_roi_re);
		cvReleaseMat(&lp_roi_pt);
		cvReleaseMat(&lp_rob);
		delete[] lp_leopar;
		delete[] lp_hyper;
		return 0;
	}
	CvMat * lp_esm_result = NULL;

	while(true)
	{
		l_timer1.start(true);

		IplImage * lp_color = l_camera.get_image();
		IplImage * lp_gray  = cv::cv_convert_color_to_gray(lp_color);
		IplImage * lp_mean	= cv::cv_smooth(lp_gray,5);

		int l_inlier=0;
		int l_num_of_points=0;

		if( l_esm_break == true && lp_esm_result != NULL )
		{
			CvMat * lp_result = l_esm.track(lp_mean,lp_esm_result,10,10);

			if( l_esm.get_ncc() > 0.8 )
			{
				if( lp_result != NULL )
				{
					cv::cv_draw_poly(lp_color,lp_result,2,0,0,0);
					cv::cv_draw_poly(lp_color,lp_result,1,0,255,0);

					cvCopy(lp_result,lp_esm_result);
				}
				else
				{
					cvReleaseMat(&lp_esm_result);
					lp_esm_result = NULL;
				}
			}
			else
			{
				cvReleaseMat(&lp_esm_result);
				lp_esm_result = NULL;
			}
			cvReleaseMat(&lp_result);
		}
		if( l_esm_break == false || lp_esm_result == NULL )
		{
			CvMat * lp_points = NULL;

			cv::cv_harris::set_radius(4);
			cv::cv_harris::set_num_of_points(70); 
			lp_points = cv::cv_harris::get_points(lp_mean);

			if( lp_points == NULL )
			{
				cv::cv_show_image(lp_color,"hallo");
				cvWaitKey(1);
				cvReleaseImage(&lp_mean);
				cvReleaseImage(&lp_gray);
				cvReleaseImage(&lp_color);
				continue;
			}
			l_num_of_points = lp_points->cols;

			CvMat * lp_best = cvCreateMat(3,4,CV_32FC1);
			float l_best_ncc = -1.0f;
			int l_best_ind = -1;

			for( int l_i=0; l_i<lp_rob->cols; ++l_i )
			{
				for( int l_j=0; l_j<lp_points->cols; ++l_j )
				{
					if( CV_MAT_ELEM(*lp_points,float,1,l_j)-l_size/2-1 < 0 ||
						CV_MAT_ELEM(*lp_points,float,0,l_j)-l_size/2-1 < 0 ||
						CV_MAT_ELEM(*lp_points,float,1,l_j)+l_size/2+1 >= lp_mean->height ||
						CV_MAT_ELEM(*lp_points,float,0,l_j)+l_size/2+1 >= lp_mean->width  )
					{
						continue;
					}
					float l_ncc = 0.9;

					CvMat * lp_quad = NULL;

					lp_quad = lp_leopar[l_i].recognize(lp_mean,CV_MAT_ELEM(*lp_points,float,1,l_j),CV_MAT_ELEM(*lp_points,float,0,l_j));
					
					if( lp_quad == NULL  )
					{
						continue;
					}
					CvMat * lp_result = lp_hyper[l_i].track(lp_mean,lp_quad,2);

					if( lp_result == NULL )
					{
						cvReleaseMat(&lp_quad);				
						continue;
					}
					if(  lp_hyper[l_i].get_ncc() < l_ncc )
					{
						cvReleaseMat(&lp_quad);				
						continue;
					}
					++l_inlier;

					if( l_best_ncc < lp_hyper[l_i].get_ncc() )
					{
						l_best_ncc = lp_hyper[l_i].get_ncc();
						l_best_ind = l_i;
						cvCopy(lp_result,lp_best);
					}
					if( l_pat_break == true )
					{
						cv::cv_draw_poly(lp_color,lp_result,2,0,0,0);
						cv::cv_draw_poly(lp_color,lp_result,1,255,255,0);
					}
					cvReleaseMat(&lp_result);
					cvReleaseMat(&lp_quad);

					break; 
				}
				if( l_one_break == true && l_inlier == 1 ) 
					break;
			}
			if( l_inlier ) 
			{
				CvMat * lp_rect  = cvCreateMat(3,4,CV_32FC1);
				CvMat * lp_guess = cvCreateMat(3,4,CV_32FC1);
						
				CV_MAT_ELEM(*lp_rect,float,0,0) = CV_MAT_ELEM(*lp_rob,float,0,l_best_ind)-l_size/2;
				CV_MAT_ELEM(*lp_rect,float,1,0) = CV_MAT_ELEM(*lp_rob,float,1,l_best_ind)-l_size/2;
				CV_MAT_ELEM(*lp_rect,float,0,1) = CV_MAT_ELEM(*lp_rob,float,0,l_best_ind)+l_size/2;
				CV_MAT_ELEM(*lp_rect,float,1,1) = CV_MAT_ELEM(*lp_rob,float,1,l_best_ind)-l_size/2;
				CV_MAT_ELEM(*lp_rect,float,0,2) = CV_MAT_ELEM(*lp_rob,float,0,l_best_ind)+l_size/2;
				CV_MAT_ELEM(*lp_rect,float,1,2) = CV_MAT_ELEM(*lp_rob,float,1,l_best_ind)+l_size/2;
				CV_MAT_ELEM(*lp_rect,float,0,3) = CV_MAT_ELEM(*lp_rob,float,0,l_best_ind)-l_size/2;
				CV_MAT_ELEM(*lp_rect,float,1,3) = CV_MAT_ELEM(*lp_rob,float,1,l_best_ind)+l_size/2;
				CV_MAT_ELEM(*lp_rect,float,2,0) = 1;
				CV_MAT_ELEM(*lp_rect,float,2,1) = 1;
				CV_MAT_ELEM(*lp_rect,float,2,2) = 1;
				CV_MAT_ELEM(*lp_rect,float,2,3) = 1;

				CvMat * lp_hom = cv::cv_homography::compute(lp_best,lp_rect);

				cvMatMul(lp_hom,lp_roi_pt,lp_guess);
				cv::cv_homogenize(lp_guess);

				CvMat * lp_result = l_esm.track(lp_mean,lp_guess,10,10);

				if( l_esm.get_ncc() > 0.7 )
				{
					if( lp_result != NULL )
					{
						cv::cv_draw_poly(lp_color,lp_result,2,0,0,0);
						cv::cv_draw_poly(lp_color,lp_result,1,0,255,0);

						lp_esm_result = cvCreateMat(3,4,CV_32FC1);
						cvCopy(lp_result,lp_esm_result);
					}
				}
				cvReleaseMat(&lp_result);
				cvReleaseMat(&lp_guess);
				cvReleaseMat(&lp_rect);
				cvReleaseMat(&lp_hom);
			}
			cvReleaseMat(&lp_points);
			cvReleaseMat(&lp_best);
		}
		cv::cv_show_image(lp_color,"hallo");
		
		l_timer1.stop();

		std::cerr << "fps: " << l_timer1.get_fps() << " num_of_harris: " << l_num_of_points << " inlier: " << l_inlier << char(13) << std::flush;
	
		int l_key = cvWaitKey(1);
				
		cvReleaseImage(&lp_mean);
		cvReleaseImage(&lp_gray);
		cvReleaseImage(&lp_color);
		
		if( l_key == 27 )
		{
			break;
		}
		if( l_key == 111 )
		{
			l_one_break = !l_one_break;
		}	
		if( l_key == 101 )
		{
			l_esm_break = !l_esm_break;
		}	
		if( l_key == 112 )
		{
			l_pat_break = !l_pat_break;
		}
	}
	cvReleaseMat(&lp_roi_re);
	cvReleaseMat(&lp_roi_pt);
	cvReleaseMat(&lp_rob);
	delete[] lp_leopar;
	delete[] lp_hyper;
	
	return 0;
}

#endif