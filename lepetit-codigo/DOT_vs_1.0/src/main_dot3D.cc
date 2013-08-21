//////////////////////////////////////////////////////////////////////////////
//																			//
// Copyright 2007 - 2010 Lehrstuhl fuer Informat XVI,						//
// CAMP (Computer Aided Medical Procedures),								//
// Technische Universitaet Muenchen, Germany.								//
//																			//
// All rights reserved.	This file is part of VISION.						//
//																			//
// VISION is free software; you can redistribute it and/or modify it		//
// under the terms of the GNU General Public License as published by		//
// the Free Software Foundation; either version 2 of the License, or		//
// (at your option) any later version.										//
//																			//
// VISION is distributed in the hope that it will be useful, but			//
// WITHOUT ANY WARRANTY; without even the implied warranty of				//
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU			//
// General Public License for more details.									//
//																			//
// You should have received a copy of the GNU General Public License		//
// along with VISION; if not, write to the Free Software Foundation,		//
// Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA		//
//																			//
// main_dot3D: main_dot3D.cc												//
//																			//
// Authors: Stefan Hinterstoisser 2010										//
// Version: 1.0																//
//																			//
//////////////////////////////////////////////////////////////////////////////

#include "cv_define.h"

#ifdef MAIN_DOT_3D

#include "cv_dot_template.h"
#include "cv_camera.h"
#include "cv_esm.h"

int main( int argc, char * argv[] )
{
	const int l_T=7;
	const int l_N=154/l_T;
	const int l_M=154/l_T;
	const int l_IN=640;
	const int l_IM=480;
	const int l_G=121;

	int l_learn_thres_up=l_G*0.95;
	int l_learn_thres_down=l_G*0.9;
	int l_detect_thres=l_G*0.6;	

	cv::cv_dot_template<l_M,l_N,l_T,l_G> l_template(10);

	cv::cv_create_window("hallo1");
	cv::cv_create_window("hallo2");

	std::vector<CvMat*> l_cur_vec;

	cv::cv_camera l_camera;
	cv::cv_timer l_timer1;
	cv::cv_timer l_timer2;
	cv::cv_mouse l_mouse;

	l_mouse.start("hallo1");
	l_camera.set_cam(cv::usb);

	int l_tra_col1=0;
	int l_tra_row1=0;

	int l_x=-1;
	int l_y=-1;

	int l_class = 0;
	int l_thres = 90;

	bool l_show_esm = false;
	bool l_show_hyp = true;
	bool l_show_rec = true;
	bool l_learn_onl = false;
	bool l_learn_3D = false;

	if( l_camera.start_capture_from_cam() == false )
	{
		printf("main_dog: the camera could not be initalized!");
		return 0;
	}
	while(true)
	{
		IplImage * lp_color = NULL;
		IplImage * lp_gray  = NULL;
		IplImage * lp_mean  = NULL;

		lp_color = l_camera.get_image();
		lp_gray  = cv::cv_convert_color_to_gray(lp_color);
		lp_mean  = cv::cv_smooth(lp_gray,5);

		int l_xx=l_mouse.get_x();
		int l_yy=l_mouse.get_y();
		int l_e=l_mouse.get_event();

		if( l_xx >= 0 && l_yy >= 0 )
		{
			l_x = l_xx;
			l_y = l_yy;		
		}
		if( l_x != -1 && l_y != -1 )
		{
			l_tra_col1 = l_x;
			l_tra_row1 = l_y;
		}
		if( l_tra_col1-l_template.get_width()/2 >= 0 ||
			l_tra_row1-l_template.get_height()/2 >= 0 ||
			l_tra_col1+l_template.get_width()/2 <= lp_gray->width-1 ||
			l_tra_row1+l_template.get_height()/2 <= lp_gray->height-1 )
		{
			CvPoint l_pt1;
			CvPoint l_pt2;
			CvPoint l_pt3;
			CvPoint l_pt4;

			l_pt1.x = l_tra_col1-l_template.get_width()/2;
			l_pt1.y = l_tra_row1-l_template.get_height()/2;
			l_pt2.x = l_tra_col1+l_template.get_width()/2;
			l_pt2.y = l_tra_row1-l_template.get_height()/2;
			l_pt3.x = l_tra_col1+l_template.get_width()/2;
			l_pt3.y = l_tra_row1+l_template.get_height()/2;
			l_pt4.x = l_tra_col1-l_template.get_width()/2;
			l_pt4.y = l_tra_row1+l_template.get_height()/2;

			if( l_show_rec )
			{
				cvLine(lp_color,l_pt1,l_pt2,CV_RGB(0,0,0),3);
				cvLine(lp_color,l_pt2,l_pt3,CV_RGB(0,0,0),3);
				cvLine(lp_color,l_pt3,l_pt4,CV_RGB(0,0,0),3);
				cvLine(lp_color,l_pt4,l_pt1,CV_RGB(0,0,0),3);

				cvLine(lp_color,l_pt1,l_pt2,CV_RGB(255,255,0),1);
				cvLine(lp_color,l_pt2,l_pt3,CV_RGB(255,255,0),1);
				cvLine(lp_color,l_pt3,l_pt4,CV_RGB(255,255,0),1);
				cvLine(lp_color,l_pt4,l_pt1,CV_RGB(255,255,0),1);
			}
		}
		if(  l_x != -1 && l_y != -1 && l_e == 2 )
		{
		
			CvMat * lp_result = cvCreateMat(3,4,CV_32F);
			cvSet(lp_result,cvRealScalar(1));

			CV_MAT_ELEM(*lp_result,float,0,0) = l_x-l_template.get_width()/2;
			CV_MAT_ELEM(*lp_result,float,1,0) = l_y-l_template.get_height()/2;
			CV_MAT_ELEM(*lp_result,float,0,1) = l_x+l_template.get_width()/2;
			CV_MAT_ELEM(*lp_result,float,1,1) = l_y-l_template.get_height()/2;
			CV_MAT_ELEM(*lp_result,float,0,2) = l_x+l_template.get_width()/2;
			CV_MAT_ELEM(*lp_result,float,1,2) = l_y+l_template.get_height()/2;
			CV_MAT_ELEM(*lp_result,float,0,3) = l_x-l_template.get_width()/2;
			CV_MAT_ELEM(*lp_result,float,1,3) = l_y+l_template.get_height()/2;

			l_template.online_create_bit_list_fast(lp_mean,lp_result,0,7,0.9);
			l_template.add_class();

			l_cur_vec.push_back(lp_result);

			l_learn_onl = true;
			l_class += 1;
		}
		
		cv::cv_timer l_timer0;		
		cv::cv_timer l_timer1;		
		cv::cv_timer l_timer2;		
		cv::cv_timer l_timer3;		


		l_timer0.start();
		l_timer1.start();
		std::pair<Ipp8u*,Ipp32f*> l_img = l_template.compute_gradients(lp_mean,1);
		l_timer1.stop();
		l_timer2.start();
		std::list<cv::cv_candidate*> * lp_list = NULL;
		if( l_learn_onl == true )
		{
			lp_list = l_template.online_process(l_img.first,l_detect_thres,l_IN/l_T,l_IM/l_T);	
		}
		else
		{
			lp_list = l_template.process(l_img.first,l_detect_thres,l_IN/l_T,l_IM/l_T);
		}
		l_timer2.stop();
		l_timer0.stop();

		if( lp_list != NULL )
		{
			int l_size=0;

			float * lp_max_val = new float[l_template.get_classes()];
			float * lp_res_val = new float[l_template.get_classes()];
			float * lp_ind_val = new float[l_template.get_classes()];
			float * lp_col_val = new float[l_template.get_classes()];
			float * lp_row_val = new float[l_template.get_classes()];

			for( int l_j=0; l_j<l_template.get_classes(); ++l_j )
			{
				int l_counter = 0;

				int l_end_counter=0;

				if( l_learn_onl == true )
				{
					l_end_counter=1;
				}
				else
				{
					l_end_counter=7;
				}
				lp_max_val[l_j]=0;
				lp_res_val[l_j]=-1;
				lp_ind_val[l_j]=0;

				l_size += lp_list[l_j].size();

				lp_list[l_j].sort(cv::cv_candidate_ptr_cmp());

				for( std::list<cv::cv_candidate*>::iterator l_i=lp_list[l_j].begin(); l_i!=lp_list[l_j].end(); ++l_i )
				{
					if( l_counter < l_end_counter )
					{
						CvMat * lp_rec = cvCreateMat(3,4,CV_32F);
						cvCopy(l_template.get_rec()[(*l_i)->m_ind-1],lp_rec);

						CV_MAT_ELEM(*lp_rec,float,0,0) += (*l_i)->m_col;
						CV_MAT_ELEM(*lp_rec,float,1,0) += (*l_i)->m_row;
						CV_MAT_ELEM(*lp_rec,float,0,1) += (*l_i)->m_col;
						CV_MAT_ELEM(*lp_rec,float,1,1) += (*l_i)->m_row;
						CV_MAT_ELEM(*lp_rec,float,0,2) += (*l_i)->m_col;
						CV_MAT_ELEM(*lp_rec,float,1,2) += (*l_i)->m_row;
						CV_MAT_ELEM(*lp_rec,float,0,3) += (*l_i)->m_col;
						CV_MAT_ELEM(*lp_rec,float,1,3) += (*l_i)->m_row;	
						
						if( l_counter == 0 )
						{
							if( (*l_i)->m_val >= l_detect_thres )
							{
								l_template.render(lp_color,l_template.get_cnt()[(*l_i)->m_ind-1],(*l_i)->m_row,(*l_i)->m_col);
							}
							lp_max_val[l_j] = (*l_i)->m_val;
							
							if( l_show_hyp == true )
							{
								cv::cv_draw_poly(lp_color,lp_rec,3,255,255,255);
								cv::cv_draw_poly(lp_color,lp_rec,1,0,0,0);
							}
							if( l_learn_onl == true )
							{
								if( lp_max_val[l_j] >= l_learn_thres_down &&
									lp_max_val[l_j] < l_learn_thres_up )
								{
									float l_center_x=0;
									float l_center_y=0;

									float l_weight=0;

									int l_off = 20;

									for( int l_r=0; l_r<l_M*l_T+l_off*2; ++l_r )
									{
										for( int l_c=0; l_c<l_N*l_T+l_off*2; ++l_c )
										{
											int l_col = l_c+(*l_i)->m_col-l_off;
											int l_row = l_r+(*l_i)->m_row-l_off;

											if( l_col >= 0 && l_row >= 0 &&
												l_col < lp_gray->width &&
												l_row < lp_gray->height )
											{
												float l_mag = *(l_img.second+l_row*lp_gray->width+l_col);

												if( l_mag > 10 )
												{		
													l_center_x += l_mag*l_col;
													l_center_y += l_mag*l_row;

													l_weight+=l_mag;
												}
											}
										}
									}
									l_center_x /= l_weight;
									l_center_y /= l_weight;

									CvMat * lp_rec1 = cvCreateMat(3,4,CV_32F);
									cvSet(lp_rec1,cvRealScalar(1));
									
									CV_MAT_ELEM(*lp_rec1,float,0,0) = l_center_x-l_N*l_T/2;
									CV_MAT_ELEM(*lp_rec1,float,1,0) = l_center_y-l_M*l_T/2;
									CV_MAT_ELEM(*lp_rec1,float,0,1) = l_center_x+l_N*l_T/2;
									CV_MAT_ELEM(*lp_rec1,float,1,1) = l_center_y-l_M*l_T/2;
									CV_MAT_ELEM(*lp_rec1,float,0,2) = l_center_x+l_N*l_T/2;
									CV_MAT_ELEM(*lp_rec1,float,1,2) = l_center_y+l_M*l_T/2;
									CV_MAT_ELEM(*lp_rec1,float,0,3) = l_center_x-l_N*l_T/2;
									CV_MAT_ELEM(*lp_rec1,float,1,3) = l_center_y+l_M*l_T/2;

									l_template.online_create_bit_list_fast(lp_mean,lp_rec,l_j,7,0.9);
									cvReleaseMat(&lp_rec1);
								}
							}
						}
						else
						{
							if( l_show_hyp == true )
							{
								cv::cv_draw_poly(lp_color,lp_rec,2,0,0,0);
							}
						}
						cvReleaseMat(&lp_rec);
					}
					else
					{
						break;
					}
					++l_counter;
				}
			}	
			for( int l_i=0; l_i<l_template.get_classes(); ++l_i )
			{
				cv::empty_ptr_list(lp_list[l_i]);
			}
			std::cerr << "tim: " << (int)(l_timer0.get_time()*1000) << "ms; fps: " << (int)l_timer0.get_fps() << "fps ";
			std::cerr << "pre: " << (int)(l_timer1.get_time()*1000) << "ms; pro: " << (int)(l_timer2.get_time()*1000) << "ms; "; 

			for( int l_i=0; l_i<l_template.get_classes(); ++l_i )
			{
				std::cerr << (int)(lp_max_val[l_i]) << ";";
			}
			std::cerr << " size: " << l_template.get_templates() << " ," << l_size << ";" << l_template.get_classes() << "  " << char(13) << std::flush;

			delete[] lp_max_val;
			delete[] lp_res_val;
			delete[] lp_row_val;
			delete[] lp_col_val;
			delete[] lp_ind_val;
			delete[] lp_list;
		}
		else
		{
			std::cerr << "tim: " << (int)(l_timer0.get_time()*1000) << "ms; fps: " << (int)l_timer0.get_fps() << "fps ";
			std::cerr << "pre: " << (int)(l_timer1.get_time()*1000) << "ms; pro: " << (int)(l_timer2.get_time()*1000) << "ms ";
			std::cerr << " size: " << l_template.get_templates() << "    " << char(13) << std::flush;
		}	
		cv::cv_show_image(lp_color,"hallo1");
		int l_key = cvWaitKey(1);

		ippsFree(l_img.first);
		ippsFree(l_img.second);

		if( l_key == 2621440 )
		{
			l_thres--;

			if( l_thres < -100 )
			{
				l_thres = -100;
			}
			std::cerr << "new threshold: " << l_thres << "  " << std::endl;
		}
		if( l_key == 2490368 )
		{
			l_thres++;

			if( l_thres > +100 )
			{
				l_thres = +100;
			}
			std::cerr << "new threshold: " << l_thres << "  " << std::endl;
		}
		if( l_key == 106 )
		{
			static int l_n=0;

			std::stringstream l_name;

			l_name << "./pics/eccv10/img" << l_n << ".png";

			cv::cv_save_image(l_name.str(),lp_color);

			++l_n;
		}
		if( l_key == 114 )
		{
			l_show_rec = !l_show_rec;
		}
		if( l_key == 115 )
		{
			l_show_hyp = !l_show_hyp;
		}
		if( l_key == 101 )
		{
			l_show_esm = !l_show_esm;
		}
		if( l_key == 105 )
		{
			l_learn_onl = !l_learn_onl;
		
			if( l_learn_onl == false )
			{
				l_template.clear_clu_list();
				l_template.cluster_heu(4);
			}
		}
		if( l_key == 100 )
		{
			for( int l_i=0; l_i<l_template.get_classes(); ++l_i )
			{
				cvReleaseMat(&l_cur_vec[l_i]);
			}
			l_cur_vec.clear();

			l_template.clear_clu_list();
			l_template.clear_bit_list();
			l_template.clear_rec_list();
			l_template.clear_cnt_list();
		}
		if( l_key == 27 )
		{
			for( int l_i=0; l_i<l_template.get_classes(); ++l_i )
			{
				cvReleaseMat(&l_cur_vec[l_i]);
			}
			l_cur_vec.clear();

			cvReleaseImage(&lp_color);
			cvReleaseImage(&lp_gray);
			cvReleaseImage(&lp_mean);

			l_template.clear_clu_list();
			l_template.clear_bit_list();
			l_template.clear_rec_list();
			l_template.clear_cnt_list();

			return(0);
		}
		cvReleaseImage(&lp_color);
		cvReleaseImage(&lp_gray);
		cvReleaseImage(&lp_mean);
	}
	return 0;
}

#endif

