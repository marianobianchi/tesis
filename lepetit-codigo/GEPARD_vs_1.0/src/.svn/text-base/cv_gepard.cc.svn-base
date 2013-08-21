//////////////////////////////////////////////////////////////////////////////
//
// cv_gepard: cv_gepard.cc
//
// Authors: Stefan Hinterstoisser 2009
// Lehrstuhl fuer Informatik XVI
// Technische Universitaet Muenchen
// Version: 1.0
//
//////////////////////////////////////////////////////////////////////////////

/******************************* includes ***********************************/

#include "cv_gepard.h"
#include "ipps.h"

#include <iostream>
#include <sstream>
#include <omp.h>

/******************************** defines ***********************************/

/******************************* namespaces *********************************/

using namespace cv;

/****************************** constructors ********************************/

cv_gepard::cv_gepard( void )
{
	m_nx = 0;
	m_ny = 0;
	m_size = 0;
	m_num_of_samples = 0;
	
	mp_pos = NULL;

	m_means.clear();
	m_poses.clear();

	mp_k = cvCreateMat(3,3,CV_32FC1);
}

/******************************** destructor ********************************/

cv_gepard::~cv_gepard()
{
	this->clear();

	cvReleaseMat(&mp_k);
}

/****************************************************************************/

void cv_gepard::clear( void )
{
	for( int l_k=0; l_k<m_poses.size(); ++l_k )
	{
		cvReleaseMat(&m_poses[l_k]);
	}
	for( int l_k=0; l_k<m_means.size(); ++l_k )
	{	
		cvReleaseMat(&m_means[l_k]);
	}
	m_poses.clear();
	m_means.clear();

	cvReleaseMat(&mp_pos);
}

/****************************************************************************/
	
void cv_gepard::set_parameters( CvMat * ap_k, int a_size,
								int a_num_of_samples,
								int a_nx, int a_ny )
{
	if( a_size < 0 )
	{
		printf("cv_gepard: a_size is not appropriate!");
		return;
	}
	if( a_num_of_samples <= 0 )
	{
		printf("cv_gepard: a_a_num_of_smaples should be over 0!");
		return;
	}
	if( ap_k == NULL )
	{
		printf("cv_gepard: ap_k is NULL!");
		return;
	}
	if( ap_k->cols != 3 ||
		ap_k->rows != 3 )
	{
		printf("cv_gepard: ap_k is not of right size!");
		return;
	}
	if( a_nx < 0 || a_ny < 0 )
	{
		printf("cv_gepard: a_nx or a_ny is smaller than 0!");
		return;
	}
	m_nx = a_nx;
	m_ny = a_ny;
	m_size = a_size;
	m_num_of_samples = a_num_of_samples;

	cvCopy(ap_k,mp_k);
}

/****************************************************************************/

bool cv_gepard::learn(	IplImage * ap_image, 
						IplImage * ap_mask,
						int a_row, int a_col )
{
	if( ap_image == NULL )
	{
		printf("cv_gepard: ap_image is NULL!");
		return false;
	}
	if( ap_image->imageData == NULL )
	{
		printf("cv_gepard: ap_image->imageData is NULL!");
		return false;
	}
	if( ap_mask == NULL )
	{
		printf("cv_gepard: ap_mask is NULL!");
		return false;
	}
	if( ap_mask->imageData == NULL )
	{
		printf("cv_gepard: ap_mask->imageData is NULL!");
		return false;
	}
	if( cvGetElemType(ap_image) != CV_32FC1 )
	{
		printf("cv_gepard: type is not 32FC1!");
		return false;
	}
	if( a_row < 0 ||
		a_col < 0 ||
		a_row+m_size/2 >= ap_image->height ||
		a_col+m_size/2 >= ap_image->width )
	{
		printf("cv_gepard: a_row or a_col are not appropriate!");
		return false;
	}
	this->clear();
	
	//pose parameters...
	/*const int l_num_of_rot = 36;
	const int l_num_of_sca = 1;
	
	float l_sca[l_num_of_sca]		= {1.0};
	float l_sca_step[l_num_of_sca]	= {0.15};
	float l_rot[l_num_of_rot] = {0,10,20,30,40,50,60,70,80,90,100,110,120,130,
								140,150,160,170,180,190,200,210,220,230,240,
								250,260,270,280,290,300,310,320,330,340,350};
	float l_rot_step[l_num_of_rot] = {5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
			   						  5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5};*/
	const int l_num_of_rot = 12;
	const int l_num_of_sca = 1;

	float l_sca[l_num_of_sca]		= {1.0};
	float l_sca_step[l_num_of_sca]	= {0.20};
	float l_rot[l_num_of_rot]		= {0,30,60,90,120,150,180,210,240,270,300,330};
	float l_rot_step[l_num_of_rot]  = {15,15,15,15,15,15,15,15,15,15,15,15};
	

	float l_fac = 1;

	CvMat * lp_k = cvCreateMat(3,3,CV_32FC1);
	CvMat * lp_rec = cvCreateMat(3,4,CV_32FC1);

	CV_MAT_ELEM(*lp_rec,float,0,0) = -m_size/2+a_col;
	CV_MAT_ELEM(*lp_rec,float,1,0) = -m_size/2+a_row;
	CV_MAT_ELEM(*lp_rec,float,0,1) = +m_size/2+a_col;
	CV_MAT_ELEM(*lp_rec,float,1,1) = -m_size/2+a_row;
	CV_MAT_ELEM(*lp_rec,float,0,2) = +m_size/2+a_col;
	CV_MAT_ELEM(*lp_rec,float,1,2) = +m_size/2+a_row;
	CV_MAT_ELEM(*lp_rec,float,0,3) = -m_size/2+a_col;
	CV_MAT_ELEM(*lp_rec,float,1,3) = +m_size/2+a_row;
	CV_MAT_ELEM(*lp_rec,float,2,0) = 1;
	CV_MAT_ELEM(*lp_rec,float,2,1) = 1;
	CV_MAT_ELEM(*lp_rec,float,2,2) = 1;
	CV_MAT_ELEM(*lp_rec,float,2,3) = 1;

	cvCopy(mp_k,lp_k);

	CV_MAT_ELEM(*lp_k,float,0,2) = a_col;
	CV_MAT_ELEM(*lp_k,float,1,2) = a_row;

	std::vector<std::pair<CvPoint3D32f,float> > l_views = this->get_views();

	std::cerr << "number of views: " << l_views.size() << std::endl;

	mp_pos = this->create_positions(m_nx,m_ny,m_size,m_size);
	
	std::cerr << "num_of_grid_points: " << mp_pos->cols << "," << m_size << "," << m_nx << "," << m_ny << std::endl;

	int l_pose=0;
		
	for( int l_si=0; l_si<l_num_of_sca; ++l_si )
	{
		for( int l_ri=0; l_ri<l_num_of_rot; ++l_ri )
		{
			for( int l_vi=0; l_vi<l_views.size(); ++l_vi )
			{
				CvMat * lp_view = cvCreateMat(3,1,CV_32FC1);

				CV_MAT_ELEM(*lp_view,float,0,0) = l_views[l_vi].first.x;
				CV_MAT_ELEM(*lp_view,float,1,0) = l_views[l_vi].first.y;
				CV_MAT_ELEM(*lp_view,float,2,0) = l_views[l_vi].first.z;

				CvMat * lp_ground_rec = cvCreateMat(3,4,CV_32FC1);
				CvMat * lp_ground_hom = this->get_pose( lp_k,
														lp_view,
														l_rot[l_ri],
														l_sca[l_si] );

				cvMatMul(lp_ground_hom,lp_rec,lp_ground_rec);
				cv_homogenize(lp_ground_rec);

				CV_MAT_ELEM(*lp_ground_rec,float,0,0) -= a_col;
				CV_MAT_ELEM(*lp_ground_rec,float,1,0) -= a_row;
				CV_MAT_ELEM(*lp_ground_rec,float,0,1) -= a_col;
				CV_MAT_ELEM(*lp_ground_rec,float,1,1) -= a_row;
				CV_MAT_ELEM(*lp_ground_rec,float,0,2) -= a_col;
				CV_MAT_ELEM(*lp_ground_rec,float,1,2) -= a_row;
				CV_MAT_ELEM(*lp_ground_rec,float,0,3) -= a_col;
				CV_MAT_ELEM(*lp_ground_rec,float,1,3) -= a_row;
				
				//cv_print(lp_ground_rec);

				m_poses.push_back(lp_ground_rec);

				CvMat * lp_mean_int = cvCreateMat(1,mp_pos->cols,CV_32FC1);
		
				cvSet(lp_mean_int,cvRealScalar(0));
		
				cvReleaseMat(&lp_ground_hom);
				cvReleaseMat(&lp_view);

				for( int l_i=0; l_i<m_num_of_samples; ++l_i )
				{
					float l_orbit_orien = l_fac*rand()/(RAND_MAX+0.0)*360.0;
					float l_orbit_angle = l_fac*l_views[l_vi].second;
					float l_scale_off = l_fac*rand()/(RAND_MAX+0.0)*l_sca_step[l_si]*2*1.2-l_sca_step[l_si]*1.2;
					float l_rotat_off = l_fac*rand()/(RAND_MAX+0.0)*l_rot_step[l_ri]*2*1.2-l_rot_step[l_ri]*1.2;
					float l_trans_row = l_fac*rand()/(RAND_MAX+0.0)*10-5;
					float l_trans_col = l_fac*rand()/(RAND_MAX+0.0)*10-5;

					CvMat * lp_axis = this->get_axis(	l_views[l_vi].first,
														l_orbit_angle,
														l_orbit_orien );
								
 					CvMat * lp_hom  = this->get_pose(	lp_k,
														lp_axis,
														l_rot[l_ri]+l_rotat_off,
														l_sca[l_si]+l_scale_off );

					CvMat * lp_mean_pos = cvCreateMat(mp_pos->rows,mp_pos->cols,CV_32FC1);

					cvCopy(mp_pos,lp_mean_pos);

					for( int l_kk=0; l_kk<lp_mean_pos->cols; ++l_kk )
					{
						CV_MAT_ELEM(*lp_mean_pos,float,0,l_kk) += a_col+l_trans_col;
						CV_MAT_ELEM(*lp_mean_pos,float,1,l_kk) += a_row+l_trans_row;
					}
					CvMat * lp_inv = cvCreateMat(3,3,CV_32FC1);
					cvInvert(lp_hom,lp_inv,CV_SVD);

					CvMat * lp_warp_mean_pos = cvCreateMat(mp_pos->rows,mp_pos->cols,CV_32FC1);

					cvMatMul(lp_inv,lp_mean_pos,lp_warp_mean_pos);
					cv_homogenize(lp_warp_mean_pos);
					
					CvMat * lp_mean_int_tmp = this->get_train_intensity(ap_image,
																		ap_mask,
																		lp_warp_mean_pos);
					cvAdd(lp_mean_int,lp_mean_int_tmp,lp_mean_int);

					cvReleaseMat(&lp_hom);
					cvReleaseMat(&lp_inv);
					cvReleaseMat(&lp_axis);
					cvReleaseMat(&lp_mean_pos);
					cvReleaseMat(&lp_mean_int_tmp);
					cvReleaseMat(&lp_warp_mean_pos);
				}
				//std::cerr << "gep: " << l_pose << std::endl;

				cvScale(lp_mean_int,lp_mean_int,1.0/m_num_of_samples);

				cv_normalize_mean_std((float*)lp_mean_int->data.ptr,lp_mean_int->cols);

				m_means.push_back(lp_mean_int);
				
				++l_pose;
			}
		}
	}
	cvReleaseMat(&lp_k);
	cvReleaseMat(&lp_rec);

	return true;
}

/****************************************************************************/

IplImage * cv_gepard::get_patch(	CvMat * ap_mat,
									int a_sup_size, 
									int a_index )
{
	int l_r=0;
	int l_c=0;

	IplImage * lp_patch = cvCreateImage(cvSize(a_sup_size,a_sup_size),IPL_DEPTH_32F,1);
	
	#pragma omp parallel for private(l_r,l_c) \
	shared(lp_patch,ap_mat,a_index,a_sup_size)
	
	for( l_r=0; l_r<a_sup_size; ++l_r )
	{
		for( l_c=0; l_c<a_sup_size; ++l_c )
		{
			CV_IMAGE_ELEM(lp_patch,float,l_r,l_c)=
			CV_MAT_ELEM(*ap_mat,float,a_index,l_r*a_sup_size+l_c);
		}
	}
	return lp_patch;
}

/****************************************************************************/

CvMat * cv_gepard::recognize(	IplImage * ap_image, 
								int a_row, 
								int a_col )
{	
	CvMat * lp_mean_pos = cvCreateMat(mp_pos->rows,mp_pos->cols,CV_32FC1);

	int	  l_final_ind = 0.0000;
	float l_final_ncc = -10e10;
	float l_final_sca = 0.0000;

	for( float l_s=0.6; l_s<=1.8; l_s+=0.6 )
	{
		cvCopy(mp_pos,lp_mean_pos);

		if( a_row-(m_size+1)/2*l_s-1 < 0 ||
			a_col-(m_size+1)/2*l_s-1 < 0 ||
			a_row+(m_size+1)/2*l_s+1 >= ap_image->height ||
			a_col+(m_size+1)/2*l_s+1 >= ap_image->width )
		{
			continue;
		}
		for( int l_i=0; l_i<mp_pos->cols; ++l_i )
		{
			CV_MAT_ELEM(*lp_mean_pos,float,0,l_i) *= l_s;
			CV_MAT_ELEM(*lp_mean_pos,float,1,l_i) *= l_s;
			CV_MAT_ELEM(*lp_mean_pos,float,0,l_i) += a_col;
			CV_MAT_ELEM(*lp_mean_pos,float,1,l_i) += a_row;
		}
		CvMat * lp_curr_mean_int = this->get_run_intensity(ap_image,lp_mean_pos);
		
		cv_normalize_mean_std((float*)lp_curr_mean_int->data.ptr,lp_curr_mean_int->cols);
		
		int   l_max_index=0;
		float l_max_value=0;
		
		for( int l_k=0; l_k<m_means.size(); ++l_k )
		{	
			float l_ncc_value = cv_dot_product(	(float*)m_means[l_k]->data.ptr,
												(float*)lp_curr_mean_int->data.ptr,
												m_means[l_k]->cols);
			if( l_ncc_value > l_final_ncc ) 
			{
				l_final_ncc = l_ncc_value;
				l_final_ind = l_k;
				l_final_sca = l_s;
			}
		}
		cvReleaseMat(&lp_curr_mean_int); 
	}
	if( l_final_sca == 0.0 )
	{
		cvReleaseMat(&lp_mean_pos);
		return NULL;
	}
	CvMat * lp_guess = cvCreateMat(3,4,CV_32FC1);

	cvCopy(m_poses[l_final_ind],lp_guess);

	CV_MAT_ELEM(*lp_guess,float,0,0) *= l_final_sca;
	CV_MAT_ELEM(*lp_guess,float,1,0) *= l_final_sca;
	CV_MAT_ELEM(*lp_guess,float,0,1) *= l_final_sca;
	CV_MAT_ELEM(*lp_guess,float,1,1) *= l_final_sca;
	CV_MAT_ELEM(*lp_guess,float,0,2) *= l_final_sca;
	CV_MAT_ELEM(*lp_guess,float,1,2) *= l_final_sca;
	CV_MAT_ELEM(*lp_guess,float,0,3) *= l_final_sca;
	CV_MAT_ELEM(*lp_guess,float,1,3) *= l_final_sca;

	CV_MAT_ELEM(*lp_guess,float,0,0) += a_col;
	CV_MAT_ELEM(*lp_guess,float,1,0) += a_row;
	CV_MAT_ELEM(*lp_guess,float,0,1) += a_col;
	CV_MAT_ELEM(*lp_guess,float,1,1) += a_row;
	CV_MAT_ELEM(*lp_guess,float,0,2) += a_col;
	CV_MAT_ELEM(*lp_guess,float,1,2) += a_row;
	CV_MAT_ELEM(*lp_guess,float,0,3) += a_col;
	CV_MAT_ELEM(*lp_guess,float,1,3) += a_row;
	
	cvReleaseMat(&lp_mean_pos);

	return lp_guess;
}

/****************************************************************************/

IplImage * cv_gepard::warp(	IplImage * ap_image,
							CvMat * ap_hom,
							int a_size,
							int a_row,
							int a_col )
{
	IplImage * lp_image = cvCreateImage(cvSize(a_size,a_size),IPL_DEPTH_32F,1);

	CvMat * lp_inv = cvCreateMat(3,3,CV_32FC1);
	cvInvert(ap_hom,lp_inv,CV_SVD);

	for( int l_r=-a_size/2; l_r<a_size/2; ++l_r )
	{
		for( int l_c=-a_size/2; l_c<a_size/2; ++l_c )
		{
			float l_col =	CV_MAT_ELEM(*lp_inv,float,0,0)*(a_col+l_c)+
							CV_MAT_ELEM(*lp_inv,float,0,1)*(a_row+l_r)+
							CV_MAT_ELEM(*lp_inv,float,0,2);
			float l_row =	CV_MAT_ELEM(*lp_inv,float,1,0)*(a_col+l_c)+
							CV_MAT_ELEM(*lp_inv,float,1,1)*(a_row+l_r)+
							CV_MAT_ELEM(*lp_inv,float,1,2);
			float l_fac =	CV_MAT_ELEM(*lp_inv,float,2,0)*(a_col+l_c)+
							CV_MAT_ELEM(*lp_inv,float,2,1)*(a_row+l_r)+
							CV_MAT_ELEM(*lp_inv,float,2,2);

			if( l_fac == 0.0 ) continue;

			l_col /= l_fac;
			l_row /= l_fac;

			float l_val = 0;

			if( l_row >= 1 &&
				l_row < ap_image->height-1 &&
				l_col >= 1 &&
				l_col < ap_image->width-1 )
			{
				l_val = this->get_linear(ap_image,l_row,l_col);
			}
			else
			{
				l_val = 0;
			}
			CV_IMAGE_ELEM(lp_image,float,l_r+a_size/2,l_c+a_size/2) = l_val;
		}
	}
	cvReleaseMat(&lp_inv);

	return lp_image;
}

/****************************************************************************/

CvMat * cv_gepard::get_pose(	CvMat * ap_k,
								CvMat * ap_v,
								float a_orientation,
								float a_scale )
{
	CvMat * lp_id = cvCreateMat(3,3,CV_32FC1);
	CvMat * lp_rot  = cvCreateMat(3,3,CV_32FC1);
	CvMat * lp_inv  = cvCreateMat(3,3,CV_32FC1);
	CvMat * lp_hom  = cvCreateMat(3,3,CV_32FC1);
	CvMat * lp_tmp1 = cvCreateMat(3,3,CV_32FC1);
	CvMat * lp_tmp2 = cvCreateMat(3,3,CV_32FC1);
	CvMat * lp_tmp3 = cvCreateMat(3,3,CV_32FC1);
	CvMat * lp_rod1 = cvCreateMat(3,3,CV_32FC1);
	CvMat * lp_rod2 = cvCreateMat(3,3,CV_32FC1);
	CvMat * lp_rot1 = cvCreateMat(3,3,CV_32FC1);
	CvMat * lp_rot2 = cvCreateMat(3,3,CV_32FC1);
	CvMat * lp_scale = cvCreateMat(3,3,CV_32FC1);
	CvMat * lp_tra  = cvCreateMat(3,1,CV_32FC1);
	CvMat * lp_tra0 = cvCreateMat(3,1,CV_32FC1);
	CvMat * lp_tra1 = cvCreateMat(3,1,CV_32FC1);
	CvMat * lp_norm = cvCreateMat(3,1,CV_32FC1);
	CvMat * lp_raxis = cvCreateMat(3,1,CV_32FC1);
	
	double l_dist = 10e10;

	cvSet(lp_id,cvRealScalar(0));
	cvSet(lp_rod1,cvRealScalar(0));
	cvSet(lp_rod2,cvRealScalar(0));
	cvSet(lp_scale,cvRealScalar(0));
	
	CV_MAT_ELEM(*lp_norm,float,0,0) = 0;
	CV_MAT_ELEM(*lp_norm,float,1,0) = 0;
	CV_MAT_ELEM(*lp_norm,float,2,0) = 1;
	
	CV_MAT_ELEM(*lp_id,float,0,0) = 1;
	CV_MAT_ELEM(*lp_id,float,1,1) = 1;
	CV_MAT_ELEM(*lp_id,float,2,2) = 1;

	CV_MAT_ELEM(*lp_tra0,float,0,0) = 0;
	CV_MAT_ELEM(*lp_tra0,float,1,0) = 0;
	CV_MAT_ELEM(*lp_tra0,float,2,0) = l_dist;

	CV_MAT_ELEM(*lp_tra1,float,0,0) = 0;
	CV_MAT_ELEM(*lp_tra1,float,1,0) = 0;
	CV_MAT_ELEM(*lp_tra1,float,2,0) = 1;

	CV_MAT_ELEM(*lp_scale,float,0,0) = a_scale;
	CV_MAT_ELEM(*lp_scale,float,1,1) = a_scale;
	CV_MAT_ELEM(*lp_scale,float,2,2) = 1;
	CV_MAT_ELEM(*lp_scale,float,0,2) = -a_scale*
										CV_MAT_ELEM(*ap_k,float,0,2)+
										CV_MAT_ELEM(*ap_k,float,0,2);
	CV_MAT_ELEM(*lp_scale,float,1,2) = -a_scale*
										CV_MAT_ELEM(*ap_k,float,1,2)+
										CV_MAT_ELEM(*ap_k,float,1,2);

	cvCrossProduct(ap_v,lp_norm,lp_raxis);
	cvNormalize(lp_raxis,lp_raxis);
	
	CV_MAT_ELEM(*lp_rod1,float,0,1) = -CV_MAT_ELEM(*lp_raxis,float,2,0);
	CV_MAT_ELEM(*lp_rod1,float,0,2) =  CV_MAT_ELEM(*lp_raxis,float,1,0);
	CV_MAT_ELEM(*lp_rod1,float,1,0) =  CV_MAT_ELEM(*lp_raxis,float,2,0);
	CV_MAT_ELEM(*lp_rod1,float,1,2) = -CV_MAT_ELEM(*lp_raxis,float,0,0);
	CV_MAT_ELEM(*lp_rod1,float,2,0) = -CV_MAT_ELEM(*lp_raxis,float,1,0);
	CV_MAT_ELEM(*lp_rod1,float,2,1) =  CV_MAT_ELEM(*lp_raxis,float,0,0);
	
	CV_MAT_ELEM(*lp_rod2,float,0,1) = -CV_MAT_ELEM(*lp_norm,float,2,0);
	CV_MAT_ELEM(*lp_rod2,float,0,2) =  CV_MAT_ELEM(*lp_norm,float,1,0);
	CV_MAT_ELEM(*lp_rod2,float,1,0) =  CV_MAT_ELEM(*lp_norm,float,2,0);
	CV_MAT_ELEM(*lp_rod2,float,1,2) = -CV_MAT_ELEM(*lp_norm,float,0,0);
	CV_MAT_ELEM(*lp_rod2,float,2,0) = -CV_MAT_ELEM(*lp_norm,float,1,0);
	CV_MAT_ELEM(*lp_rod2,float,2,1) =  CV_MAT_ELEM(*lp_norm,float,0,0);
		
	float l_angle1 = acos(cvDotProduct(ap_v,lp_norm)/(cvNorm(ap_v)*cvNorm(lp_norm)));
	float l_angle2 = a_orientation*cv_pi/180.0;

	cvGEMM(lp_rod1,lp_rod1,(1.0-cos(l_angle1)),lp_rod1,(sin(l_angle1)),lp_rot1);
	cvGEMM(lp_rod2,lp_rod2,(1.0-cos(l_angle2)),lp_rod2,(sin(l_angle2)),lp_rot2);

	cvAdd(lp_id,lp_rot1,lp_rot1);
	cvAdd(lp_id,lp_rot2,lp_rot2);

	cvMatMul(lp_rot2,lp_rot1,lp_rot);

	cvGEMM(lp_rot,lp_tra0,-1,lp_tra1,1.0,lp_tra);
	cvAdd(lp_tra,lp_tra0,lp_tra);

	cvInvert(ap_k,lp_inv,CV_SVD);

	cvGEMM(lp_tra,lp_norm,1.0/l_dist,lp_rot,1.0,lp_tmp1,CV_GEMM_B_T);
	cvMatMul(lp_scale,ap_k,lp_tmp2);
	cvMatMul(lp_tmp2,lp_tmp1,lp_tmp3);
	cvMatMul(lp_tmp3,lp_inv,lp_hom);

	cvReleaseMat(&lp_id);
	cvReleaseMat(&lp_rot);
	cvReleaseMat(&lp_inv);
	cvReleaseMat(&lp_tra);
	cvReleaseMat(&lp_tmp1);
	cvReleaseMat(&lp_tmp2);
	cvReleaseMat(&lp_tmp3);
	cvReleaseMat(&lp_rod1);
	cvReleaseMat(&lp_rod2);
	cvReleaseMat(&lp_rot1);
	cvReleaseMat(&lp_rot2);
	cvReleaseMat(&lp_tra0);
	cvReleaseMat(&lp_tra1);
	cvReleaseMat(&lp_norm);
	cvReleaseMat(&lp_scale);
	cvReleaseMat(&lp_raxis);

	return lp_hom;
}

/****************************************************************************/

CvMat * cv_gepard::get_axis(	CvPoint3D32f & a_axis,
								double a_angle,
								double a_orientation )
{
	CvMat * lp_id = cvCreateMat(3,3,CV_32FC1);
	CvMat * lp_res = cvCreateMat(3,1,CV_32FC1);
	CvMat * lp_rot = cvCreateMat(3,3,CV_32FC1);
	CvMat * lp_rod1 = cvCreateMat(3,3,CV_32FC1);
	CvMat * lp_rod2 = cvCreateMat(3,3,CV_32FC1);
	CvMat * lp_rot1 = cvCreateMat(3,3,CV_32FC1);
	CvMat * lp_rot2 = cvCreateMat(3,3,CV_32FC1);
	CvMat * lp_axis = cvCreateMat(3,1,CV_32FC1);
	CvMat * lp_perp = cvCreateMat(3,1,CV_32FC1);
	CvMat * lp_raxis = cvCreateMat(3,1,CV_32FC1);

	CV_MAT_ELEM(*lp_axis,float,0,0) = a_axis.x;
	CV_MAT_ELEM(*lp_axis,float,1,0) = a_axis.y;
	CV_MAT_ELEM(*lp_axis,float,2,0) = a_axis.z;

	cvNormalize(lp_axis,lp_axis);

	cvSet(lp_id,cvRealScalar(0));
	cvSet(lp_rod1,cvRealScalar(0));
	cvSet(lp_rod2,cvRealScalar(0));
	
	CV_MAT_ELEM(*lp_id,float,0,0) = 1;
	CV_MAT_ELEM(*lp_id,float,1,1) = 1;
	CV_MAT_ELEM(*lp_id,float,2,2) = 1;

	CV_MAT_ELEM(*lp_perp,float,0,0) = 0;
	CV_MAT_ELEM(*lp_perp,float,1,0) = -CV_MAT_ELEM(*lp_axis,float,2,0);
	CV_MAT_ELEM(*lp_perp,float,2,0) = CV_MAT_ELEM(*lp_axis,float,1,0);

	cvCrossProduct(lp_axis,lp_perp,lp_raxis);
	cvNormalize(lp_raxis,lp_raxis);
	
	CV_MAT_ELEM(*lp_rod1,float,0,1) = -CV_MAT_ELEM(*lp_raxis,float,2,0);
	CV_MAT_ELEM(*lp_rod1,float,0,2) =  CV_MAT_ELEM(*lp_raxis,float,1,0);
	CV_MAT_ELEM(*lp_rod1,float,1,0) =  CV_MAT_ELEM(*lp_raxis,float,2,0);
	CV_MAT_ELEM(*lp_rod1,float,1,2) = -CV_MAT_ELEM(*lp_raxis,float,0,0);
	CV_MAT_ELEM(*lp_rod1,float,2,0) = -CV_MAT_ELEM(*lp_raxis,float,1,0);
	CV_MAT_ELEM(*lp_rod1,float,2,1) =  CV_MAT_ELEM(*lp_raxis,float,0,0);
	
	CV_MAT_ELEM(*lp_rod2,float,0,1) = -CV_MAT_ELEM(*lp_axis,float,2,0);
	CV_MAT_ELEM(*lp_rod2,float,0,2) =  CV_MAT_ELEM(*lp_axis,float,1,0);
	CV_MAT_ELEM(*lp_rod2,float,1,0) =  CV_MAT_ELEM(*lp_axis,float,2,0);
	CV_MAT_ELEM(*lp_rod2,float,1,2) = -CV_MAT_ELEM(*lp_axis,float,0,0);
	CV_MAT_ELEM(*lp_rod2,float,2,0) = -CV_MAT_ELEM(*lp_axis,float,1,0);
	CV_MAT_ELEM(*lp_rod2,float,2,1) =  CV_MAT_ELEM(*lp_axis,float,0,0);
	
	float l_angle1 = a_angle*cv_pi/180.0;
	float l_angle2 = a_orientation*cv_pi/180.0;

	cvGEMM(lp_rod1,lp_rod1,(1.0-cos(l_angle1)),lp_rod1,(sin(l_angle1)),lp_rot1);
	cvGEMM(lp_rod2,lp_rod2,(1.0-cos(l_angle2)),lp_rod2,(sin(l_angle2)),lp_rot2);

	cvAdd(lp_id,lp_rot1,lp_rot1);
	cvAdd(lp_id,lp_rot2,lp_rot2);

	cvMatMul(lp_rot2,lp_rot1,lp_rot);
	cvMatMul(lp_rot,lp_axis,lp_res);	
	
	cvReleaseMat(&lp_id);
	cvReleaseMat(&lp_rot);
	cvReleaseMat(&lp_rod1);
	cvReleaseMat(&lp_rod2);
	cvReleaseMat(&lp_rot1);
	cvReleaseMat(&lp_rot2);
	cvReleaseMat(&lp_axis);
	cvReleaseMat(&lp_perp);
	cvReleaseMat(&lp_raxis);

	return lp_res;
}

/****************************************************************************/

std::vector<std::pair<CvPoint3D32f,float> > cv_gepard::get_views( void )
{
	/*
	CvPoint3D32f l_v0;
	CvPoint3D32f l_v1;
	CvPoint3D32f l_v2;
	CvPoint3D32f l_v3;
	CvPoint3D32f l_v4;
	CvPoint3D32f l_v5;
	CvPoint3D32f l_v6;
	CvPoint3D32f l_v7;
	CvPoint3D32f l_v8;
	CvPoint3D32f l_v9;
	CvPoint3D32f l_v10;
	CvPoint3D32f l_v11;
	CvPoint3D32f l_v12;
	CvPoint3D32f l_v13;
	CvPoint3D32f l_v14;
	CvPoint3D32f l_v15;

	l_v0.x = 0.00000000;	l_v1.x = 0.89442700;	l_v2.x = 0.2763930;
	l_v0.y = 0.00000000;	l_v1.y = 0.00000000;	l_v2.y = 0.8506510;
	l_v0.z = 1.00000000;	l_v1.z = 0.44721400;	l_v2.z = 0.4472140;

	l_v3.x	= 0.27639300;	l_v4.x	= -0.7236070;	l_v5.x	= -0.723607;
	l_v3.y  = -0.8506510;	l_v4.y  = 0.52573100;	l_v5.y  = -0.525731;
	l_v3.z  = 0.44721400;	l_v4.z  = 0.44721400;	l_v5.z  = 0.4472140;

	l_v6.x = 0.16246000;	l_v7.x = 0.68819100;	l_v8.x = 0.5257310;
	l_v6.y = 0.50000000;	l_v7.y = 0.50000000;	l_v8.y = 0.0000000;
	l_v6.z = 0.85065100;	l_v7.z = 0.52573100;	l_v8.z = 0.8506510;

	l_v9.x = -0.4253250;	l_v10.x = -0.262866;	l_v11.x = -0.262866;
	l_v9.y = 0.30901700;	l_v10.y = 0.8090170;	l_v11.y = -0.809017;
	l_v9.z = 0.85065100;	l_v10.z = 0.5257310;	l_v11.z = 0.5257310;

	l_v12.x = -0.425325;	l_v13.x = 0.1624600;	l_v14.x = 0.6881910;
	l_v12.y = -0.309017;	l_v13.y = -0.500000;	l_v14.y = -0.500000;
	l_v12.z = 0.8506510;	l_v13.z = 0.8506510;	l_v14.z = 0.5257310;

	l_v15.x = -0.850651;
	l_v15.y = 0.0000000;
	l_v15.z = 0.5257310;

	std::vector<std::pair<CvPoint3D32f,float> > l_views;

	l_views.push_back(std::pair<CvPoint3D32f,float>(l_v0,15.8587));
	l_views.push_back(std::pair<CvPoint3D32f,float>(l_v1,15.8587));
	l_views.push_back(std::pair<CvPoint3D32f,float>(l_v2,15.8587));
	l_views.push_back(std::pair<CvPoint3D32f,float>(l_v3,15.8587));
	l_views.push_back(std::pair<CvPoint3D32f,float>(l_v4,15.8587));
	l_views.push_back(std::pair<CvPoint3D32f,float>(l_v5,15.8587));
	l_views.push_back(std::pair<CvPoint3D32f,float>(l_v6,15.8587));
	l_views.push_back(std::pair<CvPoint3D32f,float>(l_v7,15.8587));
	l_views.push_back(std::pair<CvPoint3D32f,float>(l_v8,15.8587));
	l_views.push_back(std::pair<CvPoint3D32f,float>(l_v9,15.8587));
	l_views.push_back(std::pair<CvPoint3D32f,float>(l_v10,15.8587));
	l_views.push_back(std::pair<CvPoint3D32f,float>(l_v11,15.8587));
	l_views.push_back(std::pair<CvPoint3D32f,float>(l_v12,15.8587));
	l_views.push_back(std::pair<CvPoint3D32f,float>(l_v13,15.8587));
	l_views.push_back(std::pair<CvPoint3D32f,float>(l_v14,15.8587));
	l_views.push_back(std::pair<CvPoint3D32f,float>(l_v15,15.8587));

	float l_max_angle=0;

	for( int l_i=0; l_i<l_views.size(); ++l_i )
	{
		float l_angle = (l_views[0].first.x*l_views[l_i].first.x+
						 l_views[0].first.y*l_views[l_i].first.y+
						 l_views[0].first.z*l_views[l_i].first.z)/
						(sqrt(SQR(l_views[0].first.x)+SQR(l_views[0].first.y)+SQR(l_views[0].first.z))*
						 sqrt(SQR(l_views[l_i].first.x)+SQR(l_views[l_i].first.y)+SQR(l_views[l_i].first.z)));
	
		l_angle = acos(l_angle)*180.0/cv_pi;

		if( l_angle > l_max_angle )
		{
			l_max_angle = l_angle;
		}
	}
	for( int l_i=1; l_i<l_views.size(); ++l_i )
	{
		CvPoint3D32f l_norm;

		l_norm.x = l_views[0].first.y*l_views[l_i].first.z-l_views[0].first.z*l_views[l_i].first.y;
		l_norm.y = l_views[0].first.z*l_views[l_i].first.x-l_views[0].first.x*l_views[l_i].first.z;
		l_norm.z = l_views[0].first.x*l_views[l_i].first.y-l_views[0].first.y*l_views[l_i].first.x;

		float l_n = sqrt(SQR(l_norm.x)+SQR(l_norm.y)+SQR(l_norm.z));

		l_norm.x /= l_n;
		l_norm.y /= l_n;
		l_norm.z /= l_n;

		float l_angle = (l_views[0].first.x*l_views[l_i].first.x+
						 l_views[0].first.y*l_views[l_i].first.y+
						 l_views[0].first.z*l_views[l_i].first.z)/
						(sqrt(SQR(l_views[0].first.x)+SQR(l_views[0].first.y)+SQR(l_views[0].first.z))*
						 sqrt(SQR(l_views[l_i].first.x)+SQR(l_views[l_i].first.y)+SQR(l_views[l_i].first.z)));
	
		l_angle = acos(l_angle)*180/cv_pi;

		float l_new_angle = (l_angle/l_max_angle)*45*cv_pi/180.0;

		CvMat * lp_rodriguez = cvCreateMat(3,3,CV_32FC1);
		CvMat * lp_identity  = cvCreateMat(3,3,CV_32FC1);

		cvSet(lp_rodriguez,cvRealScalar(0));
		cvSet(lp_identity,cvRealScalar(0));

		CV_MAT_ELEM(*lp_rodriguez,float,0,1) = -l_norm.z;
		CV_MAT_ELEM(*lp_rodriguez,float,0,2) =  l_norm.y;
		CV_MAT_ELEM(*lp_rodriguez,float,1,0) =  l_norm.z;
		CV_MAT_ELEM(*lp_rodriguez,float,1,2) = -l_norm.x;
		CV_MAT_ELEM(*lp_rodriguez,float,2,0) = -l_norm.y;
		CV_MAT_ELEM(*lp_rodriguez,float,2,1) =  l_norm.x;

		CV_MAT_ELEM(*lp_identity,float,0,0) = 1; 
		CV_MAT_ELEM(*lp_identity,float,0,1) = 0;
		CV_MAT_ELEM(*lp_identity,float,0,2) = 0; 
		CV_MAT_ELEM(*lp_identity,float,1,0) = 0;
		CV_MAT_ELEM(*lp_identity,float,1,1) = 1;
		CV_MAT_ELEM(*lp_identity,float,1,2) = 0;
		CV_MAT_ELEM(*lp_identity,float,2,0) = 0;
		CV_MAT_ELEM(*lp_identity,float,2,1) = 0;
		CV_MAT_ELEM(*lp_identity,float,2,2) = 1;

		CvMat * lp_rotation = cvCreateMat(3,3,CV_32FC1);
		CvMat * lp_rot_sqr = cvCreateMat(3,3,CV_32FC1);
	
		cvMatMul(lp_rodriguez,lp_rodriguez,lp_rot_sqr);
		cvAddWeighted(lp_rodriguez,sin(l_new_angle),lp_rot_sqr, 1.0-cos(l_new_angle),0,lp_rotation);
		cvAdd(lp_identity,lp_rotation,lp_rotation);

		l_views[l_i].first.x = CV_MAT_ELEM(*lp_rotation,float,0,2);
		l_views[l_i].first.y = CV_MAT_ELEM(*lp_rotation,float,1,2);
		l_views[l_i].first.z = CV_MAT_ELEM(*lp_rotation,float,2,2);

		cvReleaseMat(&lp_rodriguez);
		cvReleaseMat(&lp_identity);
		cvReleaseMat(&lp_rotation);
		cvReleaseMat(&lp_rot_sqr);
	}
	for( int l_i=0; l_i<l_views.size(); ++l_i )
	{
		float l_min_angle=180;

		for( int l_j=0; l_j<l_views.size(); ++l_j )
		{
			if( l_i != l_j )
			{
				float l_angle = (l_views[l_j].first.x*l_views[l_i].first.x+
								 l_views[l_j].first.y*l_views[l_i].first.y+
								 l_views[l_j].first.z*l_views[l_i].first.z)/
								(sqrt(SQR(l_views[l_j].first.x)+SQR(l_views[l_j].first.y)+SQR(l_views[l_j].first.z))*
								 sqrt(SQR(l_views[l_i].first.x)+SQR(l_views[l_i].first.y)+SQR(l_views[l_i].first.z)));
			
				l_angle = acos(l_angle)*180.0/cv_pi;

				if( l_angle < l_min_angle )
				{
					l_min_angle = l_angle;
				}
			}
		}
		l_views[l_i].second = l_min_angle/2;
	}

	return l_views;*/

	CvPoint3D32f l_v0;
	CvPoint3D32f l_v1;
	CvPoint3D32f l_v2;
	CvPoint3D32f l_v3;
	CvPoint3D32f l_v4;
	CvPoint3D32f l_v5;
	CvPoint3D32f l_v6;
	CvPoint3D32f l_v7;
	CvPoint3D32f l_v8;
	CvPoint3D32f l_v9;
	CvPoint3D32f l_v10;
	CvPoint3D32f l_v11;
	CvPoint3D32f l_v12;
	CvPoint3D32f l_v13;
	CvPoint3D32f l_v14;
	CvPoint3D32f l_v15;

	l_v0.x = 0.00000000;	l_v1.x = 0.89442700;	l_v2.x = 0.2763930;
	l_v0.y = 0.00000000;	l_v1.y = 0.00000000;	l_v2.y = 0.8506510;
	l_v0.z = 1.00000000;	l_v1.z = 0.44721400;	l_v2.z = 0.4472140;

	l_v3.x	= 0.27639300;	l_v4.x	= -0.7236070;	l_v5.x	= -0.723607;
	l_v3.y  = -0.8506510;	l_v4.y  = 0.52573100;	l_v5.y  = -0.525731;
	l_v3.z  = 0.44721400;	l_v4.z  = 0.44721400;	l_v5.z  = 0.4472140;

	l_v6.x = 0.16246000;	l_v7.x = 0.68819100;	l_v8.x = 0.5257310;
	l_v6.y = 0.50000000;	l_v7.y = 0.50000000;	l_v8.y = 0.0000000;
	l_v6.z = 0.85065100;	l_v7.z = 0.52573100;	l_v8.z = 0.8506510;

	l_v9.x = -0.4253250;	l_v10.x = -0.262866;	l_v11.x = -0.262866;
	l_v9.y = 0.30901700;	l_v10.y = 0.8090170;	l_v11.y = -0.809017;
	l_v9.z = 0.85065100;	l_v10.z = 0.5257310;	l_v11.z = 0.5257310;

	l_v12.x = -0.425325;	l_v13.x = 0.1624600;	l_v14.x = 0.6881910;
	l_v12.y = -0.309017;	l_v13.y = -0.500000;	l_v14.y = -0.500000;
	l_v12.z = 0.8506510;	l_v13.z = 0.8506510;	l_v14.z = 0.5257310;

	l_v15.x = -0.850651;
	l_v15.y = 0.0000000;
	l_v15.z = 0.5257310;

	std::vector<std::pair<CvPoint3D32f,float> > l_views;

	l_views.push_back(std::pair<CvPoint3D32f,float>(l_v0,15.8587));
	l_views.push_back(std::pair<CvPoint3D32f,float>(l_v1,15.8587));
	l_views.push_back(std::pair<CvPoint3D32f,float>(l_v2,15.8587));
	l_views.push_back(std::pair<CvPoint3D32f,float>(l_v3,15.8587));
	l_views.push_back(std::pair<CvPoint3D32f,float>(l_v4,15.8587));
	l_views.push_back(std::pair<CvPoint3D32f,float>(l_v5,15.8587));
	l_views.push_back(std::pair<CvPoint3D32f,float>(l_v6,15.8587));
	l_views.push_back(std::pair<CvPoint3D32f,float>(l_v7,15.8587));
	l_views.push_back(std::pair<CvPoint3D32f,float>(l_v8,15.8587));
	l_views.push_back(std::pair<CvPoint3D32f,float>(l_v9,15.8587));
	l_views.push_back(std::pair<CvPoint3D32f,float>(l_v10,15.8587));
	l_views.push_back(std::pair<CvPoint3D32f,float>(l_v11,15.8587));
	l_views.push_back(std::pair<CvPoint3D32f,float>(l_v12,15.8587));
	l_views.push_back(std::pair<CvPoint3D32f,float>(l_v13,15.8587));
	l_views.push_back(std::pair<CvPoint3D32f,float>(l_v14,15.8587));
	l_views.push_back(std::pair<CvPoint3D32f,float>(l_v15,15.8587));
	
	float l_max_angle=0;

	for( int l_i=0; l_i<l_views.size(); ++l_i )
	{
		float l_angle = (l_views[0].first.x*l_views[l_i].first.x+
						 l_views[0].first.y*l_views[l_i].first.y+
						 l_views[0].first.z*l_views[l_i].first.z)/
						(sqrt(SQR(l_views[0].first.x)+SQR(l_views[0].first.y)+SQR(l_views[0].first.z))*
						 sqrt(SQR(l_views[l_i].first.x)+SQR(l_views[l_i].first.y)+SQR(l_views[l_i].first.z)));
	
		l_angle = acos(l_angle)*180.0/cv_pi;

		if( l_angle > l_max_angle )
		{
			l_max_angle = l_angle;
		}
	}
	for( int l_i=1; l_i<l_views.size(); ++l_i )
	{
		CvPoint3D32f l_norm;

		l_norm.x = l_views[0].first.y*l_views[l_i].first.z-l_views[0].first.z*l_views[l_i].first.y;
		l_norm.y = l_views[0].first.z*l_views[l_i].first.x-l_views[0].first.x*l_views[l_i].first.z;
		l_norm.z = l_views[0].first.x*l_views[l_i].first.y-l_views[0].first.y*l_views[l_i].first.x;

		float l_n = sqrt(SQR(l_norm.x)+SQR(l_norm.y)+SQR(l_norm.z));

		l_norm.x /= l_n;
		l_norm.y /= l_n;
		l_norm.z /= l_n;

		float l_angle = (l_views[0].first.x*l_views[l_i].first.x+
						 l_views[0].first.y*l_views[l_i].first.y+
						 l_views[0].first.z*l_views[l_i].first.z)/
						(sqrt(SQR(l_views[0].first.x)+SQR(l_views[0].first.y)+SQR(l_views[0].first.z))*
						 sqrt(SQR(l_views[l_i].first.x)+SQR(l_views[l_i].first.y)+SQR(l_views[l_i].first.z)));
	
		l_angle = acos(l_angle)*180/cv_pi;

		float l_new_angle = (l_angle/l_max_angle)*45*cv_pi/180.0;

		CvMat * lp_rodriguez = cvCreateMat(3,3,CV_32FC1);
		CvMat * lp_identity  = cvCreateMat(3,3,CV_32FC1);

		cvSet(lp_rodriguez,cvRealScalar(0));
		cvSet(lp_identity,cvRealScalar(0));

		CV_MAT_ELEM(*lp_rodriguez,float,0,1) = -l_norm.z;
		CV_MAT_ELEM(*lp_rodriguez,float,0,2) =  l_norm.y;
		CV_MAT_ELEM(*lp_rodriguez,float,1,0) =  l_norm.z;
		CV_MAT_ELEM(*lp_rodriguez,float,1,2) = -l_norm.x;
		CV_MAT_ELEM(*lp_rodriguez,float,2,0) = -l_norm.y;
		CV_MAT_ELEM(*lp_rodriguez,float,2,1) =  l_norm.x;

		CV_MAT_ELEM(*lp_identity,float,0,0) = 1; 
		CV_MAT_ELEM(*lp_identity,float,0,1) = 0;
		CV_MAT_ELEM(*lp_identity,float,0,2) = 0; 
		CV_MAT_ELEM(*lp_identity,float,1,0) = 0;
		CV_MAT_ELEM(*lp_identity,float,1,1) = 1;
		CV_MAT_ELEM(*lp_identity,float,1,2) = 0;
		CV_MAT_ELEM(*lp_identity,float,2,0) = 0;
		CV_MAT_ELEM(*lp_identity,float,2,1) = 0;
		CV_MAT_ELEM(*lp_identity,float,2,2) = 1;

		CvMat * lp_rotation = cvCreateMat(3,3,CV_32FC1);
		CvMat * lp_rot_sqr = cvCreateMat(3,3,CV_32FC1);
	
		cvMatMul(lp_rodriguez,lp_rodriguez,lp_rot_sqr);
		cvAddWeighted(lp_rodriguez,sin(l_new_angle),lp_rot_sqr, 1.0-cos(l_new_angle),0,lp_rotation);
		cvAdd(lp_identity,lp_rotation,lp_rotation);

		l_views[l_i].first.x = CV_MAT_ELEM(*lp_rotation,float,0,2);
		l_views[l_i].first.y = CV_MAT_ELEM(*lp_rotation,float,1,2);
		l_views[l_i].first.z = CV_MAT_ELEM(*lp_rotation,float,2,2);

		cvReleaseMat(&lp_rodriguez);
		cvReleaseMat(&lp_identity);
		cvReleaseMat(&lp_rotation);
		cvReleaseMat(&lp_rot_sqr);
	}
	for( int l_i=0; l_i<l_views.size(); ++l_i )
	{
		float l_min_angle=180;

		for( int l_j=0; l_j<l_views.size(); ++l_j )
		{
			if( l_i != l_j )
			{
				float l_angle = (l_views[l_j].first.x*l_views[l_i].first.x+
								 l_views[l_j].first.y*l_views[l_i].first.y+
								 l_views[l_j].first.z*l_views[l_i].first.z)/
								(sqrt(SQR(l_views[l_j].first.x)+SQR(l_views[l_j].first.y)+SQR(l_views[l_j].first.z))*
								 sqrt(SQR(l_views[l_i].first.x)+SQR(l_views[l_i].first.y)+SQR(l_views[l_i].first.z)));
			
				l_angle = acos(l_angle)*180.0/cv_pi;

				if( l_angle < l_min_angle )
				{
					l_min_angle = l_angle;
				}
			}
		}
		l_views[l_i].second = l_min_angle/2;
	}

	return l_views;
}

/****************************************************************************/

CvMat *	cv_gepard::get_run_intensity(	IplImage * ap_image,
										CvMat * ap_pos )
{
	CvMat * lp_int = cvCreateMat(1,ap_pos->cols,CV_32FC1);

	float * lp_col_ptr = &CV_MAT_ELEM(*ap_pos,float,0,0);
	float * lp_row_ptr = &CV_MAT_ELEM(*ap_pos,float,1,0);
	float * lp_int_ptr = &CV_MAT_ELEM(*lp_int,float,0,0);

	int l_mod = (ap_pos->cols/8)*8;
	int l_i = 0;

	for(;l_i<l_mod;)
	{		
		lp_int_ptr[0] =  CV_IMAGE_ELEM(ap_image,float,(int)lp_row_ptr[0],(int)lp_col_ptr[0]);
		lp_int_ptr[1] =  CV_IMAGE_ELEM(ap_image,float,(int)lp_row_ptr[1],(int)lp_col_ptr[1]);
		lp_int_ptr[2] =  CV_IMAGE_ELEM(ap_image,float,(int)lp_row_ptr[2],(int)lp_col_ptr[2]);
		lp_int_ptr[3] =  CV_IMAGE_ELEM(ap_image,float,(int)lp_row_ptr[3],(int)lp_col_ptr[3]);
		lp_int_ptr[4] =  CV_IMAGE_ELEM(ap_image,float,(int)lp_row_ptr[4],(int)lp_col_ptr[4]);
		lp_int_ptr[5] =  CV_IMAGE_ELEM(ap_image,float,(int)lp_row_ptr[5],(int)lp_col_ptr[5]);
		lp_int_ptr[6] =  CV_IMAGE_ELEM(ap_image,float,(int)lp_row_ptr[6],(int)lp_col_ptr[6]);
		lp_int_ptr[7] =  CV_IMAGE_ELEM(ap_image,float,(int)lp_row_ptr[7],(int)lp_col_ptr[7]);
		lp_int_ptr+=8;
		lp_col_ptr+=8;
		lp_row_ptr+=8;
		l_i+=8;
	}
	for(;l_i<ap_pos->cols;)
	{
		lp_int_ptr[0] = CV_IMAGE_ELEM(ap_image,float,(int)lp_row_ptr[0],(int)lp_col_ptr[0]);
		++lp_int_ptr;
		++lp_col_ptr;
		++lp_row_ptr;
		++l_i;
	}
	return lp_int;
}

/****************************************************************************/

CvMat *	cv_gepard::get_train_intensity(	IplImage * ap_image,
										IplImage * ap_mask,
										CvMat * ap_pos )
{
	CvMat * lp_int = cvCreateMat(1,ap_pos->cols,CV_32FC1);

	for( int l_i=0; l_i<ap_pos->cols; ++l_i )
	{
		if( CV_MAT_ELEM(*ap_pos,float,1,l_i) >= 1 &&
			CV_MAT_ELEM(*ap_pos,float,1,l_i) < ap_image->height-1 &&
			CV_MAT_ELEM(*ap_pos,float,0,l_i) >= 1 &&
			CV_MAT_ELEM(*ap_pos,float,0,l_i) < ap_image->width-1 )
		{
			if( CV_IMAGE_ELEM(	ap_mask,unsigned char,
								(int)CV_MAT_ELEM(*ap_pos,float,1,l_i),
								(int)CV_MAT_ELEM(*ap_pos,float,0,l_i)) > 0 )
			{
				CV_MAT_ELEM(*lp_int,float,0,l_i) = this->get_linear(ap_image,
												   CV_MAT_ELEM(*ap_pos,float,1,l_i),
												   CV_MAT_ELEM(*ap_pos,float,0,l_i))
												 ;//  +  rand()/(RAND_MAX+0.0)*15.0-7.5;
			}
			else
			{
				CV_MAT_ELEM(*lp_int,float,0,l_i) = rand()/(RAND_MAX+0.0)*255.0;
			}
		}
		else
		{
			CV_MAT_ELEM(*lp_int,float,0,l_i) = rand()/(RAND_MAX+0.0)*255.0;
		}
	}
	return lp_int;
}

/****************************************************************************/

float cv_gepard::get_linear(	IplImage * ap_image, 
								double a_row,
								double a_col )
{
	int l_xs0 = static_cast<int>(a_col);
	int l_ys0 = static_cast<int>(a_row);
	int l_xs1 = l_xs0+1;
	int l_ys1 = l_ys0+1;
	
	return (CV_IMAGE_ELEM(ap_image,float,l_ys0,l_xs0)*(l_xs1-a_col)+
			CV_IMAGE_ELEM(ap_image,float,l_ys0,l_xs1)*(a_col-l_xs0))*
		   (l_ys1-a_row) +
		   (CV_IMAGE_ELEM(ap_image,float,l_ys1,l_xs0)*(l_xs1-a_col)+
		    CV_IMAGE_ELEM(ap_image,float,l_ys1,l_xs1)*(a_col-l_xs0))*
		   (a_row-l_ys0);
}

/****************************************************************************/

CvMat *	cv_gepard::create_positions(	int a_nx, 
										int a_ny,
										int a_height,
										int a_width )
{
	float l_col = -a_width/2;
	float l_row = -a_height/2;

	const float l_stepx = a_width/(a_nx-1.0);
	const float l_stepy = a_height/(a_ny-1.0);

	CvMat * lp_positions = cvCreateMat(3,a_nx*a_ny,CV_32FC1);

	int l_n=0;

	for( int l_i=0; l_i<a_nx; ++l_i )
	{
		for( int l_j=0; l_j<a_ny; ++l_j )
		{
			CV_MAT_ELEM(*lp_positions,float,0,l_n) = l_col+l_i*l_stepx;
			CV_MAT_ELEM(*lp_positions,float,1,l_n) = l_row+l_j*l_stepy;
			CV_MAT_ELEM(*lp_positions,float,2,l_n) = 1.0;
			++l_n;
		}
	}
	return lp_positions;
}

/****************************************************************************/

std::ofstream & cv_gepard::write( std::ofstream & a_os )
{
	int l_num_of_poses = m_poses.size();

	a_os.write((char*)&m_nx,sizeof(m_nx));
	a_os.write((char*)&m_ny,sizeof(m_ny));
	a_os.write((char*)&m_size,sizeof(m_size));
	a_os.write((char*)&l_num_of_poses,sizeof(l_num_of_poses));
	a_os.write((char*)&m_num_of_samples,sizeof(m_num_of_samples));

	cv_write(a_os,mp_k);
	cv_write(a_os,mp_pos);
	
	for( int l_i=0; l_i<l_num_of_poses; ++l_i )
	{
		cv_write(a_os,m_poses[l_i]);
		cv_write(a_os,m_means[l_i]);
	}
	return a_os;
}

/****************************************************************************/

std::ifstream & cv_gepard::read( std::ifstream & a_is )
{
	this->clear();

	int l_num_of_poses;

	a_is.read((char*)&m_nx,sizeof(m_nx));
	a_is.read((char*)&m_ny,sizeof(m_ny));
	a_is.read((char*)&m_size,sizeof(m_size));
	a_is.read((char*)&l_num_of_poses,sizeof(l_num_of_poses));
	a_is.read((char*)&m_num_of_samples,sizeof(m_num_of_samples));

	mp_k = cv_read(a_is);
	mp_pos = cv_read(a_is);
	

	for( int l_i=0; l_i<l_num_of_poses; ++l_i )
	{
		CvMat * lp_pose = cv_read(a_is);
		CvMat * lp_mean = cv_read(a_is);
	
		m_poses.push_back(lp_pose);
		m_means.push_back(lp_mean);
	}
	return a_is;
}

/****************************************************************************/

bool cv_gepard::save( std::string a_name )
{
	std::ofstream l_file(a_name.c_str(),std::ofstream::out|std::ofstream::binary);
	
	if( l_file.fail() == true )
	{
		printf("cv_gepard: could not open for writing!");
		return false;
	}
	this->write(l_file);	

	l_file.close();

	return true;
}

/****************************************************************************/

bool cv_gepard::load( std::string a_name )
{
	std::ifstream l_file(a_name.c_str(),std::ifstream::in|std::ifstream::binary);
	
	if( l_file.fail() == true ) 
	{
		printf("cv_gepard: could not open for reading!");
		return false;
	}
	this->read(l_file);	

	l_file.close();

	return true;
}

/****************************************************************************/

std::pair<float,float> cv_gepard::compare( cv_gepard * ap_gepard )
{
	if( m_means.size() != ap_gepard->m_means.size() )
	{
		std::cerr << "error gepard: not the same mean size: " << m_means.size() << "," << ap_gepard->m_means.size() << std::endl;
		return std::pair<float,float>(-100,-100); 
	}
	float l_all = 0.0;
	float l_dif = 0.0;
	float l_cor = 0.0;
	float l_rel = 0.0;

	int l_index=0;

	//std::cerr << "num of means: " << m_means.size() << std::endl;

	for( int l_i=0; l_i<m_means.size(); ++l_i )
	{
		if( m_means[l_i]->rows != ap_gepard->m_means[l_i]->rows ||
			m_means[l_i]->cols != ap_gepard->m_means[l_i]->cols )
		{
			std::cerr << "error gepard: not the same mean mean size" << std::endl;
			return std::pair<float,float>(-200,-200); 
			
		}
		float l_var =	fabs(CV_MAT_ELEM(*m_poses[l_i],float,0,0)-CV_MAT_ELEM(*ap_gepard->m_poses[l_i],float,0,0))+
						fabs(CV_MAT_ELEM(*m_poses[l_i],float,1,0)-CV_MAT_ELEM(*ap_gepard->m_poses[l_i],float,1,0))+
						fabs(CV_MAT_ELEM(*m_poses[l_i],float,0,1)-CV_MAT_ELEM(*ap_gepard->m_poses[l_i],float,0,1))+
						fabs(CV_MAT_ELEM(*m_poses[l_i],float,1,1)-CV_MAT_ELEM(*ap_gepard->m_poses[l_i],float,1,1))+
						fabs(CV_MAT_ELEM(*m_poses[l_i],float,0,2)-CV_MAT_ELEM(*ap_gepard->m_poses[l_i],float,0,2))+
						fabs(CV_MAT_ELEM(*m_poses[l_i],float,1,2)-CV_MAT_ELEM(*ap_gepard->m_poses[l_i],float,1,2))+
						fabs(CV_MAT_ELEM(*m_poses[l_i],float,0,3)-CV_MAT_ELEM(*ap_gepard->m_poses[l_i],float,0,3))+
						fabs(CV_MAT_ELEM(*m_poses[l_i],float,1,3)-CV_MAT_ELEM(*ap_gepard->m_poses[l_i],float,1,3));

		if( l_var > 1 )
		{
			cv_print(m_poses[l_i]);
			cv_print(ap_gepard->m_poses[l_i]);
			std::cerr << "error gepard: pose " << l_i << " is different about: " << l_var << std::endl;
			return std::pair<float,float>(-300,-300); 
		}

		CvScalar l_amean1 = cvAvg(m_means[l_i]);
		CvScalar l_amean2 = cvAvg(ap_gepard->m_means[l_i]);

		float l_mean1 = l_amean1.val[0];
		float l_mean2 = l_amean2.val[0];

		//std::cerr << "mean: " << l_mean1 << "," << l_mean2 << std::endl;

		float l_pos_all=0;
		float l_pos_dif=0;

		for( int l_r=0; l_r<m_means[l_i]->rows; ++l_r )
		{
			for( int l_c=0; l_c<m_means[l_i]->cols; ++l_c )
			{
				float l_sim_all = fabs(CV_MAT_ELEM(*m_means[l_i],float,l_r,l_c));
				float l_sim_dif = fabs(CV_MAT_ELEM(*m_means[l_i],float,l_r,l_c)-CV_MAT_ELEM(*ap_gepard->m_means[l_i],float,l_r,l_c)); 

				l_pos_all += l_sim_all;
				l_pos_dif += l_sim_dif;

				float l_ccc = (CV_MAT_ELEM(*m_means[l_i],float,l_r,l_c)-l_mean1)*(CV_MAT_ELEM(*ap_gepard->m_means[l_i],float,l_r,l_c)-l_mean2);

				l_cor+=l_ccc;

				l_rel += l_sim_dif/l_sim_all;

				l_index++;
				//std::cerr << l_pos_all << "," << l_pos_dif << "," << l_sim_dif/l_sim_all << std::endl;
				//cvWaitKey(-1);
			}
		}
		//std::cerr << l_i % 16 << "," << l_pos_all << "," << l_pos_dif << "," << l_pos_dif/l_pos_all << std::endl;
		//cvWaitKey(-1);

		l_all += l_pos_all;
		l_dif += l_pos_dif;
	}
	//std::cerr << "finished: " << l_dif/l_all << " correlation: " << l_cor/l_index << " relativ: " << l_rel/l_index << std::endl;

	std::pair<float,float> l_val(l_dif/l_all,l_cor/l_index);

	return l_val;
}

/**************************** END OF FILE ***********************************/
