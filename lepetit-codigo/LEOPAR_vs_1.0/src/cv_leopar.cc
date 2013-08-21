//////////////////////////////////////////////////////////////////////////////
//
// cv_leopar: cv_leopar.cc
//
// Authors: Stefan Hinterstoisser 2008
// Lehrstuhl fuer Informatik XVI
// Technische Universitaet Muenchen
// Version: 1.0
//
//////////////////////////////////////////////////////////////////////////////

/******************************* includes ***********************************/

#include "cv_leopar.h"
#include <iostream>

/******************************** defines ***********************************/

/******************************* namespaces *********************************/

using namespace cv;

/****************************** constructors ********************************/

cv_leopar::cv_leopar( void )
{
	m_magic_factor = 0.8;

	m_size = 0;
	m_depth = 0;
	m_num_of_leaves = 0;
	m_num_of_ferns = 0;
	m_num_of_poses = 0;
	m_num_of_points = 0;
		
	mp_poses = NULL;
	mp_points = NULL;
	mp_pose_distributions1 = NULL;
	mp_pose_distributions2 = NULL;
	mp_pose_distributions3 = NULL;
	mp_pose_distributions4 = NULL;
}

/******************************** destructor ********************************/

cv_leopar::~cv_leopar()
{
	cvReleaseMat(&mp_poses);
	cvReleaseMat(&mp_points);
	cvReleaseMat(&mp_pose_distributions1);
	cvReleaseMat(&mp_pose_distributions2);
	cvReleaseMat(&mp_pose_distributions3);
	cvReleaseMat(&mp_pose_distributions4);
}

/****************************************************************************/
	
void cv_leopar::set_parameters( int a_size,
								int a_depth,
								int a_num_of_ferns,
								int a_num_of_poses,
								int a_num_of_samples )
{
	if( a_size < 0 )
	{
		printf("cv_leopar: a_size is not appropriate!");
		return;
	}
	if( a_depth <= 0 )
	{
		printf("cv_leopar: a_depth should be over 0!");
		return;
	}
	if( a_num_of_ferns <= 0 )
	{
		printf("cv_leopar: a_num_of_ferns should be over 0!");
		return;
	}
	if( a_num_of_poses <= 0 )
	{
		printf("cv_leopar: a_num_of_poses should be over 0!");
		return;
	}
	if( a_num_of_samples <= 0 )
	{
		printf("cv_leopar: a_a_num_of_smaples should be over 0!");
		return;
	}
	cvReleaseMat( &mp_poses );
	cvReleaseMat( &mp_points );
	cvReleaseMat( &mp_pose_distributions1 );
	cvReleaseMat( &mp_pose_distributions2 );
	cvReleaseMat( &mp_pose_distributions3 );
	cvReleaseMat( &mp_pose_distributions4 );

	m_size = a_size;
	m_depth = a_depth;
	m_num_of_ferns = a_num_of_ferns;
	m_num_of_poses = a_num_of_poses;

	m_num_of_leaves = pow(2.0,m_depth);
	m_num_of_points = m_num_of_ferns*m_depth*2;

	m_num_of_samples = a_num_of_samples;
	
	mp_poses = cvCreateMat(2,m_num_of_poses,CV_32FC1);
	mp_points = cvCreateMat(2,m_num_of_points,CV_32SC1);

	int l_num_of_bins = m_num_of_poses*m_num_of_leaves*m_num_of_ferns;
	
	mp_pose_distributions1 = cvCreateMat(1,l_num_of_bins,CV_32FC1);
	mp_pose_distributions2 = cvCreateMat(1,l_num_of_bins,CV_32FC1);
	mp_pose_distributions3 = cvCreateMat(1,l_num_of_bins,CV_32FC1);
	mp_pose_distributions4 = cvCreateMat(1,l_num_of_bins,CV_32FC1);
}

/****************************************************************************/

int * cv_leopar::get_closest_bins(	CvMat * ap_dest_quad,
									int a_row,
									int a_col )
{
	int * lp_bin = new int[4];

	for( int l_i=0; l_i<4; ++l_i )
	{
		float l_best_val=10e10;

		for( int l_j=0; l_j<m_num_of_poses; ++l_j )
		{
			float l_val =	(CV_MAT_ELEM(*mp_poses,float,0,l_j)-
							(CV_MAT_ELEM(*ap_dest_quad,float,0,l_i)-a_col))*
							(CV_MAT_ELEM(*mp_poses,float,0,l_j)-
							(CV_MAT_ELEM(*ap_dest_quad,float,0,l_i)-a_col))+
							(CV_MAT_ELEM(*mp_poses,float,1,l_j)-
							(CV_MAT_ELEM(*ap_dest_quad,float,1,l_i)-a_row))*
							(CV_MAT_ELEM(*mp_poses,float,1,l_j)-
							(CV_MAT_ELEM(*ap_dest_quad,float,1,l_i)-a_row));

			if( l_val < l_best_val )
			{
				l_best_val = l_val;
				lp_bin[l_i] = l_j;
			}
		}
	}
	return lp_bin;
}
	
/****************************************************************************/

std::vector<CvMat*> cv_leopar::get_closest_pose(	CvMat * ap_dest_quad,
													int a_row,
													int a_col )
{
	std::vector<CvMat*> l_vector;

	if( ap_dest_quad == NULL )
	{
		printf("cv_leopar: ap_dest_quad in closest pose is NULL!");
		return l_vector;
	}
	int * lp_bin = this->get_closest_bins(ap_dest_quad,a_row,a_col);

	CvMat * lp_closest = cvCreateMat(3,4,CV_32FC1);

	CV_MAT_ELEM(*lp_closest,float,0,0) = CV_MAT_ELEM(*mp_poses,float,0,lp_bin[0])+a_col;
	CV_MAT_ELEM(*lp_closest,float,1,0) = CV_MAT_ELEM(*mp_poses,float,1,lp_bin[0])+a_row;
	CV_MAT_ELEM(*lp_closest,float,0,1) = CV_MAT_ELEM(*mp_poses,float,0,lp_bin[1])+a_col;
	CV_MAT_ELEM(*lp_closest,float,1,1) = CV_MAT_ELEM(*mp_poses,float,1,lp_bin[1])+a_row;
	CV_MAT_ELEM(*lp_closest,float,0,2) = CV_MAT_ELEM(*mp_poses,float,0,lp_bin[2])+a_col;
	CV_MAT_ELEM(*lp_closest,float,1,2) = CV_MAT_ELEM(*mp_poses,float,1,lp_bin[2])+a_row;
	CV_MAT_ELEM(*lp_closest,float,0,3) = CV_MAT_ELEM(*mp_poses,float,0,lp_bin[3])+a_col;
	CV_MAT_ELEM(*lp_closest,float,1,3) = CV_MAT_ELEM(*mp_poses,float,1,lp_bin[3])+a_row;
	CV_MAT_ELEM(*lp_closest,float,2,0) = 1.0f;
	CV_MAT_ELEM(*lp_closest,float,2,1) = 1.0f;
	CV_MAT_ELEM(*lp_closest,float,2,2) = 1.0f;
	CV_MAT_ELEM(*lp_closest,float,2,3) = 1.0f;

	delete[] lp_bin;

	l_vector.push_back(lp_closest);

	return l_vector;
}

/****************************************************************************/

bool cv_leopar::learn(	IplImage * ap_image,
						IplImage * ap_mask,
						int a_row,
						int a_col )
{
	if( ap_image == NULL )
	{
		printf("cv_leopar: ap_image is NULL!");
		return false;
	}
	if( ap_image->imageData == NULL )
	{
		printf("cv_leopar: ap_image->imageData is NULL!");
		return false;
	}
	if( ap_mask == NULL )
	{
		printf("cv_leopar: ap_mask is NULL!");
		return false;
	}
	if( ap_mask->imageData == NULL )
	{
		printf("cv_leopar: ap_mask->imageData is NULL!");
		return false;
	}
	if( cvGetElemType(ap_image) != CV_32FC1 )
	{
		printf("cv_leopar: type is not 32FC1!");
		return false;
	}
	if( a_row < 0 ||
		a_col < 0 ||
		a_row+m_size/2 >= ap_image->height ||
		a_col+m_size/2 >= ap_image->width )
	{
		printf("cv_leopar: a_row or a_col are not appropriate!");
		return false;
	}
	CvMat * lp_dest_quad = cvCreateMat(3,4,CV_32FC1);

	CV_MAT_ELEM(*lp_dest_quad,float,0,0) = a_col-m_size/2;
	CV_MAT_ELEM(*lp_dest_quad,float,1,0) = a_row-m_size/2;
	CV_MAT_ELEM(*lp_dest_quad,float,0,1) = a_col+m_size/2;
	CV_MAT_ELEM(*lp_dest_quad,float,1,1) = a_row-m_size/2;
	CV_MAT_ELEM(*lp_dest_quad,float,0,2) = a_col+m_size/2;
	CV_MAT_ELEM(*lp_dest_quad,float,1,2) = a_row+m_size/2;
	CV_MAT_ELEM(*lp_dest_quad,float,0,3) = a_col-m_size/2;
	CV_MAT_ELEM(*lp_dest_quad,float,1,3) = a_row+m_size/2;

	this->learn_initialize();
	bool l_flag = this->learn_incremental(	ap_image,
											ap_mask,
											lp_dest_quad,
											m_num_of_samples,
											a_row,
											a_col );
	this->learn_finalize();
	cvReleaseMat(&lp_dest_quad);

	return l_flag;
}

/****************************************************************************/

void cv_leopar::learn_initialize( void )
{
	float l_prior = 1.0f;

	cvSet(mp_pose_distributions1,cvRealScalar(l_prior));
	cvSet(mp_pose_distributions2,cvRealScalar(l_prior));
	cvSet(mp_pose_distributions3,cvRealScalar(l_prior));
	cvSet(mp_pose_distributions4,cvRealScalar(l_prior));

	this->fix_tests(mp_points,m_num_of_ferns,m_depth,m_size*m_magic_factor);
	this->fix_poses(mp_poses,m_num_of_poses,m_size*m_magic_factor);
}

/****************************************************************************/

CvMat * cv_leopar::gen_rand_aff_trans(	float a_row,
										float a_col,
										float a_phi,
										float a_lam )
{
	double l_lam1 = rand()/(RAND_MAX+0.0)*a_lam-a_lam/2+1.0;
	double l_lam2 = rand()/(RAND_MAX+0.0)*a_lam-a_lam/2+1.0;
	double l_the  = rand()/(RAND_MAX+0.0)*360.0;
	double l_phi  = rand()/(RAND_MAX+0.0)*a_phi-a_phi/2;
	double l_ver  = rand()/(RAND_MAX+0.0)*6-3;
	double l_hor  = rand()/(RAND_MAX+0.0)*6-3;

	l_the *= cv_pi/180.0;
	l_phi *= cv_pi/180.0;

	CvMat * lp_aff_mat = cvCreateMat(3,3,CV_32FC1);
	CvMat * lp_sca_mat = cvCreateMat(3,3,CV_32FC1);
	CvMat * lp_tro_mat = cvCreateMat(3,3,CV_32FC1);
	CvMat * lp_ppr_mat = cvCreateMat(3,3,CV_32FC1);
	CvMat * lp_mpr_mat = cvCreateMat(3,3,CV_32FC1);
	CvMat * lp_tmp_mat1 = cvCreateMat(3,3,CV_32FC1);
	CvMat * lp_tmp_mat2 = cvCreateMat(3,3,CV_32FC1);

	cvSet(lp_sca_mat,cvRealScalar(0));
	cvSet(lp_tro_mat,cvRealScalar(0));
	cvSet(lp_ppr_mat,cvRealScalar(0));
	cvSet(lp_mpr_mat,cvRealScalar(0));

	CV_MAT_ELEM(*lp_sca_mat,float,0,0) = l_lam1;
	CV_MAT_ELEM(*lp_sca_mat,float,1,1) = l_lam2;
	CV_MAT_ELEM(*lp_sca_mat,float,2,2) = 1.0f;

	CV_MAT_ELEM(*lp_tro_mat,float,0,0) = cos(l_the);
	CV_MAT_ELEM(*lp_tro_mat,float,0,1) = sin(l_the);
	CV_MAT_ELEM(*lp_tro_mat,float,1,0) = -sin(l_the);
	CV_MAT_ELEM(*lp_tro_mat,float,1,1) = cos(l_the);
	CV_MAT_ELEM(*lp_tro_mat,float,2,2) = 1.0f;

	CV_MAT_ELEM(*lp_ppr_mat,float,0,0) = cos(l_phi);
	CV_MAT_ELEM(*lp_ppr_mat,float,0,1) = sin(l_phi);
	CV_MAT_ELEM(*lp_ppr_mat,float,1,0) = -sin(l_phi);
	CV_MAT_ELEM(*lp_ppr_mat,float,1,1) = cos(l_phi);
	CV_MAT_ELEM(*lp_ppr_mat,float,2,2) = 1.0f;

	CV_MAT_ELEM(*lp_mpr_mat,float,0,0) = cos(-l_phi);
	CV_MAT_ELEM(*lp_mpr_mat,float,0,1) = sin(-l_phi);
	CV_MAT_ELEM(*lp_mpr_mat,float,1,0) = -sin(-l_phi);
	CV_MAT_ELEM(*lp_mpr_mat,float,1,1) = cos(-l_phi);
	CV_MAT_ELEM(*lp_mpr_mat,float,2,2) = 1.0f;

	cvMatMul(lp_sca_mat,lp_ppr_mat,lp_tmp_mat1);
	cvMatMul(lp_mpr_mat,lp_tmp_mat1,lp_tmp_mat2);
	cvMatMul(lp_tro_mat,lp_tmp_mat2,lp_aff_mat);

	float l_tcol =	-(CV_MAT_ELEM(*lp_aff_mat,float,0,0)*a_col+
					CV_MAT_ELEM(*lp_aff_mat,float,0,1)*a_row)+
					a_col;
	float l_trow =	-(CV_MAT_ELEM(*lp_aff_mat,float,1,0)*a_col+
					CV_MAT_ELEM(*lp_aff_mat,float,1,1)*a_row)+
					a_row;
					
	CV_MAT_ELEM(*lp_aff_mat,float,0,2) = l_tcol+l_hor;
	CV_MAT_ELEM(*lp_aff_mat,float,1,2) = l_trow+l_ver;
	
	cvReleaseMat(&lp_tmp_mat1);
    cvReleaseMat(&lp_tmp_mat2);
	cvReleaseMat(&lp_sca_mat);
	cvReleaseMat(&lp_tro_mat);
	cvReleaseMat(&lp_ppr_mat);
	cvReleaseMat(&lp_mpr_mat);

	return lp_aff_mat;
}

/****************************************************************************/

bool cv_leopar::learn_incremental(	IplImage * ap_image,
									IplImage * ap_mask,
									CvMat * ap_dest_quad,
									int a_samples,
									int a_row,
									int a_col )
{
	if( ap_dest_quad == NULL )
	{
		printf("cv_leopar: ap_dest_quad in learning is NULL!");
		return false;
	}
	if( ap_image == NULL )
	{
		printf("cv_leopar: ap_image is NULL!");
		return false;
	}
	if( ap_image->imageData == NULL )
	{
		printf("cv_leopar: ap_image->imageData is NULL!");
		return false;
	}
	if( cvGetElemType(ap_image) != CV_32FC1 )
	{
		printf("cv_leopar: type is not 32FC1!");
		return false;
	}
	if( m_size <= 0 )
	{
		printf("cv_leopar: width or height are smaller 0!");
		return false;
	}
	if( a_row < 0 ||
		a_col < 0 ||
		a_row+m_size >= ap_image->height ||
		a_col+m_size  >= ap_image->width )
	{
		printf("cv_leopar: rows and cols are not appropriate!");
		return false;
	}
	CvMat * lp_dest_quad = cvCreateMat(3,4,CV_32FC1);
	CvMat * lp_temp_quad = cvCreateMat(3,4,CV_32FC1);
	
	int l_border = 4;
	float l_var = 10.0;
	
	CV_MAT_ELEM(*ap_dest_quad,float,2,0) = 1;
	CV_MAT_ELEM(*ap_dest_quad,float,2,1) = 1;
	CV_MAT_ELEM(*ap_dest_quad,float,2,2) = 1;
	CV_MAT_ELEM(*ap_dest_quad,float,2,3) = 1;
	
	cvCopy(ap_dest_quad,lp_dest_quad);

	CV_MAT_ELEM(*lp_dest_quad,float,0,0) -= a_col;
	CV_MAT_ELEM(*lp_dest_quad,float,1,0) -= a_row;
	CV_MAT_ELEM(*lp_dest_quad,float,0,1) -= a_col;
	CV_MAT_ELEM(*lp_dest_quad,float,1,1) -= a_row;
	CV_MAT_ELEM(*lp_dest_quad,float,0,2) -= a_col;
	CV_MAT_ELEM(*lp_dest_quad,float,1,2) -= a_row;
	CV_MAT_ELEM(*lp_dest_quad,float,0,3) -= a_col;
	CV_MAT_ELEM(*lp_dest_quad,float,1,3) -= a_row;

	IplImage * lp_patch = cvCreateImage(cvSize(	m_size*m_magic_factor+l_border,
												m_size*m_magic_factor+l_border),
												IPL_DEPTH_32F,1);
	for( int l_s=0; l_s<a_samples; ++l_s )
	{
		CvMat * lp_rand_aff= this->gen_rand_aff_trans(	a_row,
														a_col,
														180.0,
														1.0);
		this->set_aff(	ap_image,
						ap_mask,
						lp_patch,
						mp_points,
						lp_rand_aff,
						a_row,
						a_col,
						l_var);

		CV_MAT_ELEM(*lp_rand_aff,float,0,2) = 0;
		CV_MAT_ELEM(*lp_rand_aff,float,1,2) = 0;

		cvMatMul(lp_rand_aff,lp_dest_quad,lp_temp_quad);

		int * lp_bin_index = this->get_closest_bins(lp_temp_quad,0,0);
		int * lp_leaves_index = this->drop_patch(	lp_patch,
													mp_points,
													lp_patch->height/2,
													lp_patch->width/2,
													m_num_of_ferns,
													m_depth);
 		for( int l_k=0; l_k<m_num_of_ferns; ++l_k )
		{
			int l_index = (l_k*m_num_of_leaves+lp_leaves_index[l_k])*m_num_of_poses;
								
			CV_MAT_ELEM(*mp_pose_distributions1,float,0,l_index+lp_bin_index[0]) += 1;
			CV_MAT_ELEM(*mp_pose_distributions2,float,0,l_index+lp_bin_index[1]) += 1;
			CV_MAT_ELEM(*mp_pose_distributions3,float,0,l_index+lp_bin_index[2]) += 1;
			CV_MAT_ELEM(*mp_pose_distributions4,float,0,l_index+lp_bin_index[3]) += 1;
		}
		delete[] lp_bin_index;
		delete[] lp_leaves_index;
		cvReleaseMat(&lp_rand_aff);
	}
	cvReleaseImage(&lp_patch);
	cvReleaseMat(&lp_dest_quad);
	cvReleaseMat(&lp_temp_quad);

	return true;
}
	
/****************************************************************************/

void cv_leopar::set_aff(	IplImage * ap_image,
							IplImage * ap_mask,
							IplImage * ap_patch,
							CvMat * ap_pos,
							CvMat * ap_aff,
							float a_row,
							float a_col,
							float a_var )
{
	CvMat * lp_inverse = cvCreateMat(3,3,CV_32FC1);
	cvInvert(ap_aff,lp_inverse,CV_SVD);

	for( int l_c=0; l_c<ap_pos->cols; ++l_c )
	{
		float l_col = CV_MAT_ELEM(*ap_pos,int,0,l_c)+a_col;
		float l_row = CV_MAT_ELEM(*ap_pos,int,1,l_c)+a_row;

		float l_icol =	CV_MAT_ELEM(*lp_inverse,float,0,0)*l_col+
						CV_MAT_ELEM(*lp_inverse,float,0,1)*l_row+
						CV_MAT_ELEM(*lp_inverse,float,0,2);
		float l_irow =	CV_MAT_ELEM(*lp_inverse,float,1,0)*l_col+
						CV_MAT_ELEM(*lp_inverse,float,1,1)*l_row+
						CV_MAT_ELEM(*lp_inverse,float,1,2);

		float l_val = 0;

		if( l_irow >= 1 &&
			l_irow < ap_image->height-1 &&
			l_icol >= 1 &&
			l_icol < ap_image->width-1 )
		{
			if( CV_IMAGE_ELEM(	ap_mask,
								unsigned char,
								(int)(l_irow),
								(int)(l_icol)) > 0 )
			{
				l_val = this->get_linear(ap_image,l_irow,l_icol)+ 
						(rand()*a_var)/(RAND_MAX)-a_var/2.0;
			}
			else
			{
				l_val = (rand()*255.0)/(RAND_MAX);
			}
		}
		else
		{
			l_val = (rand()*255.0)/(RAND_MAX);
		}
		CV_IMAGE_ELEM(	ap_patch,float,
						CV_MAT_ELEM(*ap_pos,int,1,l_c)+ap_patch->height/2,
						CV_MAT_ELEM(*ap_pos,int,0,l_c)+ap_patch->width/2) = l_val;
	}
	cvReleaseMat(&lp_inverse);
}

/****************************************************************************/

float cv_leopar::get_linear(	IplImage * ap_image, 
								double a_row,
								double a_col )
{
	int l_xs0 = static_cast<int>(a_col);
	int l_ys0 = static_cast<int>(a_row);
	int l_xs1 = l_xs0+1;
	int l_ys1 = l_ys0+1;
	
	return (CV_IMAGE_ELEM(ap_image,float,l_ys0,l_xs0)*(l_xs1-a_col)+
			CV_IMAGE_ELEM(ap_image,float,l_ys0,l_xs1)*(a_col-l_xs0))*(l_ys1-a_row) +
		   (CV_IMAGE_ELEM(ap_image,float,l_ys1,l_xs0)*(l_xs1-a_col)+
		    CV_IMAGE_ELEM(ap_image,float,l_ys1,l_xs1)*(a_col-l_xs0))*(a_row-l_ys0);
}

/****************************************************************************/

void cv_leopar::learn_finalize( void )
{
	for( int l_f=0; l_f<m_num_of_ferns; ++l_f )
	{
		for( int l_l=0; l_l<m_num_of_leaves; ++l_l )
		{
			float l_s1=0;
			float l_s2=0;
			float l_s3=0;
			float l_s4=0;

			int l_index = (l_f*m_num_of_leaves+l_l)*m_num_of_poses;

			for( int l_s=0; l_s<m_num_of_poses; ++l_s )
			{
				l_s1 += CV_MAT_ELEM(*mp_pose_distributions1,float,0,l_index+l_s);
				l_s2 += CV_MAT_ELEM(*mp_pose_distributions2,float,0,l_index+l_s);
				l_s3 += CV_MAT_ELEM(*mp_pose_distributions3,float,0,l_index+l_s);
				l_s4 += CV_MAT_ELEM(*mp_pose_distributions4,float,0,l_index+l_s);
			}
			for( int l_s=0; l_s<m_num_of_poses; ++l_s )
			{
				CV_MAT_ELEM(*mp_pose_distributions1,float,0,l_index+l_s) = 
				log(CV_MAT_ELEM(*mp_pose_distributions1,float,0,l_index+l_s)/l_s1);
				CV_MAT_ELEM(*mp_pose_distributions2,float,0,l_index+l_s) =
				log(CV_MAT_ELEM(*mp_pose_distributions2,float,0,l_index+l_s)/l_s2);
				CV_MAT_ELEM(*mp_pose_distributions3,float,0,l_index+l_s) = 
				log(CV_MAT_ELEM(*mp_pose_distributions3,float,0,l_index+l_s)/l_s3);
				CV_MAT_ELEM(*mp_pose_distributions4,float,0,l_index+l_s) =
				log(CV_MAT_ELEM(*mp_pose_distributions4,float,0,l_index+l_s)/l_s4);
			}
		}
	}
}

/****************************************************************************/

CvMat * cv_leopar::recognize(	IplImage * ap_image, 
								int a_row, 
								int a_col )
{
	if( ap_image == NULL )
	{
		printf("cv_leopar: recognition - ap_image is NULL!");
		return NULL;
	}
	if( ap_image->imageData == NULL )
	{
		printf("cv_leopar: recognition - ap_image->imageData is NULL!");
		return NULL;
	}
	if( a_row-m_size/2-1 < 0 ||
		a_col-m_size/2-1 < 0 ||
		a_row+m_size/2+1 >= ap_image->height ||
		a_col+m_size/2+1 >= ap_image->width  )
	{
		printf("cv_leopar: col or row is not appropriate!");
		return NULL;
	}
	int * lp_leaves_index = this->drop_patch(ap_image,mp_points,a_row, a_col,m_num_of_ferns,m_depth);

	CvMat * lp_pose_distributions1 = cvCreateMat(1,m_num_of_poses,CV_32FC1);
	CvMat * lp_pose_distributions2 = cvCreateMat(1,m_num_of_poses,CV_32FC1);
	CvMat * lp_pose_distributions3 = cvCreateMat(1,m_num_of_poses,CV_32FC1);
	CvMat * lp_pose_distributions4 = cvCreateMat(1,m_num_of_poses,CV_32FC1);

	cvSet(lp_pose_distributions1,cvRealScalar(0));
	cvSet(lp_pose_distributions2,cvRealScalar(0));
	cvSet(lp_pose_distributions3,cvRealScalar(0));
	cvSet(lp_pose_distributions4,cvRealScalar(0));
	
	this->add_distributions(mp_pose_distributions1,lp_pose_distributions1,lp_leaves_index,
							m_num_of_ferns,m_num_of_leaves,m_num_of_poses);
	this->add_distributions(mp_pose_distributions2,lp_pose_distributions2,lp_leaves_index, 
							m_num_of_ferns,m_num_of_leaves,m_num_of_poses);
	this->add_distributions(mp_pose_distributions3,lp_pose_distributions3,lp_leaves_index,
							m_num_of_ferns,m_num_of_leaves,m_num_of_poses);
	this->add_distributions(mp_pose_distributions4,lp_pose_distributions4,lp_leaves_index,
							m_num_of_ferns,m_num_of_leaves,m_num_of_poses);

	int l_best_pose_index1 = this->find_max(lp_pose_distributions1,m_num_of_poses);
	int l_best_pose_index2 = this->find_max(lp_pose_distributions2,m_num_of_poses);
	int l_best_pose_index3 = this->find_max(lp_pose_distributions3,m_num_of_poses);
	int l_best_pose_index4 = this->find_max(lp_pose_distributions4,m_num_of_poses);

    cvReleaseMat(&lp_pose_distributions1);
	cvReleaseMat(&lp_pose_distributions2);
	cvReleaseMat(&lp_pose_distributions3);
	cvReleaseMat(&lp_pose_distributions4);
	delete[] lp_leaves_index;

	CvMat * lp_result_quad = cvCreateMat(3,4,CV_32FC1);

	CV_MAT_ELEM(*lp_result_quad,float,0,0) = CV_MAT_ELEM(*mp_poses,float,0,l_best_pose_index1)+a_col;
	CV_MAT_ELEM(*lp_result_quad,float,1,0) = CV_MAT_ELEM(*mp_poses,float,1,l_best_pose_index1)+a_row;
	CV_MAT_ELEM(*lp_result_quad,float,0,1) = CV_MAT_ELEM(*mp_poses,float,0,l_best_pose_index2)+a_col;
	CV_MAT_ELEM(*lp_result_quad,float,1,1) = CV_MAT_ELEM(*mp_poses,float,1,l_best_pose_index2)+a_row;
	CV_MAT_ELEM(*lp_result_quad,float,0,2) = CV_MAT_ELEM(*mp_poses,float,0,l_best_pose_index3)+a_col;
	CV_MAT_ELEM(*lp_result_quad,float,1,2) = CV_MAT_ELEM(*mp_poses,float,1,l_best_pose_index3)+a_row;
	CV_MAT_ELEM(*lp_result_quad,float,0,3) = CV_MAT_ELEM(*mp_poses,float,0,l_best_pose_index4)+a_col;
	CV_MAT_ELEM(*lp_result_quad,float,1,3) = CV_MAT_ELEM(*mp_poses,float,1,l_best_pose_index4)+a_row;
	CV_MAT_ELEM(*lp_result_quad,float,2,0) = 1.0f;
	CV_MAT_ELEM(*lp_result_quad,float,2,1) = 1.0f;
	CV_MAT_ELEM(*lp_result_quad,float,2,2) = 1.0f;
	CV_MAT_ELEM(*lp_result_quad,float,2,3) = 1.0f;

	return lp_result_quad;
}

/****************************************************************************/

void cv_leopar::add_distributions(	CvMat * ap_src_distributions,
									CvMat * ap_dst_distributions,
									int * ap_leaves_index,
									int a_num_of_ferns,
									int a_num_of_leaves,
									int a_num_of_poses )
{
	int l_pose_mod = (a_num_of_poses/8)*8;

	for( int l_f=0; l_f<a_num_of_ferns; ++l_f )
	{
		int l_index = (l_f*a_num_of_leaves+ap_leaves_index[l_f])*a_num_of_poses;

		float * lp_init_pointer = &CV_MAT_ELEM(*ap_src_distributions,float,0,l_index);
		float * lp_dest_pointer = &CV_MAT_ELEM(*ap_dst_distributions,float,0,0);

		int l_j=0;

		for(;l_j<l_pose_mod;)
		{
			lp_dest_pointer[0] += lp_init_pointer[0];
			lp_dest_pointer[1] += lp_init_pointer[1];
			lp_dest_pointer[2] += lp_init_pointer[2];
			lp_dest_pointer[3] += lp_init_pointer[3];
			lp_dest_pointer[4] += lp_init_pointer[4];
			lp_dest_pointer[5] += lp_init_pointer[5];
			lp_dest_pointer[6] += lp_init_pointer[6];
			lp_dest_pointer[7] += lp_init_pointer[7];
			
			lp_init_pointer+=8;
			lp_dest_pointer+=8;
			l_j+=8;
		}
		for(;l_j<a_num_of_poses;)
		{
			lp_dest_pointer[0] += lp_init_pointer[0];
	
			lp_init_pointer+=1;
			lp_dest_pointer+=1;
			++l_j;
		}
	}
}

/****************************************************************************/

int cv_leopar::find_max( CvMat * ap_distributions, int a_num_of_poses )
{
	int l_best_index = 0;
	int l_best_value = CV_MAT_ELEM(*ap_distributions,float,0,0);
	
	for( int l_i=a_num_of_poses-1; l_i>=0; --l_i )
	{
		if( l_best_value < CV_MAT_ELEM(*ap_distributions,float,0,l_i) )
		{
			l_best_value = CV_MAT_ELEM(*ap_distributions,float,0,l_i);
			l_best_index = l_i;
		}
	}
	return l_best_index;
}

/****************************************************************************/

void cv_leopar::fix_tests(	CvMat * ap_points, 
							int a_num_of_ferns,
							int a_depth,
							int a_size )
{
	int l_num = a_num_of_ferns*a_depth*2;

	for( int l_i=0; l_i<l_num; ++l_i )
	{
		CV_MAT_ELEM(*ap_points,int,0,l_i) = a_size*(rand()/(RAND_MAX+0.0))-a_size/2;
		CV_MAT_ELEM(*ap_points,int,1,l_i) = a_size*(rand()/(RAND_MAX+0.0))-a_size/2;
	}
}

/****************************************************************************/

void cv_leopar::fix_poses(	CvMat * ap_poses,
						   	int a_num_of_poses,
							int a_size )
{
	float l_angle_step = 2.0f*cv_pi/a_num_of_poses;
	float l_radius = sqrt(2.0f)*a_size/2.0f;
	
	for( int l_i=0; l_i<a_num_of_poses; ++l_i )
	{
		CV_MAT_ELEM(*ap_poses,float,0,l_i) = cos(l_i*l_angle_step)*l_radius;
		CV_MAT_ELEM(*ap_poses,float,1,l_i) = sin(l_i*l_angle_step)*l_radius;
	}
}

/****************************************************************************/

int * cv_leopar::drop_patch(	IplImage * ap_image,
								CvMat * ap_points,
								int a_row,
								int a_col,
								int a_num_of_ferns,
								int a_depth )
{
	int * lp_leaf_index = new int[a_num_of_ferns];
	int l_depth_mod = (a_depth/3)*3;
	int l_point_index = 0;

	for( int l_i=0; l_i<a_num_of_ferns; ++l_i )
	{
		int l_index=0;
		int l_j=0;

		for(;l_j<l_depth_mod;)
		{
			(l_index<<=1) +=	CV_IMAGE_ELEM(ap_image,float,
								CV_MAT_ELEM(*ap_points,int,1,l_point_index+0)+a_row,
								CV_MAT_ELEM(*ap_points,int,0,l_point_index+0)+a_col)>
								CV_IMAGE_ELEM(ap_image,float,
								CV_MAT_ELEM(*ap_points,int,1,l_point_index+1)+a_row,
								CV_MAT_ELEM(*ap_points,int,0,l_point_index+1)+a_col);
			(l_index<<=1) +=	CV_IMAGE_ELEM(ap_image,float,
								CV_MAT_ELEM(*ap_points,int,1,l_point_index+2)+a_row,
								CV_MAT_ELEM(*ap_points,int,0,l_point_index+2)+a_col)>
								CV_IMAGE_ELEM(ap_image,float,
								CV_MAT_ELEM(*ap_points,int,1,l_point_index+3)+a_row,
								CV_MAT_ELEM(*ap_points,int,0,l_point_index+3)+a_col);
			(l_index<<=1) +=	CV_IMAGE_ELEM(ap_image,float,
								CV_MAT_ELEM(*ap_points,int,1,l_point_index+4)+a_row,
								CV_MAT_ELEM(*ap_points,int,0,l_point_index+4)+a_col)>
								CV_IMAGE_ELEM(ap_image,float,
								CV_MAT_ELEM(*ap_points,int,1,l_point_index+5)+a_row,
								CV_MAT_ELEM(*ap_points,int,0,l_point_index+5)+a_col);

			l_point_index+=6;
			l_j+=3;
		}
		for(;l_j<a_depth;)
		{
			(l_index<<=1) +=	CV_IMAGE_ELEM(ap_image,float,
								CV_MAT_ELEM(*ap_points,int,1,l_point_index+0)+a_row,
								CV_MAT_ELEM(*ap_points,int,0,l_point_index+0)+a_col)>
								CV_IMAGE_ELEM(ap_image,float,
								CV_MAT_ELEM(*ap_points,int,1,l_point_index+1)+a_row,
								CV_MAT_ELEM(*ap_points,int,0,l_point_index+1)+a_col);
				
			l_point_index+=2;
			l_j+=1;
		}
		lp_leaf_index[l_i] = l_index;
	}
	return lp_leaf_index;
}

/****************************************************************************/

std::ofstream & cv_leopar::write( std::ofstream & a_os )
{
	int l_num_of_bins = m_num_of_poses*pow(2.0,m_depth)*m_num_of_ferns;
	
	a_os.write((char*)&m_size,sizeof(m_size));
	a_os.write((char*)&m_depth,sizeof(m_depth));
	a_os.write((char*)&m_magic_factor,sizeof(m_magic_factor));
	a_os.write((char*)&m_num_of_ferns,sizeof(m_num_of_ferns));
	a_os.write((char*)&m_num_of_leaves,sizeof(m_num_of_leaves));
	a_os.write((char*)&m_num_of_poses,sizeof(m_num_of_poses));
	a_os.write((char*)&m_num_of_points,sizeof(m_num_of_points));
	a_os.write((char*)&m_num_of_samples,sizeof(m_num_of_samples));

	for( int l_i=0; l_i<m_num_of_poses; ++l_i )
	{
		a_os.write((char*)&CV_MAT_ELEM(*mp_poses,float,0,l_i),
					sizeof(CV_MAT_ELEM(*mp_poses,float,0,l_i)));
		a_os.write((char*)&CV_MAT_ELEM(*mp_poses,float,1,l_i),
					sizeof(CV_MAT_ELEM(*mp_poses,float,1,l_i)));
	}
	for( int l_i=0; l_i<m_num_of_points; ++l_i )
	{
		a_os.write((char*)&CV_MAT_ELEM(*mp_points,int,0,l_i),
					sizeof(CV_MAT_ELEM(*mp_points,int,0,l_i)));
		a_os.write((char*)&CV_MAT_ELEM(*mp_points,int,1,l_i),
					sizeof(CV_MAT_ELEM(*mp_points,int,1,l_i)));
	}
	for( int l_i=0; l_i<l_num_of_bins; ++l_i )
	{
		a_os.write((char*)&CV_MAT_ELEM(*mp_pose_distributions1,float,0,l_i),
					sizeof(CV_MAT_ELEM(*mp_pose_distributions1,float,0,l_i)));
		a_os.write((char*)&CV_MAT_ELEM(*mp_pose_distributions2,float,0,l_i),
					sizeof(CV_MAT_ELEM(*mp_pose_distributions2,float,0,l_i)));
		a_os.write((char*)&CV_MAT_ELEM(*mp_pose_distributions3,float,0,l_i),
					sizeof(CV_MAT_ELEM(*mp_pose_distributions3,float,0,l_i)));
		a_os.write((char*)&CV_MAT_ELEM(*mp_pose_distributions4,float,0,l_i),
					sizeof(CV_MAT_ELEM(*mp_pose_distributions4,float,0,l_i)));
	}
	return a_os;
}

/****************************************************************************/

std::ifstream & cv_leopar::read( std::ifstream & a_is )
{
	cvReleaseMat( &mp_poses );
	cvReleaseMat( &mp_points );
	cvReleaseMat( &mp_pose_distributions1 );
	cvReleaseMat( &mp_pose_distributions2 );
	cvReleaseMat( &mp_pose_distributions3 );
	cvReleaseMat( &mp_pose_distributions4 );
	
	a_is.read((char*)&m_size,sizeof(m_size));
	a_is.read((char*)&m_depth,sizeof(m_depth));
	a_is.read((char*)&m_magic_factor,sizeof(m_magic_factor));
	a_is.read((char*)&m_num_of_ferns,sizeof(m_num_of_ferns));
	a_is.read((char*)&m_num_of_leaves,sizeof(m_num_of_leaves));
	a_is.read((char*)&m_num_of_poses,sizeof(m_num_of_poses));
	a_is.read((char*)&m_num_of_points,sizeof(m_num_of_points));
	a_is.read((char*)&m_num_of_samples,sizeof(m_num_of_samples));

	int l_num_of_bins = m_num_of_poses*pow(2.0,m_depth)*m_num_of_ferns;

	mp_poses = cvCreateMat(2,m_num_of_poses,CV_32FC1);
	mp_points = cvCreateMat(2,m_num_of_points,CV_32SC1);

	mp_pose_distributions1 = cvCreateMat(1,l_num_of_bins,CV_32FC1);
	mp_pose_distributions2 = cvCreateMat(1,l_num_of_bins,CV_32FC1);
	mp_pose_distributions3 = cvCreateMat(1,l_num_of_bins,CV_32FC1);
	mp_pose_distributions4 = cvCreateMat(1,l_num_of_bins,CV_32FC1);

	for( int l_i=0; l_i<m_num_of_poses; ++l_i )
	{
		a_is.read( (char*)&CV_MAT_ELEM(*mp_poses,float,0,l_i),
					sizeof(CV_MAT_ELEM(*mp_poses,float,0,l_i)));
		a_is.read( (char*)&CV_MAT_ELEM(*mp_poses,float,1,l_i),
					sizeof(CV_MAT_ELEM(*mp_poses,float,1,l_i)));
	}
	for( int l_i=0; l_i<m_num_of_points; ++l_i )
	{
		a_is.read( (char*)&CV_MAT_ELEM(*mp_points,int,0,l_i),
					sizeof(CV_MAT_ELEM(*mp_points,int,0,l_i)));
		a_is.read( (char*)&CV_MAT_ELEM(*mp_points,int,1,l_i),
					sizeof(CV_MAT_ELEM(*mp_points,int,1,l_i)));
	}
	for( int l_i=0; l_i<l_num_of_bins; ++l_i )
	{
		a_is.read( (char*)&CV_MAT_ELEM(*mp_pose_distributions1,float,0,l_i),
					sizeof(CV_MAT_ELEM(*mp_pose_distributions1,float,0,l_i)));
		a_is.read( (char*)&CV_MAT_ELEM(*mp_pose_distributions2,float,0,l_i),
					sizeof(CV_MAT_ELEM(*mp_pose_distributions2,float,0,l_i)));
		a_is.read( (char*)&CV_MAT_ELEM(*mp_pose_distributions3,float,0,l_i),
					sizeof(CV_MAT_ELEM(*mp_pose_distributions3,float,0,l_i)));
		a_is.read( (char*)&CV_MAT_ELEM(*mp_pose_distributions4,float,0,l_i),
					sizeof(CV_MAT_ELEM(*mp_pose_distributions4,float,0,l_i)));
	}
	return a_is;
}

/****************************************************************************/

bool cv_leopar::save( std::string a_name )
{
	std::ofstream l_file(a_name.c_str(),std::ofstream::out|std::ofstream::binary);
	
	if( l_file.fail() == true )
	{
		printf("cv_leopar: could not open for writing!");
		return false;
	}
	this->write(l_file);	

	l_file.close();

	return true;
}

/****************************************************************************/

bool cv_leopar::load( std::string a_name )
{
	std::ifstream l_file(a_name.c_str(),std::ifstream::in|std::ifstream::binary);
	
	if( l_file.fail() == true ) 
	{
		printf("cv_leopar: could not open for reading!");
		return false;
	}
	this->read(l_file);	

	l_file.close();

	return true;
}

/**************************** END OF FILE ***********************************/
