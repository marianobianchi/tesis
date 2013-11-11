//////////////////////////////////////////////////////////////////////////////
//																			//
// cv_pcabase: cv_pcabase.cc												//
//																			//
// Authors: Stefan Hinterstoisser 2009										//
// Lehrstuhl fuer Informatik XVI											//
// Technische Universitaet Muenchen											//
// Version: 1.0																//
//																			//
//////////////////////////////////////////////////////////////////////////////

/******************************* includes ***********************************/

#include "cv_pcabase.h"
#include "cv_homography.h"

#include <vector>
#include <sstream>
#include <omp.h>

/******************************** defines ***********************************/

/******************************* namespaces *********************************/

using namespace cv;

/****************************** constants ***********************************/

/****************************** constructors ********************************/

cv_pcabase::cv_pcabase( void )
{
	m_pat_size			= 0.00;
	m_sup_size			= 0.00;
	m_num_of_samples	= 0.00;

	mp_pos				= NULL;
	mp_pca_mean			= NULL;
	mp_pca_base			= NULL;
	mp_gep_base			= NULL;

	m_gep_poses.clear();
	m_pca_samples.clear();

	mp_k = cvCreateMat(3,3,CV_32FC1);
}

/******************************* destructor *********************************/

cv_pcabase::~cv_pcabase()
{
	this->clear();
}

/****************************************************************************/

void cv_pcabase::clear_pca( void )
{
	if( mp_pca_base != NULL )
	{
		cvReleaseMat(&mp_pca_base);
		mp_pca_base = NULL;
	}
	if( mp_pca_mean != NULL )
	{
		cvReleaseMat(&mp_pca_mean);
		mp_pca_mean = NULL;
	}
	if( mp_gep_base != NULL )
	{
		cvReleaseMat(&mp_gep_base);
		mp_gep_base = NULL;
	}
}

/****************************************************************************/

void cv_pcabase::clear_pose( void )
{
	for( int l_i=0; l_i<m_gep_poses.size(); ++l_i )
	{
		if( m_gep_poses[l_i] != NULL )
		{
			cvReleaseMat(&m_gep_poses[l_i]);
			m_gep_poses[l_i] = NULL;
		}
	}
	m_gep_poses.clear();
}

/****************************************************************************/


void cv_pcabase::clear_samples( void )
{
	for( int l_i=0; l_i<m_pca_samples.size(); ++l_i )
	{
		cvReleaseMat(&m_pca_samples[l_i]);
	}
	m_pca_samples.clear();
}

/****************************************************************************/

void cv_pcabase::clear_positions( void )
{
	cvReleaseMat(&mp_pos);
}

/****************************************************************************/

void cv_pcabase::clear( void )
{
	this->clear_pca();
	this->clear_pose();
	this->clear_samples();
	this->clear_positions();

	cvReleaseMat(&mp_k);
}

/****************************************************************************/

bool cv_pcabase::set_parameters(	CvMat * ap_k,
									int a_pat_size,
									int a_sup_size,
									int a_nx, int a_ny,
									int a_num_of_samples )
{
	if( a_pat_size  <= 0 )
	{
		printf("cv_pcabase: a_pat_size are not appropriate!");
		return false;
	}
	if( a_pat_size  % 2 == 0 )
	{
		printf("cv_pcabase: a_pat_size have to be odd!");
		return false;
	}
	if( a_sup_size  <= 0 )
	{
		printf("cv_pcabase: a_sup_size are not appropriate!");
		return false;
	}
	if( a_sup_size  < a_pat_size  )
	{
		printf("cv_pcabase: a_sup_size have to be bigger then a_pat_size!");
		return false;
	}
	if( a_num_of_samples <= 0 )
	{
		printf("cv_pcabase: a_num_of_samples should be over 0!");
		return false;
	}
	if( a_nx < 0 || a_ny < 0 )
	{
		printf("cv_pcabase: a_nx or a_ny are smaller than 0!");
		return false;
	}
	this->clear();

	m_nx = a_nx;
	m_ny = a_ny;
	m_pat_size = a_pat_size;
	m_sup_size = a_sup_size;
	m_num_of_samples = a_num_of_samples;

	mp_pos = this->create_positions(m_nx,m_ny,m_pat_size);

	mp_pca_mean = cvCreateMat(1,m_sup_size*m_sup_size,CV_32FC1);
	cvSet(mp_pca_mean,cvRealScalar(0));

	mp_k = cvCreateMat(3,3,CV_32FC1);
	cvCopy(ap_k,mp_k);

	return true;
}

/****************************************************************************/

bool cv_pcabase::add(	IplImage * ap_image,
						IplImage * ap_mask,
						int a_row,
						int a_col )
{
	CvMat * lp_patch = cvCreateMat(1,m_sup_size*m_sup_size,CV_32FC1);

	int l_halfw = m_sup_size/2;
	int l_halfh = m_sup_size/2;

	int l_index=0;

	for( int l_r=0; l_r<m_sup_size; ++l_r )
	{
		for( int l_c=0; l_c<m_sup_size; ++l_c )
		{
			int l_row = l_r+a_row-l_halfh;
			int l_col = l_c+a_col-l_halfw;

			if( l_row < 0 ||
				l_col < 0 ||
				l_row > ap_image->height-1 ||
				l_col > ap_image->width-1 )
			{
				cvReleaseMat(&lp_patch);
				return false;
			}
			if(	CV_IMAGE_ELEM(ap_mask ,unsigned char,l_row,l_col) == 0 )
			{
				cvReleaseMat(&lp_patch);
				return false;
			}
			CV_MAT_ELEM(*lp_patch,float,0,l_index) =
			CV_IMAGE_ELEM(ap_image,float,l_row,l_col);
			++l_index;
		}
	}
	cvAdd(mp_pca_mean,lp_patch,mp_pca_mean);
	m_pca_samples.push_back(lp_patch);

	return true;
}

/****************************************************************************/

CvMat * cv_pcabase::compute_pca_base(	std::vector<CvMat*> & a_pca_samples,
										CvMat * ap_pca_mean,
										int a_num_of_pcas )
{
	if( a_num_of_pcas > a_pca_samples.size()-1 )
	{
		printf("cv_pcabase.compute_pcas: a_num_of_pcas is too big!");
		return NULL;
	}
	int l_num = a_pca_samples.size();
	int l_dim = ap_pca_mean->cols;

	cvScale(ap_pca_mean,ap_pca_mean,1.0/l_num);

	CvMat * lp_data = cvCreateMat(l_dim,l_num,CV_32FC1);
	for( int l_c=0; l_c<l_num; ++l_c )
	{
        std::cout << "compute_pca_base: " << l_c << "/" << l_num << std::endl;
		for( int l_r=0; l_r<l_dim; ++l_r )
		{
			CV_MAT_ELEM(*lp_data,float,l_r,l_c) =
			CV_MAT_ELEM(*a_pca_samples[l_c],float,0,l_r)-
			CV_MAT_ELEM(*ap_pca_mean,float,0,l_r);
		}
	}
	CvMat * lp_d = cvCreateMat(l_num,1,CV_32FC1);
	CvMat * lp_v = cvCreateMat(l_num,l_num,CV_32FC1);
	CvMat * lp_p = cvCreateMat(l_dim,l_num,CV_32FC1);
	CvMat * lp_t = cvCreateMat(l_num,l_dim,CV_32FC1);

	CvMat * lp_pca_base = cvCreateMat(a_num_of_pcas,l_dim,CV_32FC1);

	std::cout << "compute_pca_base (svd): " << std::endl;

	cvSVD(lp_data,lp_d,NULL,lp_v);

	std::cout << "compute_pca_base (matmul): " << std::endl;
	cvMatMul(lp_data,lp_v,lp_p);
	std::cout << "compute_pca_base (transpose): " << std::endl;
	cvTranspose(lp_p,lp_t);

	for( int l_r=0; l_r<lp_pca_base->rows; ++l_r )
	{
        std::cout << "compute_pca_base (seg for): " << l_r << "/" << lp_pca_base->rows << std::endl;
		double l_norm=0;

		for( int l_c=0; l_c<lp_t->cols; ++l_c )
		{
			l_norm += SQR(CV_MAT_ELEM(*lp_t,float,l_r,l_c));
		}
		l_norm = sqrt(l_norm);

		for( int l_c=0; l_c<lp_pca_base->cols; ++l_c )
		{
			CV_MAT_ELEM(*lp_pca_base,float,l_r,l_c) = CV_MAT_ELEM(*lp_t,float,l_r,l_c)/l_norm;
		}
	}
	cvReleaseMat(&lp_data);
	cvReleaseMat(&lp_d);
	cvReleaseMat(&lp_v);
	cvReleaseMat(&lp_p);
	cvReleaseMat(&lp_t);

	return lp_pca_base;
}

/****************************************************************************/

CvMat * cv_pcabase::get_back_gep_mean(	CvMat * ap_lin_comb,
										int a_view_index )
{
	CvMat * lp_gep_mean = cvCreateMat(1,mp_pos->cols,CV_32FC1);

	int l_index = mp_pos->cols*a_view_index;

	for( int l_i=0; l_i<mp_pos->cols; ++l_i )
	{
		CV_MAT_ELEM(*lp_gep_mean,float,0,l_i) =
		CV_MAT_ELEM(*ap_lin_comb,float,0,l_index+l_i);
	}
	return lp_gep_mean;
}

/****************************************************************************/

void cv_pcabase::push_back_gep_mean(	CvMat * ap_all_base,
										CvMat * ap_gep_mean,
										int a_view_index,
										int a_pca_index )
{
	int l_offset = a_view_index*ap_gep_mean->cols;

	for( int l_i=0; l_i<ap_gep_mean->cols; ++l_i )
	{
		CV_MAT_ELEM(*ap_all_base,float,a_pca_index,l_offset+l_i) +=
		CV_MAT_ELEM(*ap_gep_mean,float,0,l_i);
	}
}

/****************************************************************************/

bool cv_pcabase::learn_base( int a_num_of_pcas )
{

	mp_pca_base =  this->compute_pca_base(	m_pca_samples,
											mp_pca_mean,
											a_num_of_pcas );
	if( mp_pca_base == NULL )
	{
		printf("cv_pcabase.learn_base: mp_pca_base is NULL!");
		return false;
	}
	this->clear_samples();

	//std::cerr << "learn_base: " << mp_pca_base->cols << "," << mp_pca_base->rows << std::endl;
	//std::cerr << "learn_base: " << mp_pos->cols  << "," << mp_pos->rows  << std::endl;

	cvReleaseMat(&mp_gep_base);

	mp_gep_base = this->learn_gepard_base(	mp_pca_base,mp_pca_mean,
											mp_pos,mp_k,m_gep_poses,
											m_num_of_samples,
											m_sup_size,m_pat_size );
	return true;
}

/****************************************************************************/

CvMat * cv_pcabase::learn_gepard_base(	CvMat * ap_pca_base,
										CvMat * ap_pca_mean,
										CvMat * ap_pos,
										CvMat * ap_k,
										std::vector<CvMat*> & a_gep_poses,
										int a_num_of_samples,
										int a_sup_size,
										int a_pat_size )
{
/*
	const int l_num_of_rot = 36;
	const int l_num_of_sca = 1;

	float l_sca[l_num_of_sca]		= {1.0};
	float l_sca_step[l_num_of_sca]	= {0.15};
	float l_rot[l_num_of_rot]		= {0,10,20,30,40,50,60,70,80,90,100,110,120,130,
									   140,150,160,170,180,190,200,210,220,230,240,
									   250,260,270,280,290,300,310,320,330,340,350};
	float l_rot_step[l_num_of_rot]  = {5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
			   						   5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5};
*/

	/*
	const int l_num_of_rot = 18;
	const int l_num_of_sca = 1;

	float l_sca[l_num_of_sca]		= {1.0};
	float l_sca_step[l_num_of_sca]	= {0.20};
	float l_rot[l_num_of_rot]		= {0,20,40,60,80,100,120,140,160,180,200,220,240,260,280,300,320,340};
	float l_rot_step[l_num_of_rot]  = {10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10};
	*/
	const int l_num_of_rot = 12;
	const int l_num_of_sca = 1;

	float l_sca[l_num_of_sca]		= {1.0};
	float l_sca_step[l_num_of_sca]	= {0.20};
	float l_rot[l_num_of_rot]		= {0,30,60,90,120,150,180,210,240,270,300,330};
	float l_rot_step[l_num_of_rot]  = {15,15,15,15,15,15,15,15,15,15,15,15};

	std::cout << "learn_gepard_base " << std::endl;


	std::vector<std::pair<CvPoint3D32f,float> > l_views = this->get_views();

	CvMat * lp_gep_base = cvCreateMat(ap_pca_base->rows+1,l_views.size()*l_num_of_rot*l_num_of_sca*ap_pos->cols,CV_32FC1);

	cvSet(lp_gep_base,cvRealScalar(0));

	CvMat * lp_k	= cvCreateMat(3,3,CV_32FC1);
	CvMat * lp_rec	= cvCreateMat(3,4,CV_32FC1);

	CV_MAT_ELEM(*lp_rec,float,0,0) = -a_pat_size/2+a_sup_size/2;
	CV_MAT_ELEM(*lp_rec,float,1,0) = -a_pat_size/2+a_sup_size/2;
	CV_MAT_ELEM(*lp_rec,float,0,1) = +a_pat_size/2+a_sup_size/2;
	CV_MAT_ELEM(*lp_rec,float,1,1) = -a_pat_size/2+a_sup_size/2;
	CV_MAT_ELEM(*lp_rec,float,0,2) = +a_pat_size/2+a_sup_size/2;
	CV_MAT_ELEM(*lp_rec,float,1,2) = +a_pat_size/2+a_sup_size/2;
	CV_MAT_ELEM(*lp_rec,float,0,3) = -a_pat_size/2+a_sup_size/2;
	CV_MAT_ELEM(*lp_rec,float,1,3) = +a_pat_size/2+a_sup_size/2;
	CV_MAT_ELEM(*lp_rec,float,2,0) = 1;
	CV_MAT_ELEM(*lp_rec,float,2,1) = 1;
	CV_MAT_ELEM(*lp_rec,float,2,2) = 1;
	CV_MAT_ELEM(*lp_rec,float,2,3) = 1;

	cvCopy(ap_k,lp_k);

	CV_MAT_ELEM(*lp_k,float,0,2) = a_sup_size/2;
	CV_MAT_ELEM(*lp_k,float,1,2) = a_sup_size/2;

	int l_pose=0;
	float l_fac;

	if( a_num_of_samples == 1 )
	{
		l_fac=0;
	}
	else
	{
		l_fac=1;
	}
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
				CvMat * lp_ground_hom = this->get_pose( lp_k,lp_view,
														l_rot[l_ri],
														l_sca[l_si] );

				cvMatMul(lp_ground_hom,lp_rec,lp_ground_rec);
				cv_homogenize(lp_ground_rec);

				CV_MAT_ELEM(*lp_ground_rec,float,0,0) -= a_sup_size/2;
				CV_MAT_ELEM(*lp_ground_rec,float,1,0) -= a_sup_size/2;
				CV_MAT_ELEM(*lp_ground_rec,float,0,1) -= a_sup_size/2;
				CV_MAT_ELEM(*lp_ground_rec,float,1,1) -= a_sup_size/2;
				CV_MAT_ELEM(*lp_ground_rec,float,0,2) -= a_sup_size/2;
				CV_MAT_ELEM(*lp_ground_rec,float,1,2) -= a_sup_size/2;
				CV_MAT_ELEM(*lp_ground_rec,float,0,3) -= a_sup_size/2;
				CV_MAT_ELEM(*lp_ground_rec,float,1,3) -= a_sup_size/2;

				a_gep_poses.push_back(lp_ground_rec);

				cvReleaseMat(&lp_ground_hom);
				cvReleaseMat(&lp_view);

				std::vector<CvMat*> l_homographies;
				std::vector<float>	l_xtranslation;
				std::vector<float>	l_ytranslation;

				for( int l_i=0; l_i<a_num_of_samples; ++l_i )
				{
					float l_orbit_angle = l_views[l_vi].second;
					float l_orbit_orien = l_fac*rand()/(RAND_MAX+0.0)*360.0;
					float l_scale_off	= l_fac*rand()/(RAND_MAX+0.0)*l_sca_step[l_si]*2*1.2-l_sca_step[l_si]*1.2;
					float l_rotat_off	= l_fac*rand()/(RAND_MAX+0.0)*l_rot_step[l_ri]*2*1.2-l_rot_step[l_ri]*1.2;
					float l_trans_row	= l_fac*rand()/(RAND_MAX+0.0)*10.0-5.0;
					float l_trans_col	= l_fac*rand()/(RAND_MAX+0.0)*10.0-5.0;

					CvMat * lp_axis = this->get_axis(l_views[l_vi].first,l_orbit_angle,l_orbit_orien );
					CvMat * lp_hom  = this->get_pose(lp_k,lp_axis,l_rot[l_ri]+l_rotat_off,l_sca[l_si]+l_scale_off );

					CvMat * lp_inv = cvCreateMat(3,3,CV_32FC1);
					cvInvert(lp_hom,lp_inv,CV_SVD);

					l_homographies.push_back(lp_inv);
					l_xtranslation.push_back(l_trans_col);
					l_ytranslation.push_back(l_trans_row);

					cvReleaseMat(&lp_axis);
					cvReleaseMat(&lp_hom);
				}
				for( int l_p=0; l_p<ap_pca_base->rows+1; ++l_p )
				{
					IplImage * lp_patch = NULL;

					if( l_p == ap_pca_base->rows )
					{
						lp_patch = this->get_patch(ap_pca_mean,a_sup_size,0);
					}
					else
					{
						lp_patch = this->get_patch(ap_pca_base,a_sup_size,l_p);
					}
					for( int l_t=0; l_t<l_homographies.size(); ++l_t )
					{
                        std::cout << "learn_gepard_base " << l_t << std::endl;
						CvMat * lp_gep_pos = cvCreateMat(ap_pos->rows,ap_pos->cols,CV_32FC1);
						cvCopy(ap_pos,lp_gep_pos);

						for( int l_kk=0; l_kk<lp_gep_pos->cols; ++l_kk )
						{
							CV_MAT_ELEM(*lp_gep_pos,float,0,l_kk) += a_sup_size/2+l_xtranslation[l_t];
							CV_MAT_ELEM(*lp_gep_pos,float,1,l_kk) += a_sup_size/2+l_ytranslation[l_t];
						}
						CvMat * lp_warp_pos = cvCreateMat(ap_pos->rows,ap_pos->cols,CV_32FC1);

						cvMatMul(l_homographies[l_t],lp_gep_pos,lp_warp_pos);
						cv_homogenize(lp_warp_pos);

						CvMat * lp_gep_int = this->get_train_intensity(lp_patch,lp_warp_pos,0,0);

						cvScale(lp_gep_int,lp_gep_int,1.0/a_num_of_samples);

						this->push_back_gep_mean(lp_gep_base,lp_gep_int,l_pose,l_p);

						cvReleaseMat(&lp_warp_pos);
						cvReleaseMat(&lp_gep_int);
						cvReleaseMat(&lp_gep_pos);
					}
					cvReleaseImage(&lp_patch);
				}
				for( int l_t=0; l_t<l_homographies.size(); ++l_t )
				{
					cvReleaseMat(&l_homographies[l_t]);
				}
				l_homographies.clear();
				l_xtranslation.clear();
				l_ytranslation.clear();

				std::cout << "gepard: " << l_pose << "/" << l_num_of_sca*l_num_of_rot*l_views.size() << char(13) << std::flush;

				++l_pose;
			}
		}
	}
	cvReleaseMat(&lp_rec);
	cvReleaseMat(&lp_k);

	return lp_gep_base;
}

/****************************************************************************/

cv_gepard * cv_pcabase::get_gepard( std::vector<CvMat*> & a_gep_poses,
									CvMat * ap_gep_base,
									CvMat * ap_pos,
									CvMat * ap_alphas,
									CvMat * ap_k,
									int a_num_of_pcas,
									int a_num_of_samples,
								    int a_nx, int a_ny,
									int a_sup_size,
									int a_pat_size )
{
	cv_gepard * lp_gepard = new cv_gepard;

	lp_gepard->m_nx					= a_nx;
	lp_gepard->m_ny					= a_ny;
	lp_gepard->m_size				= a_pat_size;
	lp_gepard->m_num_of_samples		= a_num_of_samples;

	cvReleaseMat(&lp_gepard->mp_k);
	cvReleaseMat(&lp_gepard->mp_pos);

	lp_gepard->mp_k		= cvCreateMat(3,3,CV_32FC1);
	lp_gepard->mp_pos	= cvCreateMat(3,ap_pos->cols,CV_32FC1);

	cvCopy(ap_k,lp_gepard->mp_k);
	cvCopy(ap_pos,lp_gepard->mp_pos);

	lp_gepard->m_means.resize(a_gep_poses.size());
	lp_gepard->m_poses.resize(a_gep_poses.size());

	CvMat * lp_lin_comb = this->get_lin_combination(ap_gep_base,ap_alphas,
													a_num_of_pcas,true);
	CvMat * lp_pose=NULL;
	CvMat * lp_mean=NULL;

	#pragma omp parallel for private(lp_mean,lp_pose) \
	shared(a_gep_poses,lp_gepard,lp_lin_comb)
	for( int l_i=0; l_i<a_gep_poses.size(); ++l_i )
	{
		lp_pose = cvCreateMat(3,4,CV_32FC1);
		cvCopy(a_gep_poses[l_i],lp_pose);

		lp_mean = this->get_back_gep_mean(lp_lin_comb,l_i);
		cv_normalize_mean_std((float*)lp_mean->data.ptr,lp_mean->cols);

		lp_gepard->m_means[l_i] = lp_mean;
		lp_gepard->m_poses[l_i] = lp_pose;
	}
	cvReleaseMat(&lp_lin_comb);

	return lp_gepard;
}

/****************************************************************************/

cv_gepard	* cv_pcabase::get_tracker(	IplImage * ap_image,
										int a_num_of_pcas,
										int a_row,
										int a_col )
{
	if( ap_image == NULL )
	{
		printf("cv_pcabase: get_tracker - ap_image is NULL!");
		return NULL;
	}
	if( ap_image->imageData == NULL )
	{
		printf("cv_pcabase: imageData is NULL!");
		return NULL;
	}
	if( a_col-m_sup_size/2 < 0 ||
		a_row-m_sup_size/2 < 0 ||
		a_col+m_sup_size/2 > ap_image->width-1 ||
		a_row+m_sup_size/2 > ap_image->height-1 )
	{
		printf("cv_pcabase: get_tracker - a_col or a_row is out of space!");
		return NULL;
	}
	if( a_num_of_pcas >= mp_pca_base->rows )
	{
		printf("cv_pcabase: get_tracker - a_num_of_pcas is bigger the available number!");
		return NULL;
	}
	CvMat * lp_patch_src = this->get_patch(ap_image,m_sup_size,a_row,a_col);
	cvSub(lp_patch_src,mp_pca_mean,lp_patch_src);
	CvMat * lp_alphas = this->get_pca_projection(mp_pca_base,lp_patch_src,a_num_of_pcas);
	cvReleaseMat(&lp_patch_src);



/*

	IplImage * lp_image1 = get_patch(mp_pca_base,m_sup_size,1);
	IplImage * lp_image2 = get_patch(mp_pca_base,m_sup_size,10);
	IplImage * lp_image3 = get_patch(mp_pca_base,m_sup_size,20);
	IplImage * lp_image4 = get_patch(mp_pca_base,m_sup_size,50);

	cvScale(lp_image1,lp_image1,10000,100);
	cvScale(lp_image2,lp_image2,10000,100);
	cvScale(lp_image3,lp_image3,10000,100);
	cvScale(lp_image4,lp_image4,10000,100);


	cv_show_image(lp_image1);
	cv_show_image(lp_image2);
	cv_show_image(lp_image3);
	cv_show_image(lp_image4);
*/




	cv_gepard * lp_gepard = get_gepard( m_gep_poses,
										mp_gep_base,
										mp_pos,
										lp_alphas,
										mp_k,
										a_num_of_pcas,
										m_num_of_samples,
										m_nx,m_ny,
										m_sup_size,
										m_pat_size );
	cvReleaseMat(&lp_alphas);

	return lp_gepard;
}

/****************************************************************************/

CvMat * cv_pcabase::get_pca_projection( CvMat * ap_pcas,
										CvMat * ap_patch,
										int a_num_of_pcas )
{
	CvMat * lp_alphas = cvCreateMat(1,ap_pcas->rows,CV_32FC1);

	#pragma omp parallel for shared(lp_alphas,ap_pcas,ap_patch)
	for( int l_i=0; l_i<a_num_of_pcas; ++l_i )
	{
		CV_MAT_ELEM(*lp_alphas,float,0,l_i) = cv_dot_product(&CV_MAT_ELEM(*ap_pcas,float,l_i,0),
															 &CV_MAT_ELEM(*ap_patch,float,0,0),
															 ap_pcas->cols);
	}
	return lp_alphas;
}

/****************************************************************************/

CvMat * cv_pcabase::get_lin_combination( CvMat * ap_pcas,
										 CvMat * ap_alphas,
										 int a_num_of_pcas,
										 bool a_pca_plus_mean )
{
	CvMat * lp_result = cvCreateMat(1,ap_pcas->cols,CV_32FC1);

	cvSet(lp_result,cvRealScalar(0));

	if( a_pca_plus_mean == true )
	{
		#pragma omp parallel for shared(lp_result,ap_pcas,ap_alphas)
		for( int l_i=0; l_i<a_num_of_pcas; ++l_i )
		{
			cv_lin_combination(	&CV_MAT_ELEM(*ap_pcas,float,l_i,0),
								CV_MAT_ELEM(*ap_alphas,float,0,l_i),
								&CV_MAT_ELEM(*lp_result,float,0,0),1.0,
								&CV_MAT_ELEM(*lp_result,float,0,0),
								ap_pcas->cols);
		}
		cv_lin_combination(	&CV_MAT_ELEM(*ap_pcas,float,ap_pcas->rows-1,0),1.0,
							&CV_MAT_ELEM(*lp_result,float,0,0),1.0,
							&CV_MAT_ELEM(*lp_result,float,0,0),
							ap_pcas->cols);
	}
	else
	{
		#pragma omp parallel for shared(lp_result,ap_pcas,ap_alphas)
		for( int l_i=0; l_i<a_num_of_pcas; ++l_i )
		{
			cv_lin_combination(	&CV_MAT_ELEM(*ap_pcas,float,l_i,0),
								CV_MAT_ELEM(*ap_alphas,float,0,l_i),
								&CV_MAT_ELEM(*lp_result,float,0,0),1.0,
								&CV_MAT_ELEM(*lp_result,float,0,0),
								ap_pcas->cols);
		}
	}
	return lp_result;
}

/****************************************************************************/

CvMat * cv_pcabase::get_patch(	IplImage * ap_image,
								int a_sup_size,
								int a_row,
								int a_col )
{
	CvMat * lp_patch = cvCreateMat(1,a_sup_size*a_sup_size,CV_32FC1);

	int l_row = a_row-a_sup_size/2;
	int l_col = a_col-a_sup_size/2;
	int l_r   = 0;
	int l_c   = 0;
	int l_rr  = 0;
	int l_cc  = 0;

	#pragma omp parallel for private(l_r,l_c,l_rr,l_cc) \
	shared(l_row,l_col,lp_patch,a_sup_size,ap_image)

	for( l_r=0; l_r<a_sup_size; ++l_r )
	{
		l_rr = l_r+l_row;

		for( l_c=0; l_c<a_sup_size; ++l_c )
		{
			l_cc = l_c+l_col;

			CV_MAT_ELEM(*lp_patch,float,0,l_r*a_sup_size+l_c) =
			CV_IMAGE_ELEM(ap_image,float,l_rr,l_cc);
		}
	}
	return lp_patch;
}

/****************************************************************************/

IplImage * cv_pcabase::get_patch(	CvMat * ap_mat,
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

CvMat *	cv_pcabase::create_positions(	int a_nx,
										int a_ny,
										int a_pat_size )
{
	float l_col = -a_pat_size/2;
	float l_row = -a_pat_size/2;

	const float l_stepx = a_pat_size/(a_nx-1.0);
	const float l_stepy = a_pat_size/(a_ny-1.0);

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

IplImage * cv_pcabase::warp(	IplImage * ap_image,
								CvMat * ap_hom,
								int a_pat_size,
								int a_row,
								int a_col )
{
	IplImage * lp_image = cvCreateImage(cvSize(a_pat_size,a_pat_size),IPL_DEPTH_32F,1);

	CvMat * lp_inv = cvCreateMat(3,3,CV_32FC1);
	cvInvert(ap_hom,lp_inv,CV_SVD);

	for( int l_r=-a_pat_size/2; l_r<a_pat_size/2; ++l_r )
	{
		for( int l_c=-a_pat_size/2; l_c<a_pat_size/2; ++l_c )
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
			CV_IMAGE_ELEM(lp_image,float,l_r+a_pat_size/2,l_c+a_pat_size/2) = l_val;
		}
	}
	cvReleaseMat(&lp_inv);

	return lp_image;
}

/****************************************************************************/

CvMat * cv_pcabase::get_pose(	CvMat * ap_k,
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

CvMat * cv_pcabase::get_axis(	CvPoint3D32f & a_axis,
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

std::vector<std::pair<CvPoint3D32f,float> > cv_pcabase::get_views( void )
{
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

CvMat *	cv_pcabase::get_train_intensity(	IplImage * ap_image,
											CvMat * ap_pos,
											int a_row,
											int a_col )
{
	CvMat * lp_int = cvCreateMat(1,ap_pos->cols,CV_32FC1);

	for( int l_i=0; l_i<ap_pos->cols; ++l_i )
	{
		if( CV_MAT_ELEM(*ap_pos,float,1,l_i)+a_row >= 1 &&
			CV_MAT_ELEM(*ap_pos,float,1,l_i)+a_row < ap_image->height-1 &&
			CV_MAT_ELEM(*ap_pos,float,0,l_i)+a_col >= 1 &&
			CV_MAT_ELEM(*ap_pos,float,0,l_i)+a_col < ap_image->width-1  )
		{
			CV_MAT_ELEM(*lp_int,float,0,l_i) = this->get_linear(ap_image,
											   CV_MAT_ELEM(*ap_pos,float,1,l_i)+a_row,
											   CV_MAT_ELEM(*ap_pos,float,0,l_i)+a_col);
		}
		else
		{
			CV_MAT_ELEM(*lp_int,float,0,l_i) = 0;
		}
	}
	return lp_int;
}

/****************************************************************************/

float cv_pcabase::get_linear(	IplImage * ap_image,
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

std::ofstream & cv_pcabase::write( std::ofstream & a_os )
{
	int l_num_of_poses = m_gep_poses.size();

	a_os.write((char*)&m_nx,sizeof(m_nx));
	a_os.write((char*)&m_ny,sizeof(m_ny));
	a_os.write((char*)&m_pat_size,sizeof(m_pat_size));
	a_os.write((char*)&m_sup_size,sizeof(m_sup_size));
	a_os.write((char*)&l_num_of_poses,sizeof(l_num_of_poses));
	a_os.write((char*)&m_num_of_samples,sizeof(m_num_of_samples));

	cv_write(a_os,mp_pca_base);
	cv_write(a_os,mp_pca_mean);
	cv_write(a_os,mp_gep_base);
	cv_write(a_os,mp_pos);
	cv_write(a_os,mp_k);

	for( int l_i=0; l_i<l_num_of_poses; ++l_i )
	{
		cv_write(a_os,m_gep_poses[l_i]);
	}
	return a_os;
}

/****************************************************************************/

std::ifstream & cv_pcabase::read( std::ifstream & a_is )
{
	this->clear();

	int l_num_of_poses=0;

	a_is.read((char*)&m_nx,sizeof(m_nx));
	a_is.read((char*)&m_ny,sizeof(m_ny));
	a_is.read((char*)&m_pat_size,sizeof(m_pat_size));
	a_is.read((char*)&m_sup_size,sizeof(m_sup_size));
	a_is.read((char*)&l_num_of_poses,sizeof(l_num_of_poses));
	a_is.read((char*)&m_num_of_samples,sizeof(m_num_of_samples));

	mp_pca_base = cv_read(a_is);
	mp_pca_mean = cv_read(a_is);
	mp_gep_base = cv_read(a_is);
	mp_pos = cv_read(a_is);
	mp_k = cv_read(a_is);

	for( int l_i=0; l_i<l_num_of_poses; ++l_i )
	{
		CvMat * lp_pose = cv_read(a_is);
		m_gep_poses.push_back(lp_pose);
	}
	return a_is;
}

/****************************************************************************/

bool cv_pcabase::save( std::string a_name )
{
	std::ofstream l_file(	a_name.c_str(),
							std::ofstream::out |
							std::ofstream::binary );

	if( l_file.fail() == true )
	{
		//printf("cv_pca_base: could not open file for writing!\n");
		return false;
	}
	this->write( l_file );

	l_file.close();

	return true;
}

/****************************************************************************/

bool cv_pcabase::load( std::string a_name )
{
	std::ifstream l_file(	a_name.c_str(),
							std::ifstream::in |
							std::ifstream::binary );

	if( l_file.fail() == true )
	{
		//printf("cv_pca_base: could not open file for reading!\n");
		return false;
	}
	this->read( l_file );

	l_file.close();

	return true;
}

/******************************** END OF FILE *******************************/
