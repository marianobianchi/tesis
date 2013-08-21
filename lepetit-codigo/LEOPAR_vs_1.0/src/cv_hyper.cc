//////////////////////////////////////////////////////////////////////////////
//
// cv_hyper: cv_hyper.cc
//
// Authors: Stefan Hinterstoisser 2009
// Lehrstuhl fuer Informatik XVI
// Technische Universitaet Muenchen
// Version: 1.0
//
//////////////////////////////////////////////////////////////////////////////

/******************************* includes ***********************************/

#include "cxtypes.h"
#include "cv_hyper.h"
#include "cv_homography.h"

#include <fstream>

/******************************** defines ***********************************/

/******************************* namespaces *********************************/

using namespace cv;

/****************************** constructors ********************************/

cv_hyper::cv_hyper(void)
{
	m_nx = 0;
	m_ny = 0;
	m_ncc = 0;
	m_border = 0;
	m_max_motion = 0;
	m_num_of_levels = 0;
	m_num_of_samples = 0;

	lp_pix = NULL;
	mp_pos = NULL;
	mp_rec = NULL;
	mp_int = NULL;
	mp_as  = NULL;

	this->clear();
}

/****************************** destructors ********************************/

cv_hyper::~cv_hyper()
{
	this->clear();
}

/****************************************************************************/

void cv_hyper::clear( void )
{
	if( lp_pix != NULL )
		cvReleaseMat(&lp_pix);
	if( mp_pos != NULL )
		cvReleaseMat(&mp_pos);
	if( mp_rec != NULL )
		cvReleaseMat(&mp_rec);
	if( mp_int != NULL )
		cvReleaseMat(&mp_int);

	lp_pix = NULL;
	mp_pos = NULL;
	mp_rec = NULL;
	mp_int = NULL;

	if( mp_as != NULL )
	{
		for( int l_i=0; l_i<m_num_of_levels; ++l_i )
		{
			if( mp_as[l_i] != NULL )
			{
				cvReleaseMat(&mp_as[l_i]);
				mp_as[l_i] = NULL;
			}
		}
		delete[] mp_as;
		mp_as = NULL;
	}
}

/****************************************************************************/

void cv_hyper::set_parameters(	int a_nx, int a_ny, 
								int a_num_of_levels,
								int a_max_motion,
								int a_num_of_samples,
								int a_border )
{
	this->clear();

	m_num_of_levels		= a_num_of_levels;
	m_max_motion		= a_max_motion;
	m_num_of_samples	= a_num_of_samples;
	m_border			= a_border;
	m_nx = a_nx;
	m_ny = a_ny;
}

/****************************************************************************/

void cv_hyper::move(	float a_row1,
						float a_col1,
						float & a_row2,
						float & a_col2,
						float a_amp )
{
	float l_d = rand()/(RAND_MAX+0.0)*a_amp;
	float l_a = rand()/(RAND_MAX+0.0)*2.0*cv_pi;

	a_col2 = a_col1 + l_d*cosf(l_a);
	a_row2 = a_row1 + l_d*sinf(l_a);
}

/****************************************************************************/

void cv_hyper::add_noise( CvMat * ap_vec )
{
	float * lp_v = ap_vec->data.fl;

	float l_pow = rand()/(RAND_MAX+0.0)*0.1-0.05+1.0;

	for( int l_i=0; l_i<ap_vec->rows; ++l_i ) 
	{
		lp_v[l_i] = pow(lp_v[l_i],l_pow)+rand()/(RAND_MAX+0.0)*10.0-5.0;

		if( lp_v[l_i]<0 ) lp_v[l_i] = 0;
		if( lp_v[l_i]>255 ) lp_v[l_i] = 255;
	}
}

/****************************************************************************/

IplImage * cv_hyper::compute_gradient(	IplImage * ap_image,
										int a_ulr, 
										int a_ulc,
										int a_height,
										int a_width )
{
	IplImage * lp_dx = cvCreateImage(cvSize(a_width,a_height),IPL_DEPTH_32F,1);
	IplImage * lp_dy = cvCreateImage(cvSize(a_width,a_height),IPL_DEPTH_32F,1);
	IplImage * lp_re = cvCreateImage(cvSize(a_width,a_height),IPL_DEPTH_32F,1);

	cvSetImageROI(ap_image,cvRect(a_ulc,a_ulr,a_width,a_height));

	cvSobel(ap_image,lp_dx,1,0,3);
	cvSobel(ap_image,lp_dy,0,1,3);

	cvMul(lp_dx,lp_dx,lp_dx);
	cvMul(lp_dy,lp_dy,lp_dy);
	cvAdd(lp_dx,lp_dy,lp_re);

	cvReleaseImage(&lp_dx);
	cvReleaseImage(&lp_dy);

	cvResetImageROI(ap_image);

	return lp_re;
}

/****************************************************************************/

void cv_hyper::get_local_maximum(	IplImage * ap_image,
									float a_row1, float a_col1,
									float a_height, float a_width,
									float & a_row2, float & a_col2 )
{
	float l_max = -10e10;

	for( int l_r=a_row1-a_height/2; l_r<=a_row1+a_height/2; ++l_r ) 
	{
		float * lp_row = ((float*)((ap_image)->imageData+(l_r)*(ap_image)->widthStep));

		for( int l_c=a_col1-a_width/2; l_c<=a_col1+a_width/2; ++l_c )
		{
			if( lp_row[l_c]>l_max ) 
			{
				l_max  = lp_row[l_c];
				a_col2 = l_c;
				a_row2 = l_r;
			}
		}
	}
}

/****************************************************************************/

void cv_hyper::find_2d_points( IplImage * ap_image )
{
#define CV_GRAD
#ifdef CV_GRAD	
	int l_ulx = CV_MAT_ELEM(*mp_rec,float,0,0);
	int l_uly = CV_MAT_ELEM(*mp_rec,float,1,0);
	int l_lrx = CV_MAT_ELEM(*mp_rec,float,0,2);
	int l_lry = CV_MAT_ELEM(*mp_rec,float,1,2);

	int l_width  = l_lrx-l_ulx;
	int l_height = l_lry-l_uly;
	
	const float l_stepx = float(l_width-2*m_border)/m_nx;
	const float l_stepy = float(l_height-2*m_border)/m_ny;
	
	IplImage * lp_gradient = this->compute_gradient(ap_image,l_uly-1,l_ulx-1,l_height+2,l_width+2);

	float l_var = 0.7;

	for(int l_j=0; l_j<m_ny; ++l_j )
	{
		for(int l_i=0; l_i<m_nx; ++l_i )
		{
			this->get_local_maximum(	lp_gradient,
										(m_border+l_stepy/2+l_j*l_stepy+1.5),
										(m_border+l_stepx/2+l_i*l_stepx+1.5),
										(l_stepy*l_var), (l_stepx*l_var),
										CV_MAT_ELEM(*mp_pos,float,1,l_j*m_nx+l_i),
										CV_MAT_ELEM(*mp_pos,float,0,l_j*m_nx+l_i));

			CV_MAT_ELEM(*mp_pos,float,0,l_j*m_nx+l_i) += l_ulx;
			CV_MAT_ELEM(*mp_pos,float,1,l_j*m_nx+l_i) += l_uly;
		}
	}
	cvReleaseImage(&lp_gradient);
#else
	int l_ulx = CV_MAT_ELEM(*mp_rec,float,0,0);
	int l_uly = CV_MAT_ELEM(*mp_rec,float,1,0);
	int l_lrx = CV_MAT_ELEM(*mp_rec,float,0,2);
	int l_lry = CV_MAT_ELEM(*mp_rec,float,1,2);

	int l_width  = l_lrx-l_ulx;
	int l_height = l_lry-l_uly;
	
	const float l_stepx = l_width/(m_nx-1.0);
	const float l_stepy = l_height/(m_ny-1.0);

	int l_n=0;

	for( int l_i=0; l_i<m_nx; ++l_i )
	{
		for( int l_j=0; l_j<m_ny; ++l_j )
		{
			CV_MAT_ELEM(*mp_pos,float,0,l_n) = l_ulx+l_i*l_stepx;
			CV_MAT_ELEM(*mp_pos,float,1,l_n) = l_uly+l_j*l_stepy;
			CV_MAT_ELEM(*mp_pos,float,2,l_n) = 1.0;
			++l_n;
		}
	}
	return;
#endif
#undef CV_GRAD	
}

/****************************************************************************/

void cv_hyper::compute_as_level(	IplImage * ap_image,
									CvMat * ap_rec,
									CvMat * ap_pos,
									CvMat * ap_int,
									CvMat ** ap_as,
									int a_num_of_samples,
									int a_num_of_levels,
									int a_max_motion,
									int a_level,
									int a_nx, int a_ny )
{
	CvMat * lp_y = cvCreateMat(8,a_num_of_samples,CV_32FC1);
	CvMat * lp_h = cvCreateMat(a_nx*a_ny,a_num_of_samples,CV_32FC1);
	CvMat * lp_hht = cvCreateMat(a_nx*a_ny,a_nx*a_ny,CV_32FC1);
	CvMat * lp_yht = cvCreateMat(8,a_nx*a_ny,CV_32FC1);
	CvMat * lp_inv = cvCreateMat(a_nx*a_ny,a_nx*a_ny,CV_32FC1);
	CvMat * lp_int = cvCreateMat(m_nx*m_ny,1,CV_32F);
	
	int	l_n=0;

	while( l_n<a_num_of_samples ) 
	{
		CvMat * lp_rec = cvCreateMat(3,4,CV_32FC1);
		cvSet(lp_rec,cvRealScalar(1));

		//float l_k = exp(1.0/(a_num_of_levels-1)*log(5.0/a_max_motion));
		//float l_amp = pow(l_k,float(a_level))*a_max_motion;

		float l_amp = (a_num_of_levels-a_level)/(float)(a_num_of_levels)*a_max_motion;

		//std::cerr << l_n << "," << l_amp << std::endl;

		for( int l_i=0; l_i<4; ++l_i ) 
		{
			this->move(	CV_MAT_ELEM(*ap_rec,float,1,l_i),CV_MAT_ELEM(*ap_rec,float,0,l_i),
						CV_MAT_ELEM(*lp_rec,float,1,l_i),CV_MAT_ELEM(*lp_rec,float,0,l_i),
						l_amp);
		}
		for( int l_i=0; l_i<4; ++l_i )
		{
			cvmSet(lp_y,l_i*2+0,l_n,CV_MAT_ELEM(*lp_rec,float,0,l_i)-CV_MAT_ELEM(*ap_rec,float,0,l_i));
			cvmSet(lp_y,l_i*2+1,l_n,CV_MAT_ELEM(*lp_rec,float,1,l_i)-CV_MAT_ELEM(*ap_rec,float,1,l_i));
		}
		CvMat * lp_hom = cv_homography::compute(lp_rec,ap_rec);
		
		if( lp_hom == NULL )
		{
			continue;
		}
		CvMat * lp_pos = cvCreateMat(ap_pos->rows,ap_pos->cols,CV_32FC1);

		cv_mat_mul(lp_hom,ap_pos,lp_pos);
		cv_homogenize(lp_pos);

		for( int l_i=0; l_i<lp_pos->cols; ++l_i )
		{
			int l_col = (int)(CV_MAT_ELEM(*lp_pos,float,0,l_i)+0.5);
			int l_row = (int)(CV_MAT_ELEM(*lp_pos,float,1,l_i)+0.5);

			lp_int->data.fl[l_i] = CV_IMAGE_ELEM(ap_image,float,l_row,l_col);
		}
		this->add_noise(lp_int);

		cv_normalize_mean_std(lp_int->data.fl,lp_int->rows);

		for( int l_i=0; l_i<m_nx*m_ny; ++l_i )
		{
			cvmSet(lp_h,l_i,l_n,lp_int->data.fl[l_i]-ap_int->data.fl[l_i]);
		}
		++l_n;
		
		cvReleaseMat(&lp_pos);
		cvReleaseMat(&lp_hom);
		cvReleaseMat(&lp_rec);
	}
	cv_mat_mul_transposed(lp_h,lp_h,lp_hht);
	cv_mat_mul_transposed(lp_y,lp_h,lp_yht);
	cvInvert(lp_hht,lp_inv,CV_SVD_SYM);
	cv_mat_mul(lp_yht,lp_inv,ap_as[a_level]);

	cvReleaseMat(&lp_inv);
	cvReleaseMat(&lp_yht);
	cvReleaseMat(&lp_hht);
	cvReleaseMat(&lp_int);
	cvReleaseMat(&lp_y);
	cvReleaseMat(&lp_h);
}

/****************************************************************************/

void cv_hyper::compute_as_matrices( IplImage * ap_image,
								    CvMat ** ap_as,
									CvMat * ap_rec,
									CvMat * ap_pos,
									CvMat * ap_int,
									int a_nx, int a_ny,
									int a_max_motion,
									int a_num_of_levels,
									int a_num_of_samples )
{
	int l_level=0;

	#pragma omp parallel for shared(ap_image,a_nx,a_ny,a_num_of_samples, \
	a_num_of_levels,a_max_motion,ap_rec,ap_pos,ap_int,ap_as) private(l_level)
	
	for( l_level=0; l_level<a_num_of_levels; ++l_level ) 
	{
		this->compute_as_level(	ap_image, 
								ap_rec, 
								ap_pos,
								ap_int,
								ap_as,
								a_num_of_samples,
								a_num_of_levels,
								a_max_motion,
								l_level,
								a_nx, a_ny );
	}
}

/****************************************************************************/

void cv_hyper::learn(	IplImage * ap_image,
					    CvMat * ap_rec )
{
	this->clear();

	lp_pix = cvCreateMat(3,m_nx*m_ny,CV_32FC1);

	mp_pos = cvCreateMat(3,m_nx*m_ny,CV_32FC1);
	mp_rec = cvCreateMat(3,4,CV_32FC1);
	
	cvSet(mp_pos,cvRealScalar(1));
	cvCopy(ap_rec,mp_rec);

	this->find_2d_points(ap_image);

	mp_int = cvCreateMat(m_nx*m_ny,1,CV_32FC1);
	
	mp_as = new CvMat*[m_num_of_levels];
	for( int l_level=0; l_level<m_num_of_levels; ++l_level )
	{
		mp_as[l_level] = cvCreateMat(8,m_nx*m_ny,CV_32F);
	}
	for( int l_i=0; l_i<m_nx*m_ny; ++l_i )
	{
		int l_col = (int)CV_MAT_ELEM(*mp_pos,float,0,l_i);
		int l_row = (int)CV_MAT_ELEM(*mp_pos,float,1,l_i);

		mp_int->data.fl[l_i] = CV_IMAGE_ELEM(ap_image,float,l_row,l_col);
	}
	cv_normalize_mean_std(mp_int->data.fl,mp_int->rows);

	this->compute_as_matrices(	ap_image,mp_as,mp_rec,mp_pos,mp_int,m_nx,m_ny,
								m_max_motion,m_num_of_levels,m_num_of_samples);
}

/****************************************************************************/

CvMat * cv_hyper::track( IplImage * ap_image,
						 CvMat * ap_rec,
						 int a_num_of_iters )
{
	CvMat * lp_hom = cv_homography::compute(ap_rec,mp_rec);

	if( lp_hom == NULL )
	{
		return NULL;
	}
	CvMat * lp_int = cvCreateMat(m_nx*m_ny,1,CV_32FC1);
	CvMat * lp_idf = cvCreateMat(m_nx*m_ny,1,CV_32FC1);
	CvMat * lp_udf = cvCreateMat(8,1,CV_32FC1);
	CvMat * lp_res = cvCreateMat(3,4,CV_32FC1);

	for( int l_level=0; l_level<m_num_of_levels; ++l_level ) 
	{
		for( int l_iter=0; l_iter<a_num_of_iters; ++l_iter ) 
		{
			CvMat * lp_pos = cvCreateMat(mp_pos->rows,mp_pos->cols,CV_32FC1);

			cv_mat_mul(lp_hom,mp_pos,lp_pos);
			cv_homogenize(lp_pos);

			cvCopy(lp_pos,lp_pix);

			for( int l_i=0; l_i<lp_pos->cols; ++l_i )
			{
				int l_col = (int)(CV_MAT_ELEM(*lp_pos,float,0,l_i)+0.5);
				int l_row = (int)(CV_MAT_ELEM(*lp_pos,float,1,l_i)+0.5);

				if( l_col<0 || l_row<0 || l_col>=ap_image->width || l_row>=ap_image->height )
				{
					cvReleaseMat(&lp_hom);
					cvReleaseMat(&lp_int);
					cvReleaseMat(&lp_idf);
					cvReleaseMat(&lp_udf);
					cvReleaseMat(&lp_res);
					cvReleaseMat(&lp_pos);
					
					return NULL;
				}
				lp_int->data.fl[l_i] = CV_IMAGE_ELEM(ap_image,float,l_row,l_col);
			}
			cv_normalize_mean_std(lp_int->data.fl,lp_int->rows);
			
			cvSub(lp_int,mp_int,lp_idf);
			cv_mat_mul(mp_as[l_level],lp_idf,lp_udf);

			CvMat * lp_rec = cvCreateMat(3,4,CV_32FC1);
			cvCopy(mp_rec,lp_rec);
			
			CV_MAT_ELEM(*lp_rec,float,0,0) -= lp_udf->data.fl[0];
			CV_MAT_ELEM(*lp_rec,float,1,0) -= lp_udf->data.fl[1]; 
			CV_MAT_ELEM(*lp_rec,float,0,1) -= lp_udf->data.fl[2]; 
			CV_MAT_ELEM(*lp_rec,float,1,1) -= lp_udf->data.fl[3]; 
			CV_MAT_ELEM(*lp_rec,float,0,2) -= lp_udf->data.fl[4]; 
			CV_MAT_ELEM(*lp_rec,float,1,2) -= lp_udf->data.fl[5]; 
			CV_MAT_ELEM(*lp_rec,float,0,3) -= lp_udf->data.fl[6]; 
			CV_MAT_ELEM(*lp_rec,float,1,3) -= lp_udf->data.fl[7]; 

			CvMat * lp_cur = cv_homography::compute(lp_rec,mp_rec);
			
			if( lp_cur == NULL )
			{
				cvReleaseMat(&lp_rec);
				cvReleaseMat(&lp_pos);
				return NULL;
			}
			cv_mat_mul(lp_hom,lp_cur,lp_hom);

			cvReleaseMat(&lp_cur);
			//*/
			/*
			CvMat * lp_war = cvCreateMat(3,4,CV_32FC1);
			cv_mat_mul(lp_hom,lp_rec,lp_war);
			cv_homogenize(lp_war);
			cvReleaseMat(&lp_hom);
			lp_hom = cv_homography::compute(lp_war,mp_rec);
			cvReleaseMat(&lp_war);
			//*/

			cvReleaseMat(&lp_rec);
			cvReleaseMat(&lp_pos);

			float l_norm = 0.0;

			for( int l_j=0; l_j<9; ++l_j )
			{
				l_norm += lp_hom->data.fl[l_j]*lp_hom->data.fl[l_j];
			}
			l_norm = sqrt(l_norm);
			
			for( int l_j=0; l_j<9; ++l_j )
			{
				lp_hom->data.fl[l_j] /= l_norm;
			}
		}
	}
	m_ncc = this->compute_ncc(ap_image,mp_pos,mp_int,lp_hom);

	cv_mat_mul(lp_hom,mp_rec,lp_res);
	cv_homogenize(lp_res);

	cvReleaseMat(&lp_hom);
	cvReleaseMat(&lp_int);
	cvReleaseMat(&lp_idf);
	cvReleaseMat(&lp_udf);
	
	return lp_res;
}

/****************************************************************************/

float cv_hyper::compute_ncc(	IplImage * ap_image,
								CvMat * ap_pos,
								CvMat * ap_int,
								CvMat * ap_hom )
{
	CvMat * lp_pos = cvCreateMat(ap_pos->rows,ap_pos->cols,CV_32FC1);
	CvMat * lp_int = cvCreateMat(ap_int->rows,ap_int->cols,CV_32FC1);

	cv_mat_mul(ap_hom,ap_pos,lp_pos);
	cv_homogenize(lp_pos);

	for( int l_i=0; l_i<lp_pos->cols; ++l_i )
	{
		int l_col = (int)(CV_MAT_ELEM(*lp_pos,float,0,l_i)+0.5);
		int l_row = (int)(CV_MAT_ELEM(*lp_pos,float,1,l_i)+0.5);

		if( l_col<0 || l_row<0 || l_col>=ap_image->width || l_row>=ap_image->height )
		{
			cvReleaseMat(&lp_int);
			cvReleaseMat(&lp_pos);
			
			return -1;
		}
		lp_int->data.fl[l_i] = CV_IMAGE_ELEM(ap_image,float,l_row,l_col);
	}
	cv_normalize_mean_std(lp_int->data.fl,lp_int->rows);

	float l_ncc = cv_dot_product(ap_int->data.fl,lp_int->data.fl,ap_int->rows)/ap_int->rows;

	cvReleaseMat(&lp_pos);
	cvReleaseMat(&lp_int);

	return l_ncc;
}
			
/****************************************************************************/

std::ofstream & cv_hyper::write( std::ofstream & a_os )
{
	a_os.write((char*)&m_nx,sizeof(m_nx));
	a_os.write((char*)&m_ny,sizeof(m_ny));
	a_os.write((char*)&m_ncc,sizeof(m_ncc));
	a_os.write((char*)&m_border,sizeof(m_border));
	a_os.write((char*)&m_max_motion,sizeof(m_max_motion));
	a_os.write((char*)&m_num_of_levels,sizeof(m_num_of_levels));
	a_os.write((char*)&m_num_of_samples,sizeof(m_num_of_samples));

	cv_write(a_os,lp_pix);
	cv_write(a_os,mp_pos);
	cv_write(a_os,mp_rec);
	cv_write(a_os,mp_int);
	
	for( int l_i=0; l_i<m_num_of_levels; ++l_i )
	{
		cv_write(a_os,mp_as[l_i]);
	}
	return a_os;
}

/****************************************************************************/

std::ifstream & cv_hyper::read( std::ifstream & a_is )
{
	this->clear();

	a_is.read((char*)&m_nx,sizeof(m_nx));
	a_is.read((char*)&m_ny,sizeof(m_ny));
	a_is.read((char*)&m_ncc,sizeof(m_ncc));
	a_is.read((char*)&m_border,sizeof(m_border));
	a_is.read((char*)&m_max_motion,sizeof(m_max_motion));
	a_is.read((char*)&m_num_of_levels,sizeof(m_num_of_levels));
	a_is.read((char*)&m_num_of_samples,sizeof(m_num_of_samples));

	lp_pix = cv_read(a_is);
	mp_pos = cv_read(a_is);
	mp_rec = cv_read(a_is);
	mp_int = cv_read(a_is);
	
	mp_as = new CvMat*[m_num_of_levels];

	for( int l_i=0; l_i<m_num_of_levels; ++l_i )
	{
		mp_as[l_i] = cv_read(a_is);
	}
	return a_is;
}

/****************************************************************************/

bool cv_hyper::save( std::string a_name )
{
	std::ofstream l_file(a_name.c_str(),std::ofstream::out|std::ofstream::binary);
	
	if( l_file.fail() == true )
	{
		printf("cv_hyper: could not open for writing!");
		return false;
	}
	this->write(l_file);	

	l_file.close();

	return true;
}

/****************************************************************************/

bool cv_hyper::load( std::string a_name )
{
	std::ifstream l_file(a_name.c_str(),std::ifstream::in|std::ifstream::binary);
	
	if( l_file.fail() == true ) 
	{
		printf("cv_hyper: could not open for reading!");
		return false;
	}
	this->read(l_file);	

	l_file.close();

	return true;
}

/**************************** END OF FILE ***********************************/
