//////////////////////////////////////////////////////////////////////////////
//																			//
// cv_harris: cv_harris.cc													//
//																			//
// Authors: Stefan Hinterstoisser 2007										//
// Lehrstuhl fuer Informatik XVI											//
// Technische Universitaet Muenchen											//
// Version: 1.0																//
//																			//
//////////////////////////////////////////////////////////////////////////////

/******************************* includes ***********************************/

#include "cv_harris.h"
#include "cxmisc.h"
#include <algorithm>

/******************************** defines ***********************************/

#define  cv_cmp_features( f1, f2 )(*(f1) > *(f2))
static CV_IMPLEMENT_QSORT( cv_sort_features, int *, cv_cmp_features )

/******************************** typedefs **********************************/

typedef std::vector<std::pair<int,CvPoint2D32f> > cv_harris_container;

/******************************* namespaces *********************************/

using namespace cv;

/****************************** constructors ********************************/

/******************************** destructor ********************************/

/****************************************************************************/

double	cv_harris::m_radius = 4.0;
double	cv_harris::m_distance = 3.0;
double	cv_harris::m_threshold = 10000.0;
int		cv_harris::m_num_of_points = -1;
int		cv_harris::m_num_of_iterations = 50;

/****************************************************************************/

CvMat * cv_harris::get_points(	IplImage * ap_image )
{
	if( ap_image == NULL )
	{
		printf("cv_harris: ap_image is NULL!");
		return NULL;
	}
	if( ap_image->imageData == NULL )
	{
		printf("cv_harris: ap_image->imageData is NULL!");
		return NULL;
	}
	if( cvGetElemType(ap_image) != CV_32FC1 )
	{
		printf("cv_harris: type is not 32FC1!");
		return NULL;
	}
	if( m_radius <= 0.0 )
	{
		printf("cv_harris: m_radius <= 0.0!");
		return NULL;
	}
	if( m_num_of_points <= 0 )
	{
		printf("cv_harris: m_num_of_points <= 0!");
		return NULL;
	}
	return get_points(	ap_image,
						m_num_of_points,
						m_threshold,
						m_radius );
}

/****************************************************************************/

CvMat * cv_harris::get_most_robust_points(	IplImage * ap_image,
											IplImage * ap_mask )
{
	if( ap_image == NULL )
	{
		printf("cv_harris: ap_image is NULL!");
		return NULL;
	}
	if( ap_image->imageData == NULL )
	{
		printf("cv_harris: ap_image->imageData is NULL!");
		return NULL;
	}
	if( cvGetElemType(ap_image) != CV_32FC1 )
	{
		printf("cv_harris: type is not 32FC1!");
		return NULL;
	}
	if( m_radius <= 0.0 )
	{
		printf("cv_harris: m_radius <= 0.0!");
		return NULL;
	}
	if( m_num_of_points <= 0 )
	{
		printf("cv_harris: m_num_of_points <= 0!");
		return NULL;
	}
	if( ap_mask != NULL )
	{
		if( ap_mask->imageData == NULL )
		{
			printf("cv_harris: ap_mask->imageData is NULL!");
			return NULL;
		}
	}
	return get_most_robust_points(	ap_image,
									ap_mask,
									m_num_of_points,
									m_threshold,
									m_radius );
}

/****************************************************************************/

CvMat * cv_harris::get_points(	IplImage * ap_image,
								int a_num_of_points,
								double a_threshold,
								double a_radius )
{
	IplImage * lp_corners = cvCreateImage(cvGetSize(ap_image),IPL_DEPTH_32F,1);
	IplImage * lp_dilated = cvCreateImage(cvGetSize(ap_image),IPL_DEPTH_32F,1);
	IplImage * lp_tempory = cvCreateImage(cvGetSize(ap_image),IPL_DEPTH_32F,1);

	cvCornerHarris(ap_image,lp_corners,7,5);
	cvDilate(lp_corners,lp_dilated);

	float *  lp_eig_data = (float *)(lp_corners->imageData);
	float *  lp_tmp_data = (float *)(lp_dilated->imageData);
	float ** lp_ptr_data = (float**)(lp_tempory->imageData);
	
	int l_eigstep = ap_image->widthStep/sizeof(float);
	int l_tmpstep = lp_dilated->widthStep/sizeof(float);
	
	int l_height = ap_image->height;
	int l_width  = ap_image->width;

	int l_k = 0; 
	int l_y = 0;
	int l_i = 0;
	int l_x = 0;
	int l_j = 0;

	for( l_y=1; l_y<l_height-1; ++l_y )
	{
		lp_eig_data += l_eigstep;
        lp_tmp_data += l_tmpstep;
        
        for( l_x=1; l_x<l_width-1; ++l_x )
		{
            float l_val = lp_eig_data[l_x];

			if( l_val!=0 && l_val==lp_tmp_data[l_x] && l_val>a_threshold )
			{
        		lp_ptr_data[l_k] = lp_eig_data + l_x;
				++l_k;
			}
        }
    }
	if( l_k <= 0 )
	{
		cvReleaseImage( &lp_dilated );
		cvReleaseImage( &lp_corners );
		cvReleaseImage( &lp_tempory );
		return NULL;
	}
	cv_sort_features( (int**)lp_ptr_data, l_k, 0 );
		
	int l_index = 0;
	int l_min_dist = (a_radius*2+1)*(a_radius*2+1);
	  
	lp_eig_data = (float *)(lp_corners->imageData);
	lp_tmp_data = (float *)(lp_dilated->imageData);
	
	std::vector< std::pair<float,float> > l_container;

	l_container.reserve(l_k);

	for( l_i=0; (l_i<l_k)&&(l_index<a_num_of_points); ++l_i )
	{
		long l_obs = (lp_ptr_data[l_i]-&lp_eig_data[0]);
		int l_yy = l_obs/l_eigstep;
        int l_xx = (l_obs-l_yy*l_eigstep);
		bool l_flag = true; 

		if( l_min_dist != 0 )
		{
        	for( l_j=0; l_j<l_index; ++l_j )
			{
            	int l_dx = l_xx - l_container[l_j].first;
                int l_dy = l_yy - l_container[l_j].second;
                int l_dist = l_dx*l_dx+l_dy*l_dy;

				if( l_dist < l_min_dist )
				{
                	l_flag = false;
					break;
				}
            }
        }
		if( l_flag == true )
		{
			l_container.push_back(std::pair<float,float>(l_xx,l_yy));
			++l_index;
        }
	}
	CvMat * ap_points = NULL;
	
	if( l_index < a_num_of_points )
	{
		ap_points = cvCreateMat(3,l_index,CV_32FC1);
	}
	if( l_index >= a_num_of_points )
	{
		ap_points = cvCreateMat(3,a_num_of_points,CV_32FC1);
	}
	for( int l_i=0; l_i<ap_points->cols; l_i++ )
	{
		CV_MAT_ELEM(*ap_points,float,0,l_i) = l_container[l_i].first;
		CV_MAT_ELEM(*ap_points,float,1,l_i) = l_container[l_i].second;
		CV_MAT_ELEM(*ap_points,float,2,l_i) = 1.0;
	}
	cvReleaseImage( &lp_dilated );
	cvReleaseImage( &lp_corners );
	cvReleaseImage( &lp_tempory );
	
	return ap_points;
}

/****************************************************************************/


CvMat * cv_harris::get_most_robust_points(	IplImage * ap_image,
											IplImage * ap_mask,
											int a_num_of_points,
											double a_threshold,
											double a_radius )
{

	std::vector<std::pair<int,CvPoint2D32f> > l_container;

	IplImage * lp_masked = get_masked_image(ap_image,ap_mask);
	IplImage * lp_warped = cvCreateImage(	cvGetSize(ap_image),
											IPL_DEPTH_32F,
											1);

	for( int l_i=0; l_i<m_num_of_iterations; ++l_i )
	{
		std::pair<CvMat*,CvMat*> l_pair = get_random_transformation(
													ap_image->height/2,
													ap_image->width/2);
		cvWarpAffine(	lp_masked,
						lp_warped,
						l_pair.first,
						CV_WARP_FILL_OUTLIERS,
						cvRealScalar(0));

		CvMat * lp_points = get_points(	lp_warped,
										m_num_of_points*3,
										m_threshold,
										m_radius);
		if( lp_points == NULL )
		{
			printf("cv_harris: the points are NULL!");
			return NULL;
		}
		backproject_and_match(l_container,l_pair.second,lp_points);

		printf("%d points - %d.th iteration...\n",l_container.size(),l_i);
		cvReleaseMat(&l_pair.first);
		cvReleaseMat(&l_pair.second);
		cvReleaseMat(&lp_points);
	}
	struct cv_sorter
	{
		bool operator()(	std::pair<int,CvPoint2D32f> & a_lhs,
							std::pair<int,CvPoint2D32f> & a_rhs )
		{
			return a_lhs.first > a_rhs.first;
		}
	};
	int l_inside=0;

	if( ap_mask == NULL )
	{
		l_inside = l_container.size();
	}
	else
	{
		for( int l_i=0; l_i<l_container.size(); ++l_i )
		{
			float l_c = l_container[l_i].second.x/l_container[l_i].first;
			float l_r = l_container[l_i].second.y/l_container[l_i].first;

			if( l_c >= 0 &&
				l_c < ap_mask->width &&
				l_r >= 0 &&
				l_r < ap_mask->height )
			{
				if( CV_IMAGE_ELEM(ap_mask,unsigned char,(int)(l_r),(int)(l_c)) > 0 )
					++l_inside;
			}
		}
	}
	if( l_inside <= 0 )
	{
		cvReleaseImage(&lp_warped);
		cvReleaseImage(&lp_masked);
	
		return NULL;
	}
	std::sort(l_container.begin(),l_container.end(),cv_sorter());

	if( a_num_of_points > l_inside )
		a_num_of_points = l_inside;

	CvMat * lp_points1 = cvCreateMat(3,a_num_of_points,CV_32FC1);

	int l_i=0;
	int l_j=0;

	while( l_j < a_num_of_points )
	{
		if( ap_mask == NULL )
		{
			float l_c = l_container[l_i].second.x/l_container[l_i].first;
			float l_r = l_container[l_i].second.y/l_container[l_i].first;

			if(	l_c >= 0 &&
				l_c < ap_image->width &&
				l_r >= 0 &&
				l_r < ap_image->height )
			{
				CV_MAT_ELEM(*lp_points1,float,0,l_j) = l_c;
				CV_MAT_ELEM(*lp_points1,float,1,l_j) = l_r;
				CV_MAT_ELEM(*lp_points1,float,2,l_j) = 1;
				++l_j;
			}
			++l_i;
		}
		else
		{
			float l_c = l_container[l_i].second.x/l_container[l_i].first;
			float l_r = l_container[l_i].second.y/l_container[l_i].first;

			if(	l_c >= 0 &&
				l_c < ap_image->width &&
				l_r >= 0 &&
				l_r < ap_image->height )
			{
				if( CV_IMAGE_ELEM(ap_mask,unsigned char,(int)(l_r),(int)(l_c)) > 0 )
				{
					CV_MAT_ELEM(*lp_points1,float,0,l_j) =	l_container[l_i].second.x/
															l_container[l_i].first;
					CV_MAT_ELEM(*lp_points1,float,1,l_j) =	l_container[l_i].second.y/
															l_container[l_i].first;
					CV_MAT_ELEM(*lp_points1,float,2,l_j) =	1;
					++l_j;
				}
			}
			++l_i;
		}
	}
	cvReleaseImage(&lp_warped);
	cvReleaseImage(&lp_masked);
	
	return lp_points1;
}

/****************************************************************************/

void cv_harris::backproject_and_match(	cv_harris_container & a_con,
										CvMat * ap_transformation,
										CvMat * ap_points )
{
	CvMat * lp_inverse = cvCreateMat(3,3,CV_32FC1);
	CvMat * lp_points = cvCreateMat(ap_points->rows,
									ap_points->cols,
									CV_32FC1);

	cvInvert(ap_transformation,lp_inverse,CV_SVD);
	cvMatMul(lp_inverse,ap_points,lp_points);
	cv_homogenize(lp_points);

	for( int l_i=0; l_i<lp_points->cols; ++l_i)
	{
		bool l_taken_flag = false;

		double l_min_distance=10e10;
		int l_min_index=-1;

		for( int l_j=0; l_j<a_con.size(); ++l_j)
		{
			double l_distance = sqrt(
								SQR(CV_MAT_ELEM(*lp_points,float,0,l_i)-
								a_con[l_j].second.x/a_con[l_j].first)+
								SQR(CV_MAT_ELEM(*lp_points,float,1,l_i)-
								a_con[l_j].second.y/a_con[l_j].first));
			
			if( l_distance < m_distance	&& 
				l_distance < l_min_distance )
			{
				l_taken_flag = true;
				l_min_distance = l_distance;
				l_min_index = l_j;
			}
		}
		if( l_taken_flag == false )
		{
			std::pair<int,CvPoint2D32f> l_pair;

			l_pair.first = 1;
			l_pair.second.x = CV_MAT_ELEM(*lp_points,float,0,l_i);
			l_pair.second.y = CV_MAT_ELEM(*lp_points,float,1,l_i);
			a_con.push_back( l_pair );
		}
		else
		{
			a_con[l_min_index].first += 1;
			a_con[l_min_index].second.x += CV_MAT_ELEM(*lp_points,float,0,l_i);
			a_con[l_min_index].second.y += CV_MAT_ELEM(*lp_points,float,1,l_i);
		}
	}
	cvReleaseMat(&lp_points);
	cvReleaseMat(&lp_inverse);
}

/****************************************************************************/

IplImage * cv_harris::get_masked_image( IplImage * ap_image, 
										IplImage * ap_mask )
{
	IplImage * lp_result = cvCreateImage(	cvGetSize(ap_image),
											IPL_DEPTH_32F,
											ap_image->nChannels);
	cvSet(lp_result,cvRealScalar(-1));

	if( ap_mask == NULL )
	{
		cvCopy(ap_image,lp_result);
	}
	else
	{
		IplImage * lp_mask = cvCreateImage(	cvGetSize(ap_image),
											IPL_DEPTH_8U,
											1);
		IplConvKernel * lp_kernel = cvCreateStructuringElementEx(
									21,21,10,10,CV_SHAPE_ELLIPSE);
		cvDilate(ap_mask,lp_mask,lp_kernel,1);
		cvCopy(ap_image,lp_result,lp_mask);
		cvReleaseStructuringElement(&lp_kernel);
		cvReleaseImage(&lp_mask);
	}
	return lp_result;
}

/****************************************************************************/

std::pair<CvMat*,CvMat*> cv_harris::get_random_transformation(	int a_row,
																int a_col )
{
	double l_lam1 = rand()/(RAND_MAX+0.0)*1.0+0.5;
	double l_lam2 = rand()/(RAND_MAX+0.0)*1.0+0.5;
	double l_the  = rand()/(RAND_MAX+0.0)*360.0;
	double l_phi  = rand()/(RAND_MAX+0.0)*180.0;
	
	l_the *= cv_pi/180.0;
	l_phi *= cv_pi/180.0;

	CvMat * lp_dst_mat = cvCreateMat(2,3,CV_32FC1);
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

	float l_bx = -(CV_MAT_ELEM(*lp_aff_mat,float,0,0)*a_col+CV_MAT_ELEM(*lp_aff_mat,float,0,1)*a_row)+a_col;
	float l_by = -(CV_MAT_ELEM(*lp_aff_mat,float,1,0)*a_col+CV_MAT_ELEM(*lp_aff_mat,float,1,1)*a_row)+a_row;

	CV_MAT_ELEM(*lp_aff_mat,float,0,2) = l_bx;
	CV_MAT_ELEM(*lp_aff_mat,float,1,2) = l_by;
	
	CV_MAT_ELEM(*lp_dst_mat,float,0,0) = CV_MAT_ELEM(*lp_aff_mat,float,0,0);
	CV_MAT_ELEM(*lp_dst_mat,float,1,0) = CV_MAT_ELEM(*lp_aff_mat,float,1,0);
	CV_MAT_ELEM(*lp_dst_mat,float,0,1) = CV_MAT_ELEM(*lp_aff_mat,float,0,1);
	CV_MAT_ELEM(*lp_dst_mat,float,1,1) = CV_MAT_ELEM(*lp_aff_mat,float,1,1);
	CV_MAT_ELEM(*lp_dst_mat,float,0,2) = CV_MAT_ELEM(*lp_aff_mat,float,0,2);
	CV_MAT_ELEM(*lp_dst_mat,float,1,2) = CV_MAT_ELEM(*lp_aff_mat,float,1,2);

    cvReleaseMat(&lp_tmp_mat1);
    cvReleaseMat(&lp_tmp_mat2);
	cvReleaseMat(&lp_sca_mat);
	cvReleaseMat(&lp_tro_mat);
	cvReleaseMat(&lp_ppr_mat);
	cvReleaseMat(&lp_mpr_mat);

	return std::pair<CvMat*,CvMat*>(lp_dst_mat,lp_aff_mat);
}

/******************************* END OF FILE ********************************/

