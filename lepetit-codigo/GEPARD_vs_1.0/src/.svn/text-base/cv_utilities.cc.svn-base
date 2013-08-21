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
// cv_utilities: cv_utilities.cc											//
//																			//
// Authors: Stefan Hinterstoisser 2010										//
// Version: 1.0																//
//																			//
//////////////////////////////////////////////////////////////////////////////

/******************************* includes ***********************************/

#include "cv_utilities.h"
#include <omp.h>



#ifdef IPP_INCLUDED 
#include "ippi.h"
#include "ipps.h" 
#include "ippm.h" 
#endif

/******************************** defines ***********************************/

/******************************* namespaces *********************************/

/****************************** constructors ********************************/

/******************************** destructor ********************************/

/****************************************************************************/

void cv::print_bit( Ipp8u a_n )
{
	std::cerr << "8ubit " << (int)a_n << ": ";

	for( int l_i=0; l_i<8; ++l_i )
	{
		if( a_n & 0x1u == 1 ) std::cerr << "1"; else std::cerr << "0";     
		a_n >>= 1;
	}
	std::cerr << std::endl;
}
	
void cv::print_bit( Ipp16u a_n )
{
	std::cerr << "16ubit " << a_n << ": ";

	for( int l_i=0; l_i<16; ++l_i )
	{
		if( a_n & 0x1u == 1 ) std::cerr << "1"; else std::cerr << "0";     
		a_n >>= 1;
	}
	std::cerr << std::endl;
}

void cv::print_bit( Ipp32s a_n )
{
	std::cerr << "32sbit " << a_n << ": ";

	for( int l_i=0; l_i<32; ++l_i )
	{
		if( a_n & 0x1u == 1 ) std::cerr << "1"; else std::cerr << "0";     
		a_n >>= 1;
	}
	std::cerr << std::endl;
}

void cv::print_bit( Ipp64u a_n )
{
	std::cerr << "64ubit " << a_n << ": ";

	for( int l_i=0; l_i<64; ++l_i )
	{
		if( a_n & 0x1u == 1 ) std::cerr << "1"; else std::cerr << "0";     
		a_n >>= 1;
	}
	std::cerr << std::endl;
}


void cv::print_bit( __m128i a_n )
{
	std::cerr << "128ibit: ";

	Ipp8u * lp_ptr = (Ipp8u*)&a_n;

	for( int l_j=0; l_j<16; ++l_j )
	{
		for( int l_i=0; l_i<8; ++l_i )
		{
			if( lp_ptr[l_j] & 0x1u == 1 ) std::cerr << "1"; else std::cerr << "0";     
			lp_ptr[l_j] >>= 1;
		}
		std::cerr << "|";
	}
	std::cerr << std::endl;
}

/****************************************************************************/

void cv::cv_copy( float * ap_src, float * ap_dst, int a_length )
{
#ifdef IPP_INCLUDED
	ippsCopy_32f(ap_src,ap_dst,a_length);
#else

	int l_c = 0;

	int l_length_mod = (a_length/8)*8;
	
	for(;l_c<l_length_mod;)
	{
		ap_dst[0] =	ap_src[0];
		ap_dst[1] =	ap_src[1];
		ap_dst[2] =	ap_src[2];
		ap_dst[3] =	ap_src[3];
		ap_dst[4] =	ap_src[4];
		ap_dst[5] =	ap_src[5];
		ap_dst[6] =	ap_src[6];
		ap_dst[7] =	ap_src[7];

		ap_src+=8;
		ap_dst+=8;
		l_c+=8;
	}
	for(;l_c<a_length;)
	{
		ap_dst[0] = ap_src[0];
		++ap_src;
		++ap_dst;
		++l_c;
	}
#endif
}

/****************************************************************************/

void cv::cv_add( float * ap_src, float * ap_dst, int a_length )
{
#ifdef IPP_INCLUDED
	ippsAdd_32f_I(ap_src,ap_dst,a_length);
#else
	int l_c = 0;

	int l_length_mod = (a_length/8)*8;
	
	for(;l_c<l_length_mod;)
	{
		ap_dst[0] +=	ap_src[0];
		ap_dst[1] +=	ap_src[1];
		ap_dst[2] +=	ap_src[2];
		ap_dst[3] +=	ap_src[3];
		ap_dst[4] +=	ap_src[4];
		ap_dst[5] +=	ap_src[5];
		ap_dst[6] +=	ap_src[6];
		ap_dst[7] +=	ap_src[7];

		ap_src+=8;
		ap_dst+=8;
		l_c+=8;
	}
	for(;l_c<a_length;)
	{
		ap_dst[0] += ap_src[0];
		++ap_src;
		++ap_dst;
		++l_c;
	}
#endif
}

/****************************************************************************/

void cv::cv_set( float * ap_srcdst, float a_scalar, int a_length )
{
#ifdef IPP_INCLUDED
	ippsSet_32f(a_scalar,ap_srcdst,a_length);
#else
	int l_c = 0;

	int l_length_mod = (a_length/8)*8;
	
	for(;l_c<l_length_mod;)
	{
		ap_srcdst[0] =	a_scalar;
		ap_srcdst[1] =	a_scalar;
		ap_srcdst[2] =	a_scalar;
		ap_srcdst[3] =	a_scalar;
		ap_srcdst[4] =	a_scalar;
		ap_srcdst[5] =	a_scalar;
		ap_srcdst[6] =	a_scalar;
		ap_srcdst[7] =	a_scalar;

		ap_srcdst+=8;
		l_c+=8;
	}
	for(;l_c<a_length;)
	{
		ap_srcdst[0] = a_scalar;
		++ap_srcdst;
		++l_c;
	}
#endif
}

/****************************************************************************/

void cv::cv_sqrt( float * ap_src, float * ap_dst, int a_length )
{
#ifdef IPP_INCLUDED
	ippsSqrt_32f(ap_src,ap_dst,a_length);
#else
	int l_c = 0;

	int l_length_mod = (a_length/8)*8;
	
	float * lp_dst = ap_dst;
	float * lp_src = ap_src;

	for(;l_c<l_length_mod;)
	{
		lp_dst[0] =	sqrtf(lp_src[0]);
		lp_dst[1] =	sqrtf(lp_src[1]);
		lp_dst[2] =	sqrtf(lp_src[2]);
		lp_dst[3] =	sqrtf(lp_src[3]);
		lp_dst[4] =	sqrtf(lp_src[4]);
		lp_dst[5] =	sqrtf(lp_src[5]);
		lp_dst[6] =	sqrtf(lp_src[6]);
		lp_dst[7] =	sqrtf(lp_src[7]);

		lp_dst+=8;
		lp_src+=8;
		l_c+=8;
	}
	for(;l_c<a_length;)
	{
		lp_dst[0] = sqrtf(lp_src[0]);
		++lp_dst;
		++lp_src;
		++l_c;
	}
#endif
}

/****************************************************************************/

void cv::cv_sqr( float * ap_src, float * ap_dst, int a_length )
{
#ifdef IPP_INCLUDED
	ippsSqr_32f(ap_src,ap_dst,a_length);
#else
	int l_c = 0;

	int l_length_mod = (a_length/8)*8;
	
	float * lp_dst = ap_dst;
	float * lp_src = ap_src;

	for(;l_c<l_length_mod;)
	{
		lp_dst[0] =	SQR(lp_src[0]);
		lp_dst[1] =	SQR(lp_src[1]);
		lp_dst[2] =	SQR(lp_src[2]);
		lp_dst[3] =	SQR(lp_src[3]);
		lp_dst[4] =	SQR(lp_src[4]);
		lp_dst[5] =	SQR(lp_src[5]);
		lp_dst[6] =	SQR(lp_src[6]);
		lp_dst[7] =	SQR(lp_src[7]);

		lp_dst+=8;
		lp_src+=8;
		l_c+=8;
	}
	for(;l_c<a_length;)
	{
		lp_dst[0] = SQR(lp_src[0]);
		++lp_dst;
		++lp_src;
		++l_c;
	}
#endif
}

/****************************************************************************/

std::pair<int,float> cv::cv_find_max( float * ap_src, int a_length )
{
	std::pair<int,float> l_max(-1,-10e100);

#ifdef IPP_INCLUDED
	ippsMaxIndx_32f(ap_src,a_length,&l_max.second,&l_max.first);
#else
	for( int l_c=0; l_c<a_length; ++l_c )
	{
		if( ap_src[l_c] > l_max.second )
		{
			l_max.second = ap_src[l_c];
			l_max.first  = l_c;
		}
	}
#endif
	return l_max;
}

/****************************************************************************/

float cv::cv_dot_product( float * ap_src1, float * ap_src2, int a_length )
{
	float l_val = 0;

#ifdef IPP_INCLUDED
	ippsDotProd_32f(ap_src1,ap_src2,a_length,&l_val);
#else

	int l_c=0;
	int l_length_mod = (a_length/8)*8;

	float * lp_src1 = ap_src1;
	float * lp_src2 = ap_src2;
	
	for(;l_c<l_length_mod;)
	{
		l_val +=	lp_src1[0]*lp_src2[0]+
					lp_src1[1]*lp_src2[1]+
					lp_src1[2]*lp_src2[2]+
					lp_src1[3]*lp_src2[3]+
					lp_src1[4]*lp_src2[4]+
					lp_src1[5]*lp_src2[5]+
					lp_src1[6]*lp_src2[6]+
					lp_src1[7]*lp_src2[7];

		lp_src1+=8;
		lp_src2+=8;
		l_c+=8;
	}
	for(;l_c<a_length;)
	{
		l_val += lp_src1[0]*lp_src2[0];
		++lp_src1;
		++lp_src2;
		++l_c;
	}
#endif
	return l_val;
}

/****************************************************************************/

void cv::cv_lin_combination(	float * ap_src1, float a_scalar1,
								float * ap_src2, float a_scalar2,
								float * ap_dest, int a_length )
{
#ifdef IPP_INCLUDED
	ippmLComb_vv_32f(ap_src1, sizeof(Ipp32f), a_scalar1,
				     ap_src2, sizeof(Ipp32f), a_scalar2, 
					 ap_dest, sizeof(Ipp32f), a_length );
#else

	int l_c=0;
	int l_length_mod = (a_length/8)*8;
	
	float * lp_dest = ap_dest;
	float * lp_src1 = ap_src1;
	float * lp_src2 = ap_src2;


	for(;l_c<l_length_mod;)
	{
		lp_dest[0] = lp_src1[0]*a_scalar1+lp_src2[0]*a_scalar2;
		lp_dest[1] = lp_src1[1]*a_scalar1+lp_src2[1]*a_scalar2;
		lp_dest[2] = lp_src1[2]*a_scalar1+lp_src2[2]*a_scalar2;
		lp_dest[3] = lp_src1[3]*a_scalar1+lp_src2[3]*a_scalar2;
		lp_dest[4] = lp_src1[4]*a_scalar1+lp_src2[4]*a_scalar2;
		lp_dest[5] = lp_src1[5]*a_scalar1+lp_src2[5]*a_scalar2;
		lp_dest[6] = lp_src1[6]*a_scalar1+lp_src2[6]*a_scalar2;
		lp_dest[7] = lp_src1[7]*a_scalar1+lp_src2[7]*a_scalar2;

		lp_src1+=8;
		lp_src2+=8;
		lp_dest+=8;
		l_c+=8;
	}
	for(;l_c<a_length;)
	{
		lp_dest[0] = lp_src1[0]*a_scalar1+lp_src2[0]*a_scalar2;
		++lp_src1;
		++lp_src2;
		++lp_dest;
		++l_c;
	}
#endif
}

/****************************************************************************/

void cv::cv_normalize_mean_std( float * ap_src, int a_length )
{
#ifdef IPP_INCLUDED

	float l_mean=0;
	float l_std=0;


	ippsMeanStdDev_32f( ap_src, 
						a_length,
						&l_mean,
						&l_std,
						ippAlgHintFast );

	ippsNormalize_32f(	ap_src,
						ap_src,
						a_length,
						l_mean,
						l_std );
#else

	CvScalar l_mean;
	CvScalar l_std_dev;

	l_mean.val[0] = 0;
	l_std_dev.val[0] = 0;

	for( int l_i=0; l_i<a_length; ++l_i )
		l_mean.val[0] += ap_src[l_i];

	l_mean.val[0] /= a_length;

	for( int l_i=0; l_i<a_length; ++l_i )
		l_std_dev.val[0] += SQR(ap_src[l_i]-l_mean.val[0]);

	l_std_dev.val[0] = sqrt(l_std_dev.val[0]/(a_length-1));
	
	l_std_dev.val[0] = 1.0/l_std_dev.val[0];
	
	int l_c=0;
	int l_num = a_length;
	int l_num_mod = (l_num/8)*8;
	float * lp_pointer = ap_src;
	
	for(;l_c<l_num_mod;)
	{
		lp_pointer[0] = (lp_pointer[0]-l_mean.val[0])*l_std_dev.val[0];
		lp_pointer[1] = (lp_pointer[1]-l_mean.val[0])*l_std_dev.val[0];
		lp_pointer[2] = (lp_pointer[2]-l_mean.val[0])*l_std_dev.val[0];
		lp_pointer[3] = (lp_pointer[3]-l_mean.val[0])*l_std_dev.val[0];
		lp_pointer[4] = (lp_pointer[4]-l_mean.val[0])*l_std_dev.val[0];
		lp_pointer[5] = (lp_pointer[5]-l_mean.val[0])*l_std_dev.val[0];
		lp_pointer[6] = (lp_pointer[6]-l_mean.val[0])*l_std_dev.val[0];
		lp_pointer[7] = (lp_pointer[7]-l_mean.val[0])*l_std_dev.val[0];
		lp_pointer+=8;
		l_c+=8;
	}
	for(;l_c<l_num;)
	{
		lp_pointer[0] = (lp_pointer[0]-l_mean.val[0])*l_std_dev.val[0];
		++lp_pointer;
		++l_c;
	}
#endif
}

/****************************************************************************/

void cv::cv_normalize_mean( float * ap_src, int a_length )
{
#ifdef IPP_INCLUDED

	float l_mean=0;
	float l_std=0;

	ippsMean_32f(ap_src,a_length,&l_mean,ippAlgHintFast );
	ippsSubC_32f_I(l_mean,ap_src,a_length);
	
#else

	CvScalar l_mean;
	
	l_mean.val[0] = 0;
	
	for( int l_i=0; l_i<a_length; ++l_i )
		l_mean.val[0] += ap_src[l_i];

	l_mean.val[0] /= a_length;

	int l_c=0;
	int l_num = a_length;
	int l_num_mod = (l_num/8)*8;
	float * lp_pointer = ap_src;
	
	for(;l_c<l_num_mod;)
	{
		lp_pointer[0] = (lp_pointer[0]-l_mean.val[0]);
		lp_pointer[1] = (lp_pointer[1]-l_mean.val[0]);
		lp_pointer[2] = (lp_pointer[2]-l_mean.val[0]);
		lp_pointer[3] = (lp_pointer[3]-l_mean.val[0]);
		lp_pointer[4] = (lp_pointer[4]-l_mean.val[0]);
		lp_pointer[5] = (lp_pointer[5]-l_mean.val[0]);
		lp_pointer[6] = (lp_pointer[6]-l_mean.val[0]);
		lp_pointer[7] = (lp_pointer[7]-l_mean.val[0]);
		lp_pointer+=8;
		l_c+=8;
	}
	for(;l_c<l_num;)
	{
		lp_pointer[0] = (lp_pointer[0]-l_mean.val[0]);
		++lp_pointer;
		++l_c;
	}
#endif
}


/***************************************************************************/

void cv::cv_normalize_std( float * ap_src, int a_length )
{
#ifdef IPP_INCLUDED

	float l_std=0;

	ippsNorm_L2_32f(ap_src,a_length,&l_std);
	ippsDivC_32f_I(l_std,ap_src,a_length);

#else

	double l_std_dev=0;

	for( int l_i=0; l_i<a_length; ++l_i )
		l_std_dev += SQR(ap_src[l_i]);

	l_std_dev = sqrt(l_std_dev);
	
	l_std_dev = 1.0/l_std_dev;
	
	int l_c=0;
	int l_num = a_length;
	int l_num_mod = (l_num/8)*8;
	float * lp_pointer = ap_src;
	
	for(;l_c<l_num_mod;)
	{
		lp_pointer[0] = (lp_pointer[0])*l_std_dev;
		lp_pointer[1] = (lp_pointer[1])*l_std_dev;
		lp_pointer[2] = (lp_pointer[2])*l_std_dev;
		lp_pointer[3] = (lp_pointer[3])*l_std_dev;
		lp_pointer[4] = (lp_pointer[4])*l_std_dev;
		lp_pointer[5] = (lp_pointer[5])*l_std_dev;
		lp_pointer[6] = (lp_pointer[6])*l_std_dev;
		lp_pointer[7] = (lp_pointer[7])*l_std_dev;
		lp_pointer+=8;
		l_c+=8;
	}
	for(;l_c<l_num;)
	{
		lp_pointer[0] = (lp_pointer[0])*l_std_dev;
		++lp_pointer;
		++l_c;
	}
#endif
}


/****************************************************************************/

void cv::cv_mat_mul( CvMat * ap_src1, CvMat * ap_src2, CvMat * ap_src3 )
{
#ifdef IPP_INCLUDED
	ippmMul_mm_32f( (float*)ap_src1->data.ptr, sizeof(float)*ap_src1->cols, sizeof(float), ap_src1->cols, ap_src1->rows,
					(float*)ap_src2->data.ptr, sizeof(float)*ap_src2->cols, sizeof(float), ap_src2->cols, ap_src2->rows,
					(float*)ap_src3->data.ptr, sizeof(float)*ap_src3->cols, sizeof(float) );
#else
	cvGEMM(ap_src1,ap_src2,1,NULL,0,ap_src3);
#endif
}

/****************************************************************************/

void cv::cv_mat_mul_transposed( CvMat * ap_src1, CvMat * ap_src2, CvMat * ap_src3 )
{
#ifdef IPP_INCLUDED
	ippmMul_mt_32f(	(float*)ap_src1->data.ptr, sizeof(float)*ap_src1->cols, sizeof(float), ap_src1->cols, ap_src1->rows,
					(float*)ap_src2->data.ptr, sizeof(float)*ap_src2->cols, sizeof(float), ap_src2->cols, ap_src2->rows,
					(float*)ap_src3->data.ptr, sizeof(float)*ap_src3->cols, sizeof(float) );
#else
	cvGEMM(ap_src1,ap_src2,1,NULL,0,ap_src3,CV_GEMM_B_T);
#endif
}

/****************************************************************************/

IplImage * cv::cv_convert_color_to_gray( IplImage * ap_image )
{
	if( ap_image == NULL )
	{
		printf("convert_color_to_gray: ap_image is NULL!");
		return NULL;
	}
	if( ap_image->imageData == NULL )
	{
		printf("convert_color_to_gray: imageData is NULL!");
		return NULL;
	}
	if( ap_image->nChannels == 3 )
	{
		IplImage * lp_image2 = cvCreateImage(cvGetSize(ap_image),IPL_DEPTH_32F,1);

		cvCvtColor(ap_image,lp_image2,CV_BGR2GRAY);
			
		return lp_image2;
	}
	if( ap_image->nChannels == 1 )
	{
		IplImage * lp_image2 = cvCreateImage(cvGetSize(ap_image),IPL_DEPTH_32F,1);

		cvCopy(ap_image,lp_image2);
			
		return lp_image2;
	}
	return NULL;
}

/****************************************************************************/

IplImage * cv::cv_convert_gray_to_color( IplImage * ap_image )
{
	if( ap_image == NULL )
	{
		printf("convert_gray_to_color: ap_image is NULL!");
		return NULL;
	}
	if( ap_image->imageData == NULL )
	{
		printf("convert_gray_to_color: imageData is NULL!");
		return NULL;
	}
	if( ap_image->nChannels == 1 )
	{
		IplImage * lp_image2 = cvCreateImage(cvGetSize(ap_image),IPL_DEPTH_32F,3);

		cvCvtColor(ap_image,lp_image2,CV_GRAY2BGR);
					
		return lp_image2;
	}
	return NULL;
}

/****************************************************************************/

IplImage * cv::cv_smooth(	IplImage * ap_image,
							int a_block_size )
{
	if( a_block_size > 7 ||
		a_block_size < 3 )
	{
		printf("cv_smooth: a_block_size does not fit!");
		return NULL;
	}
	IplImage * lp_image = cvCreateImage(cvGetSize(ap_image),
										IPL_DEPTH_32F,
										ap_image->nChannels);

	cvSmooth(ap_image,lp_image,CV_BLUR,a_block_size,a_block_size,0,0);

	return lp_image;
}

/****************************************************************************/

IplImage * cv::cv_roi_mask(	IplImage * ap_image,
							CvMat * ap_points )
{
	IplImage * lp_image = cvCreateImage(cvGetSize(ap_image),IPL_DEPTH_8U,1);
	CvPoint * lp_points = new CvPoint[ap_points->cols];

	cvSet(lp_image,cvRealScalar(0));

	for( int l_i=0; l_i<ap_points->cols; ++l_i )
	{
		lp_points[l_i].x = CV_MAT_ELEM(*ap_points,float,0,l_i);
		lp_points[l_i].y = CV_MAT_ELEM(*ap_points,float,1,l_i);
	}
	cvFillConvexPoly( lp_image, lp_points, ap_points->cols, CV_RGB(255,255,255) );

	delete[] lp_points;
	return lp_image;
}

/****************************************************************************/

CvMat * cv::cv_right_pseudo_inverse( CvMat * ap_mat )
{

	CvMat * lp_inv = cvCreateMat(ap_mat->cols,ap_mat->cols,CV_32FC1);
	CvMat * lp_sym = cvCreateMat(ap_mat->cols,ap_mat->cols,CV_32FC1);
	CvMat * lp_dst = cvCreateMat(ap_mat->cols,ap_mat->rows,CV_32FC1);
	
	cvGEMM(ap_mat,ap_mat,1,NULL,0,lp_sym,CV_GEMM_A_T);
	cvInvert(lp_sym,lp_inv,CV_SVD_SYM);
	cvGEMM(lp_inv,ap_mat,1,NULL,0,lp_dst,CV_GEMM_B_T);
	
	cvReleaseMat(&lp_inv);
	cvReleaseMat(&lp_sym);

	return lp_dst;
}

/***************************************************************************/

CvMat * cv::cv_left_pseudo_inverse( CvMat * ap_mat )
{

	CvMat * lp_inv = cvCreateMat(ap_mat->rows,ap_mat->rows,CV_32FC1);
	CvMat * lp_sym = cvCreateMat(ap_mat->rows,ap_mat->rows,CV_32FC1);
	CvMat * lp_dst = cvCreateMat(ap_mat->cols,ap_mat->rows,CV_32FC1);
	
	cvGEMM(ap_mat,ap_mat,1,NULL,0,lp_sym,CV_GEMM_B_T);
	cvInvert(lp_sym,lp_inv,CV_SVD_SYM);
	cvGEMM(ap_mat,lp_inv,1,NULL,0,lp_dst,CV_GEMM_A_T);
	
	cvReleaseMat(&lp_inv);
	cvReleaseMat(&lp_sym);

	return lp_dst;
}

/***************************************************************************/

CvMat * cv::cv_pseudo_inverse( CvMat * ap_mat )
{
	if( ap_mat == NULL )
	{
		printf("cv_pseudo_inverse: ap_mat is NULL!");
		return NULL;
	}
	if( ap_mat->rows >= ap_mat->cols )
	{
		CvMat * lp_u = cvCreateMat(ap_mat->rows,ap_mat->cols,CV_32FC1);
		CvMat * lp_w = cvCreateMat(ap_mat->cols,ap_mat->cols,CV_32FC1);
		CvMat * lp_v = cvCreateMat(ap_mat->cols,ap_mat->cols,CV_32FC1);

		cvSVD(ap_mat,lp_w,lp_u,lp_v);

		CvMat * lp_r = cvCreateMat(ap_mat->cols,ap_mat->rows,CV_32FC1);
		CvMat * lp_t = cvCreateMat(ap_mat->cols,ap_mat->cols,CV_32FC1);

		for( int l_i=0; l_i<ap_mat->cols; ++l_i )
		{
			if( CV_MAT_ELEM(*lp_w,float,l_i,l_i) != 0.0 )
			{
				CV_MAT_ELEM(*lp_w,float,l_i,l_i) = 1.0/CV_MAT_ELEM(*lp_w,float,l_i,l_i);
			}
		}
		cvGEMM(lp_v,lp_w,1,NULL,0.0,lp_t);
		cvGEMM(lp_t,lp_u,1,NULL,0.0,lp_r,CV_GEMM_B_T);

		cvReleaseMat(&lp_u);
		cvReleaseMat(&lp_w);
		cvReleaseMat(&lp_v);
		cvReleaseMat(&lp_t);

		return lp_r;
	}
	else
	{
		CvMat * lp_u = cvCreateMat(ap_mat->rows,ap_mat->rows,CV_32FC1);
		CvMat * lp_w = cvCreateMat(ap_mat->rows,ap_mat->cols,CV_32FC1);
		CvMat * lp_v = cvCreateMat(ap_mat->cols,ap_mat->cols,CV_32FC1);

		cvSVD(ap_mat,lp_w,lp_u,lp_v);

		CvMat * lp_r = cvCreateMat(ap_mat->cols,ap_mat->rows,CV_32FC1);
		CvMat * lp_t = cvCreateMat(ap_mat->cols,ap_mat->rows,CV_32FC1);

		for( int l_i=0; l_i<ap_mat->rows; ++l_i )
		{
			if( CV_MAT_ELEM(*lp_w,float,l_i,l_i) != 0.0 )
			{
				CV_MAT_ELEM(*lp_w,float,l_i,l_i) = 1.0/CV_MAT_ELEM(*lp_w,float,l_i,l_i);
			}
		}
		cvGEMM(lp_v,lp_w,1,NULL,0.0,lp_t,CV_GEMM_B_T);
		cvGEMM(lp_t,lp_u,1,NULL,0.0,lp_r,CV_GEMM_B_T);

		cvReleaseMat(&lp_u);
		cvReleaseMat(&lp_w);
		cvReleaseMat(&lp_v);
		cvReleaseMat(&lp_t);

		return lp_r;
	}
}

/***************************************************************************/

void cv::cv_add_noise(	IplImage * ap_image,
						float a_var )
{
	for( int l_r=0; l_r<ap_image->height; ++l_r )
	{
		for( int l_c=0; l_c<ap_image->width; ++l_c )
		{
			CV_IMAGE_ELEM(ap_image,float,l_r,l_c) += 
			rand()/(RAND_MAX+0.0)*a_var*2-a_var;
		}
	}
}

/****************************************************************************/

void cv::cv_add_illumination(	IplImage * ap_image,
								float a_factor,
								float a_shift )
{
	if( fabs(a_factor) > 0.5 )
	{
		printf("cv_add_illumination: a_factor is maybe too high!");
	}
	float l_factor = rand()/(RAND_MAX+0.0)*a_factor*2-a_factor+1.0;
	float l_shift  = rand()/(RAND_MAX+0.0)*a_shift*2-a_shift;

	for( int l_r=0; l_r<ap_image->height; ++l_r )
	{
		for( int l_c=0; l_c<ap_image->width; ++l_c )
		{
			CV_IMAGE_ELEM(ap_image,float,l_r,l_c) *= l_factor;
			CV_IMAGE_ELEM(ap_image,float,l_r,l_c) += l_shift; 
		}
	}

}

/****************************************************************************/

void cv::cv_homogenize( CvMat * ap_points )
{
	if( ap_points == NULL )
	{
		printf("homogenize: ap_points is NULL!");
		return;
	}
#ifdef IPP_INCLUDED

	if( ap_points->rows == 3 )
	{
		int l_length	= ap_points->cols;
		float * lp_cur1	= &CV_MAT_ELEM(*ap_points,float,0,0);
		float * lp_cur2	= &CV_MAT_ELEM(*ap_points,float,1,0);
		float * lp_div	= &CV_MAT_ELEM(*ap_points,float,2,0);

		ippsDiv_32f(lp_div,lp_cur1,lp_cur1,l_length);
		ippsDiv_32f(lp_div,lp_cur2,lp_cur2,l_length);
		ippsSet_32f(1.0,lp_div,l_length);
	}
	else
	{
		int l_length	= ap_points->cols;
		float * lp_div	= &CV_MAT_ELEM(*ap_points,float,ap_points->rows-1,0);

		for( int l_i=0; l_i<ap_points->rows-1; ++l_i )
		{
			float * lp_cur = &CV_MAT_ELEM(*ap_points,float,l_i,0);

			ippsDiv_32f(lp_div,lp_cur,lp_cur,l_length);
		}
		ippsSet_32f(1.0,lp_div,l_length);
	}
#else
	for( int l_r=0; l_r<ap_points->rows; ++l_r )
	{
		for( int l_c=0; l_c<ap_points->cols; ++l_c )
		{
			CV_MAT_ELEM(*ap_points,float,l_r,l_c) /= 
			CV_MAT_ELEM(*ap_points,float,ap_points->rows-1,l_c);
		}
	}
#endif
}

/****************************************************************************/

void cv::cv_draw_points(	IplImage * ap_image,
							CvMat * ap_points,
							int a_radius,
							int a_r,
							int a_g,
							int a_b,
							int a_i)
{
	if( ap_points == NULL )
	{
		printf("draw_points: ap_points is NULL!");
		return;
	}
	if( ap_image == NULL )
	{
		printf("draw_points: ap_image is NULL!");
		return;
	}
	if( ap_image->imageData == NULL )
	{
		printf("draw_points: imageData is NULL!");
		return;
	}
	if( a_i == -1 )
	{
		a_i = ap_points->cols;
	}
	else
	{
		if( a_i >= ap_points->cols )
		{
			a_i = ap_points->cols;
		}
	}
	for( int l_i=0; l_i<a_i; ++l_i )
	{
		cvCircle(	ap_image, 
					cvPoint(CV_MAT_ELEM(*ap_points,float,0,l_i),
							CV_MAT_ELEM(*ap_points,float,1,l_i)),
					a_radius,
					CV_RGB(a_r,a_g,a_b),
					1 );
	}
}

/****************************************************************************/

void cv::cv_draw_poly(	IplImage * ap_image,
						CvMat * ap_points,
						int a_thickness,
						int a_r,
						int a_g,
						int a_b )
{
	if( ap_points == NULL )
	{
		printf("draw_poly: ap_points is NULL!");
		return;
	}
	if( ap_image == NULL )
	{
		printf("draw_poly: ap_image is NULL!");
		return;
	}
	if( ap_image->imageData == NULL )
	{
		printf("draw_poly: imageData is NULL!");
		return;
	}
	for( int l_i=0; l_i<ap_points->cols-1; ++l_i )
	{
		cvLine(	ap_image,	
				cvPoint(CV_MAT_ELEM(*ap_points,float,0,l_i),CV_MAT_ELEM(*ap_points,float,1,l_i)), 
				cvPoint(CV_MAT_ELEM(*ap_points,float,0,l_i+1),CV_MAT_ELEM(*ap_points,float,1,l_i+1)),
				CV_RGB(a_r,a_g,a_b),
				a_thickness );
	}
	cvLine(	ap_image,	
			cvPoint(CV_MAT_ELEM(*ap_points,float,0,ap_points->cols-1),CV_MAT_ELEM(*ap_points,float,1,ap_points->cols-1)), 
			cvPoint(CV_MAT_ELEM(*ap_points,float,0,0),CV_MAT_ELEM(*ap_points,float,1,0)),
			CV_RGB(a_r,a_g,a_b),
			a_thickness );
}

/****************************************************************************/

void cv::cv_print( CvMat * ap_mat )
{
	if( ap_mat == NULL )
	{
		printf("print: ap_mat is NULL!");
		return;
	}
	printf("rows: %d cols: %d \n",ap_mat->rows,ap_mat->cols);

	for( int l_r=0; l_r<ap_mat->rows; ++l_r )
	{
		for( int l_c=0; l_c<ap_mat->cols; ++l_c )
		{
			std::cerr << CV_MAT_ELEM(*ap_mat,float,l_r,l_c) << " , ";
		}
		std::cerr << std::endl;
	}
}

/****************************************************************************/

void cv::cv_print( IplImage * ap_img )
{
	if( ap_img == NULL )
	{
		printf("print: ap_img is NULL!");
		return;
	}
	printf("height: %d width: %d \n",ap_img->height,ap_img->width);

	for( int l_r=0; l_r<ap_img->height; ++l_r )
	{
		for( int l_c=0; l_c<ap_img->width; ++l_c )
		{
			std::cerr << CV_IMAGE_ELEM(ap_img,float,l_r,l_c) << " , ";
		}
		std::cerr << std::endl;
	}
}

/****************************************************************************/

void cv::cv_print_real( IplImage * ap_mat, int a_borderx, int a_bordery )
{
	if( ap_mat == NULL )
	{
		printf("print: ap_mat is NULL!");
		return;
	}
	printf("height: %d width: %d \n",ap_mat->height,ap_mat->width);
	
	if( a_borderx == -1 )
	{
		a_borderx = ap_mat->width;
	}
	if( a_bordery == -1 )
	{
		a_bordery = ap_mat->height;
	}
	for( int l_r=0; l_r<a_bordery; ++l_r )
	{
		for( int l_c=0; l_c<a_borderx; ++l_c )
		{
			std::cerr << CV_IMAGE_ELEM(ap_mat,float,l_r,l_c) << " , ";
		}
		std::cerr << std::endl;
	}
}


/****************************************************************************/

void cv::cv_print_complex( IplImage * ap_mat, int a_borderx, int a_bordery )
{
	if( ap_mat == NULL )
	{
		printf("print: ap_mat is NULL!");
		return;
	}
	printf("height: %d width: %d \n",ap_mat->height,ap_mat->width);

	if( a_borderx == -1 )
	{
		a_borderx = ap_mat->width;
	}
	if( a_bordery == -1 )
	{
		a_bordery = ap_mat->height;
	}
	for( int l_r=0; l_r<a_bordery; ++l_r )
	{
		for( int l_c=0; l_c<a_borderx; l_c+=2 )
		{
			std::cerr << CV_IMAGE_ELEM(ap_mat,float,l_r,l_c+0) << " i" << CV_IMAGE_ELEM(ap_mat,float,l_r,l_c+1) << " , ";
		}
		std::cerr << std::endl;
	}
}

/****************************************************************************/

void cv::cv_create_window( std::string a_window_name )
{
	cvNamedWindow(a_window_name.c_str(),1);
}

/****************************************************************************/

void cv::cv_show_image( IplImage * ap_image )
{
	if( ap_image == NULL )
	{
		printf("show_image: ap_image is NULL!");
		return;
	}
	if( ap_image->imageData == NULL )
	{
		printf("show_image: imageData is NULL!");
		return;
	}
	cvNamedWindow("static_image",1);
	IplImage * lp_image = cvCreateImage(cvGetSize(ap_image),IPL_DEPTH_8U,ap_image->nChannels);
	cvConvert(ap_image,lp_image);
	std::cerr << "press esc to proceed..." << std::endl;
	cvShowImage("static_image", lp_image );
	cvWaitKey(-1);
	cvReleaseImage(&lp_image);
}

/****************************************************************************/

void cv::cv_show_points(	IplImage * ap_image,
							CvMat * ap_points,
							int a_radius,
							int a_r,
							int a_g,
							int a_b )
{
	if( ap_points == NULL )
	{
		printf("show_points: ap_points is NULL!");
		return;
	}
	if( ap_image == NULL )
	{
		printf("show_points: ap_image is NULL!");
		return;
	}
	if( ap_image->imageData == NULL )
	{
		printf("show_points: imageData is NULL!");
		return;
	}
	IplImage * lp_image = cvCreateImage(cvGetSize(ap_image),
										IPL_DEPTH_32F,
										ap_image->nChannels);
	cvCopy(ap_image,lp_image);
	cv_draw_points(	lp_image,
					ap_points,
					a_radius,
					a_r,
					a_g,
					a_b);

	cv_show_image(lp_image);
	cvReleaseImage(&lp_image);
}
/****************************************************************************/

void cv::cv_show_image( IplImage * ap_image, std::string a_window_name )
{
	if( ap_image == NULL )
	{
		printf("show_image: image is NULL!");
		return;
	}
	
	if( ap_image->imageData == NULL )
	{
		printf("show_image: imageData is NULL!");
		return;
	}
	if( cvGetElemType(ap_image) == CV_8UC1 || cvGetElemType(ap_image) == CV_8UC3 )
	{
		cvShowImage(a_window_name.c_str(),ap_image);
	}
	else
	{
		IplImage * lp_image = cvCreateImage(cvGetSize(ap_image),IPL_DEPTH_8U,ap_image->nChannels);
		cvConvert(ap_image,lp_image);
		cvShowImage(a_window_name.c_str(),lp_image);
		//cvWaitKey(-1);
		cvReleaseImage(&lp_image);
	}
}

/****************************************************************************/

IplImage * cv::cv_load_image( std::string a_name )
{
	IplImage * lp_image1 = cvLoadImage(a_name.c_str(),-1);
	IplImage * lp_image2 = cvCreateImage(	cvGetSize(lp_image1),
											IPL_DEPTH_32F,
											lp_image1->nChannels);
	cvConvert(lp_image1,lp_image2);

	cvReleaseImage(&lp_image1);
	return lp_image2;
}

/****************************************************************************/

IplImage * cv::cv_load_image_uchar( std::string a_name )
{
	IplImage * lp_image1 = cvLoadImage(a_name.c_str(),-1);
	IplImage * lp_image2 = cvCreateImage(	cvGetSize(lp_image1),
											IPL_DEPTH_8U,
											lp_image1->nChannels);
	cvConvert(lp_image1,lp_image2);

	cvReleaseImage(&lp_image1);
	return lp_image2;
}

/****************************************************************************/

void cv::cv_save_image( std::string a_name, IplImage * ap_image)
{
	IplImage * lp_image = cvCreateImage(cvGetSize(ap_image),IPL_DEPTH_8U,ap_image->nChannels);
	cvCvtScale(ap_image,lp_image);
	cvSaveImage(a_name.c_str(),lp_image);
	cvReleaseImage(&lp_image);
}

/****************************************************************************/

int cv::cv_mouse::m_event = -1;
int cv::cv_mouse::m_x     = -1;
int cv::cv_mouse::m_y     = -1;

/***************************************************************************/

std::ofstream & cv::cv_write( std::ofstream & a_os, CvMat * ap_mat )
{
	int l_row = ap_mat->rows;
	int l_col = ap_mat->cols;

	a_os.write((char*)&l_row,sizeof(l_row));
	a_os.write((char*)&l_col,sizeof(l_col));

	for( int l_i=0; l_i<l_row*l_col; ++l_i )
	{
		a_os.write((char*)&ap_mat->data.fl[l_i],sizeof(ap_mat->data.fl[l_i]));
	}
	return a_os;
}

/***************************************************************************/

std::ofstream & cv::cv_write( std::ofstream & a_os, IplImage * ap_img )
{
	int l_row = ap_img->height;
	int l_col = ap_img->width;

	a_os.write((char*)&l_row,sizeof(l_row));
	a_os.write((char*)&l_col,sizeof(l_col));
	a_os.write((char*)&ap_img->depth,sizeof(ap_img->depth));
	a_os.write((char*)&ap_img->nChannels,sizeof(ap_img->nChannels));

	for( int l_i=0; l_i<l_row*l_col*ap_img->nChannels*ap_img->depth/8; ++l_i )
	{
		a_os.write((char*)&(ap_img->imageData)[l_i],sizeof((ap_img->imageData)[l_i]));
	}
	return a_os;
}

/***************************************************************************/

CvMat * cv::cv_read( std::ifstream & a_is )
{
	int l_row;
	int l_col;

	a_is.read((char*)&l_row,sizeof(l_row));
	a_is.read((char*)&l_col,sizeof(l_col));

	CvMat * lp_mat = cvCreateMat(l_row,l_col,CV_32FC1);

	for( int l_i=0; l_i<l_row*l_col; ++l_i )
	{
		a_is.read((char*)&lp_mat->data.fl[l_i],sizeof(lp_mat->data.fl[l_i]));
	}
	return lp_mat;
}

/***************************************************************************/

IplImage * cv::cv_read_img( std::ifstream & a_is )
{
	int l_row;
	int l_col;
	int l_depth;
	int l_channels;

	a_is.read((char*)&l_row,sizeof(l_row));
	a_is.read((char*)&l_col,sizeof(l_col));
	a_is.read((char*)&l_depth,sizeof(l_depth));
	a_is.read((char*)&l_channels,sizeof(l_channels));

	IplImage * lp_img = cvCreateImage(cvSize(l_col,l_row),l_depth,l_channels);

	for( int l_i=0; l_i<l_row*l_col*l_channels*l_depth/8; ++l_i )
	{
		a_is.read((char*)&(lp_img->imageData)[l_i],sizeof((lp_img->imageData)[l_i]));
	}
	return lp_img;
}

/***************************************************************************/

void cv::cv_save_mat( std::string a_name, CvMat * ap_mat )
{
	std::ofstream l_file(	a_name.c_str(),
							std::ofstream::out |
							std::ofstream::binary );
	
	if( l_file.fail() == true ) 
	{
		printf("cv_save_mat: could not open file for writing!\n");
		return; 
	}
	cv::cv_write(l_file,ap_mat);	

	l_file.close();
}

/***************************************************************************/

CvMat * cv::cv_load_mat( std::string a_name )
{
	std::ifstream l_file(	a_name.c_str(),
							std::ifstream::in |
							std::ifstream::binary );
	
	if( l_file.fail() == true ) 
	{
		printf("cv_load_mat: could not open file for reading!\n");
		return NULL;
	}
	CvMat * lp_mat = cv_read(l_file);	

	l_file.close();

	return lp_mat;
}

/***************************************************************************/

bool cv::cv_homography_heuristic( CvMat * ap_quad, float a_fac, float a_pix )
{
	float l_x1 = CV_MAT_ELEM(*ap_quad,float,0,0);
	float l_y1 = CV_MAT_ELEM(*ap_quad,float,1,0);
	float l_x2 = CV_MAT_ELEM(*ap_quad,float,0,1);
	float l_y2 = CV_MAT_ELEM(*ap_quad,float,1,1);
	float l_x3 = CV_MAT_ELEM(*ap_quad,float,0,2);
	float l_y3 = CV_MAT_ELEM(*ap_quad,float,1,2);
	float l_x4 = CV_MAT_ELEM(*ap_quad,float,0,3);
	float l_y4 = CV_MAT_ELEM(*ap_quad,float,1,3);

	float l_distance1 = sqrt(SQR(l_x1-l_x2)+SQR(l_y1-l_y2));
	float l_distance2 = sqrt(SQR(l_x2-l_x3)+SQR(l_y2-l_y3));
	float l_distance3 = sqrt(SQR(l_x3-l_x4)+SQR(l_y3-l_y4));
	float l_distance4 = sqrt(SQR(l_x4-l_x1)+SQR(l_y4-l_y1));

	if( l_distance1/l_distance3 < a_fac || l_distance3/l_distance1 < a_fac ||
		l_distance2/l_distance4 < a_fac || l_distance4/l_distance2 < a_fac )
	{
		//std::cerr << "heuristic: " << l_distance1/l_distance3 << "," << l_distance2/l_distance4 << std::endl;
		return false;
	}
	if( l_distance1 < a_pix || l_distance2 < a_pix || l_distance3 < a_pix || l_distance4 < a_pix )
	{
		//std::cerr << "heuristic: " << l_distance1 << "," << l_distance2 << "," << l_distance3 << "," << l_distance4 << std::endl;
		return false;
	}
	return true;
}

/********************************* END OF FILE ******************************/





/*
CvMat * cv_hyper::create_rand_trans(	CvMat * ap_k,
										double a_deg )
{
	CvMat * lp_k = cvCreateMat(3,3,CV_32FC1);
	CvMat * lp_a = cvCreateMat(3,1,CV_32FC1);
	CvMat * lp_n = cvCreateMat(3,1,CV_32FC1);
	CvMat * lp_d = cvCreateMat(3,1,CV_32FC1);
	CvMat * lp_t = cvCreateMat(3,1,CV_32FC1);
	CvMat * lp_t0 = cvCreateMat(3,1,CV_32FC1);
	CvMat * lp_t1 = cvCreateMat(3,1,CV_32FC1);
	CvMat * lp_r1 = cvCreateMat(3,3,CV_32FC1);
	CvMat * lp_r2 = cvCreateMat(3,3,CV_32FC1);
	CvMat * lp_id = cvCreateMat(3,3,CV_32FC1);
	CvMat * lp_inv  = cvCreateMat(3,3,CV_32FC1);
	CvMat * lp_rot0 = cvCreateMat(3,3,CV_32FC1);
	CvMat * lp_rot1 = cvCreateMat(3,3,CV_32FC1);
	CvMat * lp_rot2 = cvCreateMat(3,3,CV_32FC1);
	CvMat * lp_hom0 = cvCreateMat(3,3,CV_32FC1);
	CvMat * lp_hom1 = cvCreateMat(3,3,CV_32FC1);
	CvMat * lp_hom2 = cvCreateMat(3,3,CV_32FC1);
	
	cvCopy(ap_k,lp_k);

	CV_MAT_ELEM(*lp_k,float,0,2) = 0;
	CV_MAT_ELEM(*lp_k,float,1,2) = 0;

	do
	{
		CV_MAT_ELEM(*lp_a,float,0,0) = rand()/(RAND_MAX+0.0)-0.5;
		CV_MAT_ELEM(*lp_a,float,1,0) = rand()/(RAND_MAX+0.0)-0.5;
		CV_MAT_ELEM(*lp_a,float,2,0) = -1;
	}
	while(	CV_MAT_ELEM(*lp_a,float,0,0) ==  0 &&
		    CV_MAT_ELEM(*lp_a,float,1,0) ==  0 &&
			CV_MAT_ELEM(*lp_a,float,2,0) == -1 );

	CV_MAT_ELEM(*lp_n,float,0,0) = 0;
	CV_MAT_ELEM(*lp_n,float,1,0) = 0;
	CV_MAT_ELEM(*lp_n,float,2,0) = -1;

	cvCrossProduct(lp_a,lp_n,lp_d);
	cvNormalize(lp_d,lp_d);

	CV_MAT_ELEM(*lp_r1,float,0,0) = 0;
	CV_MAT_ELEM(*lp_r1,float,0,1) = -CV_MAT_ELEM(*lp_d,float,2,0);
	CV_MAT_ELEM(*lp_r1,float,0,2) = +CV_MAT_ELEM(*lp_d,float,1,0);
	CV_MAT_ELEM(*lp_r1,float,1,0) = +CV_MAT_ELEM(*lp_d,float,2,0);
	CV_MAT_ELEM(*lp_r1,float,1,1) = 0;
	CV_MAT_ELEM(*lp_r1,float,1,2) = -CV_MAT_ELEM(*lp_d,float,0,0);
	CV_MAT_ELEM(*lp_r1,float,2,0) = -CV_MAT_ELEM(*lp_d,float,1,0);
	CV_MAT_ELEM(*lp_r1,float,2,1) = +CV_MAT_ELEM(*lp_d,float,0,0);
	CV_MAT_ELEM(*lp_r1,float,2,2) = 0;

	CV_MAT_ELEM(*lp_r2,float,0,0) = 0;
	CV_MAT_ELEM(*lp_r2,float,0,1) = -CV_MAT_ELEM(*lp_n,float,2,0);
	CV_MAT_ELEM(*lp_r2,float,0,2) = +CV_MAT_ELEM(*lp_n,float,1,0);
	CV_MAT_ELEM(*lp_r2,float,1,0) = +CV_MAT_ELEM(*lp_n,float,2,0);
	CV_MAT_ELEM(*lp_r2,float,1,1) = 0;
	CV_MAT_ELEM(*lp_r2,float,1,2) = -CV_MAT_ELEM(*lp_n,float,0,0);
	CV_MAT_ELEM(*lp_r2,float,2,0) = -CV_MAT_ELEM(*lp_n,float,1,0);
	CV_MAT_ELEM(*lp_r2,float,2,1) = +CV_MAT_ELEM(*lp_n,float,0,0);
	CV_MAT_ELEM(*lp_r2,float,2,2) = 0;

	CV_MAT_ELEM(*lp_id,float,0,0) = 1;
	CV_MAT_ELEM(*lp_id,float,0,1) = 0;
	CV_MAT_ELEM(*lp_id,float,0,2) = 0;
	CV_MAT_ELEM(*lp_id,float,1,0) = 0;
	CV_MAT_ELEM(*lp_id,float,1,1) = 1;
	CV_MAT_ELEM(*lp_id,float,1,2) = 0;
	CV_MAT_ELEM(*lp_id,float,2,0) = 0;
	CV_MAT_ELEM(*lp_id,float,2,1) = 0;
	CV_MAT_ELEM(*lp_id,float,2,2) = 1;

	double l_angle1 = (a_deg+(rand()/(RAND_MAX+0.0)*10-5))*cv_pi/180.0;
	double l_angle2 = (rand()/(RAND_MAX+1.0))*2*cv_pi;

	cvGEMM(lp_r1,lp_r1,1.0-cos(l_angle1),lp_r1,sin(l_angle1),lp_rot1);
	cvGEMM(lp_r2,lp_r2,1.0-cos(l_angle2),lp_r2,sin(l_angle2),lp_rot2);

	cvAdd(lp_rot1,lp_id,lp_rot1);
	cvAdd(lp_rot2,lp_id,lp_rot2);

	cvMatMul(lp_rot2,lp_rot1,lp_rot0);

	CV_MAT_ELEM(*lp_t0,float,0,0) = 0;
	CV_MAT_ELEM(*lp_t0,float,1,0) = 0;
	CV_MAT_ELEM(*lp_t0,float,2,0) = 10e10;

	CV_MAT_ELEM(*lp_t1,float,0,0) = 0;
	CV_MAT_ELEM(*lp_t1,float,1,0) = 0;
	CV_MAT_ELEM(*lp_t1,float,2,0) = 1;

	cvGEMM(lp_rot0,lp_t0,-1.0,lp_t0,1.0,lp_t);
	cvAdd(lp_t,lp_t1,lp_t);

	cvGEMM(lp_t,lp_n,1.0/10e10,lp_rot0,1.0,lp_hom0,CV_GEMM_B_T);
	cvInvert(lp_k,lp_inv,CV_SVD);
	cvMatMul(lp_hom0,lp_inv,lp_hom1);
	cvMatMul(lp_k,lp_hom1,lp_hom2);

	cvReleaseMat(&lp_k);
	cvReleaseMat(&lp_a);
	cvReleaseMat(&lp_n);
	cvReleaseMat(&lp_d);
	cvReleaseMat(&lp_t);
	cvReleaseMat(&lp_t0);
	cvReleaseMat(&lp_t1);
	cvReleaseMat(&lp_r1);
	cvReleaseMat(&lp_r2);
	cvReleaseMat(&lp_id);
	cvReleaseMat(&lp_inv);
	cvReleaseMat(&lp_rot0);
	cvReleaseMat(&lp_rot1);
	cvReleaseMat(&lp_rot2);
	cvReleaseMat(&lp_hom0);
	cvReleaseMat(&lp_hom1);
	cvReleaseMat(&lp_hom2);

	return lp_hom2;
}
*/