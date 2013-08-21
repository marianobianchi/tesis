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
// cv_esm : cv_esm.cc														//
//																			//
// Authors: Stefan Hinterstoisser 2010										//
// Version: 1.0																//
//																			//
//////////////////////////////////////////////////////////////////////////////

/******************************* includes **********************************/

#include "cv_esm.h"

/******************************** defines **********************************/

/******************************* namespaces ********************************/

using namespace cv;

/****************************** constructors *******************************/

cv_esm::cv_esm( void ):
m_delete_flag(false),
m_mask_flag(false),
m_height(0),
m_width(0)
{
	mp_ref = NULL;
}

/******************************* destructor ********************************/


cv_esm::~cv_esm()
{
	if( m_delete_flag == true && m_mask_flag == false ) 
		FreeTrack( &m_T );

	cvReleaseImage(&mp_ref);
}

/***************************************************************************/


bool cv_esm::learn( IplImage * ap_image,
					int a_row,
					int a_col,
					int a_height,
					int a_width )
{
	if( ap_image == NULL )
	{
		printf("cv_esm: ap_image is NULL!");
		return false;
	}
	if( ap_image->imageData == NULL )
	{
		printf("cv_esm: ap_image->imageData is NULL!");
		return false;
	}
	if( cvGetElemType(ap_image) != CV_32FC1 )
	{
		printf("cv_esm: type is not 32FC1!");
		return false;
	}
	if( a_height <= 0 || a_width <= 0 )
	{
		printf("cv_esm: width or height are smaller 0!");
		return false;
	}
	if( a_row < 0 ||
		a_col < 0 ||
		a_row+a_height >= ap_image->height ||
		a_col+a_width  >= ap_image->width )
	{
		printf("cv_esm: rows and cols are not appropriate!");
		return false;
	}
	if( m_delete_flag ) 
		FreeTrack( &m_T );

	if( mp_ref != NULL )
		cvReleaseImage(&mp_ref);

	imageStruct l_I; 

	m_width  = a_width;
	m_height = a_height;

	mp_rec = cvCreateMat(3,4,CV_32FC1);
	
	CV_MAT_ELEM(*mp_rec,float,0,0) = 0;
	CV_MAT_ELEM(*mp_rec,float,1,0) = 0;
	CV_MAT_ELEM(*mp_rec,float,0,1) = m_width-1;
	CV_MAT_ELEM(*mp_rec,float,1,1) = 0;
	CV_MAT_ELEM(*mp_rec,float,0,2) = m_width-1;
	CV_MAT_ELEM(*mp_rec,float,1,2) = m_height-1;
	CV_MAT_ELEM(*mp_rec,float,0,3) = 0.0;
	CV_MAT_ELEM(*mp_rec,float,1,3) = m_height-1;
	CV_MAT_ELEM(*mp_rec,float,2,0) = 1;
	CV_MAT_ELEM(*mp_rec,float,2,1) = 1;
	CV_MAT_ELEM(*mp_rec,float,2,2) = 1;
	CV_MAT_ELEM(*mp_rec,float,2,3) = 1;

	l_I.data = (float *)(ap_image->imageData);
	l_I.rows = ap_image->height;
	l_I.cols = ap_image->widthStep/sizeof(float);
	l_I.clrs = 1;

	if( MallTrack( &m_T, &l_I, a_col, a_row, m_width, m_height, 10, 10 ) )
	{
		printf("cv_esm: esm could not be learned!");
		return false;
	}
	m_delete_flag = true;
	m_mask_flag = false;

	mp_ref = get_ref_pattern();
	cv::cv_normalize_mean_std((float*)mp_ref->imageData,mp_ref->widthStep/sizeof(float)*mp_ref->height);

	return true;
}
	
/***************************************************************************/

CvMat * cv_esm::track(	IplImage * ap_image, 
						CvMat * ap_rec,
						int a_max_iter,
						int a_prec )
{
	if( ap_rec == NULL )
	{
		printf("cv_esm: ap_rec is NULL!");
		return NULL;
	}
	if( ap_image == NULL )
	{
		printf("cv_esm: ap_image is NULL!");
		return NULL;
	}
	if( ap_image->imageData == NULL )
	{
		printf("cv_esm: ap_image->imageData is NULL!");
		return NULL;
	}
	if( cvGetElemType(ap_image) != CV_32FC1 )
	{
		printf("cv_esm: type is not 32FC1!");
		return NULL;
	}
	imageStruct l_I; 

	l_I.data = (float *)(ap_image->imageData);
	l_I.rows = ap_image->height;
	l_I.cols = ap_image->widthStep/sizeof(float);
	l_I.clrs = 1;
	
	m_T.miter = a_max_iter;
	m_T.mprec = a_prec;

	CvPoint2D32f * lp_src = new CvPoint2D32f[4];
	CvPoint2D32f * lp_dst = new CvPoint2D32f[4];

	lp_src[0].x = CV_MAT_ELEM(*ap_rec,float,0,0);
	lp_src[0].y = CV_MAT_ELEM(*ap_rec,float,1,0);
	lp_src[1].x = CV_MAT_ELEM(*ap_rec,float,0,1);
	lp_src[1].y = CV_MAT_ELEM(*ap_rec,float,1,1);
	lp_src[2].x = CV_MAT_ELEM(*ap_rec,float,0,2);
	lp_src[2].y = CV_MAT_ELEM(*ap_rec,float,1,2);
	lp_src[3].x = CV_MAT_ELEM(*ap_rec,float,0,3);
	lp_src[3].y = CV_MAT_ELEM(*ap_rec,float,1,3);

	lp_dst[0].x = CV_MAT_ELEM(*mp_rec,float,0,0);
	lp_dst[0].y = CV_MAT_ELEM(*mp_rec,float,1,0);
	lp_dst[1].x = CV_MAT_ELEM(*mp_rec,float,0,1);
	lp_dst[1].y = CV_MAT_ELEM(*mp_rec,float,1,1);
	lp_dst[2].x = CV_MAT_ELEM(*mp_rec,float,0,2);
	lp_dst[2].y = CV_MAT_ELEM(*mp_rec,float,1,2);
	lp_dst[3].x = CV_MAT_ELEM(*mp_rec,float,0,3);
	lp_dst[3].y = CV_MAT_ELEM(*mp_rec,float,1,3);

	CvMat * lp_pre_homo = cvCreateMat(3,3,CV_32FC1);

	cvGetPerspectiveTransform(lp_dst,lp_src,lp_pre_homo);
	
	delete[] lp_src;
	delete[] lp_dst;

	if( lp_pre_homo == NULL )
	{
		return NULL;
	}
	m_T.homog[0] = CV_MAT_ELEM(*lp_pre_homo,float,0,0);
	m_T.homog[1] = CV_MAT_ELEM(*lp_pre_homo,float,0,1);
	m_T.homog[2] = CV_MAT_ELEM(*lp_pre_homo,float,0,2);
	m_T.homog[3] = CV_MAT_ELEM(*lp_pre_homo,float,1,0);
	m_T.homog[4] = CV_MAT_ELEM(*lp_pre_homo,float,1,1);
	m_T.homog[5] = CV_MAT_ELEM(*lp_pre_homo,float,1,2);
	m_T.homog[6] = CV_MAT_ELEM(*lp_pre_homo,float,2,0);
	m_T.homog[7] = CV_MAT_ELEM(*lp_pre_homo,float,2,1);
	m_T.homog[8] = CV_MAT_ELEM(*lp_pre_homo,float,2,2);
	
	if( MakeTrack( &m_T, &l_I ) )
	{
		return NULL;
	}
	CvMat * lp_new_homo = cvCreateMat(3,3,CV_32FC1);
	CV_MAT_ELEM(*lp_new_homo,float,0,0) = m_T.homog[0];
	CV_MAT_ELEM(*lp_new_homo,float,0,1) = m_T.homog[1];
	CV_MAT_ELEM(*lp_new_homo,float,0,2) = m_T.homog[2];
	CV_MAT_ELEM(*lp_new_homo,float,1,0) = m_T.homog[3];
	CV_MAT_ELEM(*lp_new_homo,float,1,1) = m_T.homog[4];
	CV_MAT_ELEM(*lp_new_homo,float,1,2) = m_T.homog[5];
	CV_MAT_ELEM(*lp_new_homo,float,2,0) = m_T.homog[6];
	CV_MAT_ELEM(*lp_new_homo,float,2,1) = m_T.homog[7];
	CV_MAT_ELEM(*lp_new_homo,float,2,2) = m_T.homog[8];

	CvMat * lp_result = cvCreateMat(3,4,CV_32FC1);
	cvMatMul(lp_new_homo,mp_rec,lp_result);

	cv_homogenize(lp_result);

	cvReleaseMat(&lp_new_homo);
	cvReleaseMat(&lp_pre_homo);

	return lp_result;
}

/***************************************************************************/

IplImage * cv_esm::get_cur_pattern( void )
{
	imageStruct * lp_I = GetPatc( &m_T );
	
	IplImage * lp_image = cvCreateImage(cvSize(lp_I->cols,lp_I->rows),IPL_DEPTH_32F,1);
	IplImage * lp_current = cvCreateImageHeader(cvSize(lp_I->cols,lp_I->rows),IPL_DEPTH_32F,1);

	lp_current->imageData = (char *)lp_I->data;
	lp_current->widthStep = lp_I->cols*sizeof(float);

	cvSet(lp_image,cvRealScalar(0));
	cvCopy(lp_current,lp_image);

	cvReleaseImageHeader(&lp_current);
	//do not free the imageStruct here...
	return lp_image;
}

/***************************************************************************/


IplImage * cv_esm::get_ref_pattern( void )
{
	imageStruct * lp_I = GetPatr( &m_T );
	
	IplImage * lp_image = cvCreateImage(cvSize(lp_I->cols,lp_I->rows),IPL_DEPTH_32F,1);
	IplImage * lp_current = cvCreateImageHeader(cvSize(lp_I->cols,lp_I->rows),IPL_DEPTH_32F,1);

	lp_current->imageData = (char *)lp_I->data;
	lp_current->widthStep = lp_I->cols*sizeof(float);

	cvSet(lp_image,cvRealScalar(0));
	cvCopy(lp_current,lp_image);

	cvReleaseImageHeader(&lp_current);
	//do not free the imageStruct here...
	return lp_image;
}

/***************************************************************************/

float cv_esm::get_ncc( void )
{
	IplImage * lp_image = this->get_cur_pattern();

	int l_size = lp_image->width*lp_image->height;

	cv::cv_normalize_mean_std((float*)lp_image->imageData,lp_image->widthStep/sizeof(float)*lp_image->height);
	
	float l_val = cv::cv_dot_product((float*)lp_image->imageData,(float*)mp_ref->imageData,l_size);
	
	return l_val/l_size;
}

/***************************************************************************/

CvMat * cv_esm::get_H( void )
{
	CvMat * lp_homo = cvCreateMat(3,3,CV_32FC1);

	CV_MAT_ELEM(*lp_homo,float,0,0) = m_T.homog[0];
	CV_MAT_ELEM(*lp_homo,float,0,1) = m_T.homog[1];
	CV_MAT_ELEM(*lp_homo,float,0,2) = m_T.homog[2];
	CV_MAT_ELEM(*lp_homo,float,1,0) = m_T.homog[3];
	CV_MAT_ELEM(*lp_homo,float,1,1) = m_T.homog[4];
	CV_MAT_ELEM(*lp_homo,float,1,2) = m_T.homog[5];
	CV_MAT_ELEM(*lp_homo,float,2,0) = m_T.homog[6];
	CV_MAT_ELEM(*lp_homo,float,2,1) = m_T.homog[7];
	CV_MAT_ELEM(*lp_homo,float,2,2) = m_T.homog[8];

	return lp_homo;
}

/******************************* END OF FILE *******************************/
