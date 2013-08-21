//////////////////////////////////////////////////////////////////////////////
//                                                                          //
// cv_homography: cv_homography.cc											//
//                                                                          //
// Copyright (c) Stefan Hinterstoisser 2007                                 //
// Lehrstuhl fuer Informatik XVI                                            //
// Technische Universitaet Muenchen                                         //
// Version: 1.0                                                             //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////

/******************************* includes ***********************************/

#include "cv_homography.h"

/******************************** defines ***********************************/

/******************************* namespaces *********************************/

using namespace cv;

/****************************** constructors ********************************/

/******************************** destructor ********************************/

/****************************************************************************/

CvMat * cv_homography::compute(	CvMat * ap_dst_points,
								CvMat * ap_src_points )
{
	if( ap_dst_points == NULL )
	{
		printf("cv_homography: ap_dst_points is NULL!");
		return NULL;
	}
	if( ap_src_points == NULL )
	{
		printf("cv_homography: ap_src_points is NULL!");
		return NULL;
	}
	if( ap_dst_points->cols != ap_src_points->cols )
	{
		printf("cv_homography: point correspondences are not consistent!");
		return NULL;
	}
	if( ap_dst_points->cols < 4 )
	{
		printf("cv_homography: at least 4 correspondences are needed!");
		return NULL;
	}
	if( ap_dst_points->cols == 4 )
	{	
		return compute_p4_closed( ap_dst_points, ap_src_points );
	}
	return compute_pn( ap_dst_points, ap_src_points );;
}

/****************************************************************************/

CvMat * cv_homography::compute_p4(	CvMat * ap_dst_points,
									CvMat * ap_src_points )
{
	CvPoint2D32f * lp_src = new CvPoint2D32f[4];
	CvPoint2D32f * lp_dst = new CvPoint2D32f[4];

	lp_src[0].x = CV_MAT_ELEM(*ap_src_points,float,0,0);
	lp_src[0].y = CV_MAT_ELEM(*ap_src_points,float,1,0);
	lp_src[1].x = CV_MAT_ELEM(*ap_src_points,float,0,1);
	lp_src[1].y = CV_MAT_ELEM(*ap_src_points,float,1,1);
	lp_src[2].x = CV_MAT_ELEM(*ap_src_points,float,0,2);
	lp_src[2].y = CV_MAT_ELEM(*ap_src_points,float,1,2);
	lp_src[3].x = CV_MAT_ELEM(*ap_src_points,float,0,3);
	lp_src[3].y = CV_MAT_ELEM(*ap_src_points,float,1,3);

	lp_dst[0].x = CV_MAT_ELEM(*ap_dst_points,float,0,0);
	lp_dst[0].y = CV_MAT_ELEM(*ap_dst_points,float,1,0);
	lp_dst[1].x = CV_MAT_ELEM(*ap_dst_points,float,0,1);
	lp_dst[1].y = CV_MAT_ELEM(*ap_dst_points,float,1,1);
	lp_dst[2].x = CV_MAT_ELEM(*ap_dst_points,float,0,2);
	lp_dst[2].y = CV_MAT_ELEM(*ap_dst_points,float,1,2);
	lp_dst[3].x = CV_MAT_ELEM(*ap_dst_points,float,0,3);
	lp_dst[3].y = CV_MAT_ELEM(*ap_dst_points,float,1,3);

	CvMat * lp_homography = cvCreateMat(3,3,CV_32FC1);
	
	cvGetPerspectiveTransform(	lp_src,
								lp_dst,
								lp_homography );
	delete[] lp_src;
	delete[] lp_dst;

	return lp_homography;
}

/****************************************************************************/


static void homography_from_4pt(	const float * ap_x, 
									const float * ap_y,
									const float * ap_z,
									const float * ap_w,
									float * ap_cgret )
{
	float l_t1 = ap_x[0];
	float l_t2 = ap_z[0];
	float l_t4 = ap_y[1];
	float l_t5 = l_t1 *l_t2 *l_t4;
	float l_t6 = ap_w[1];
	float l_t7 = l_t1 *l_t6;
	float l_t8 = l_t2 *l_t7;
	float l_t9 = ap_z[1];
	float l_t10 = l_t1 *l_t9;
	float l_t11 = ap_y[0];
	float l_t14 = ap_x[1];
	float l_t15 = ap_w[0];
	float l_t16 = l_t14 *l_t15;
	float l_t18 = l_t16 *l_t11;
	float l_t20 = l_t15 *l_t11 *l_t9;
	float l_t21 = l_t15 *l_t4;
	float l_t24 = l_t15 *l_t9;
	float l_t25 = l_t2 *l_t4;
	float l_t26 = l_t6 *l_t2;
	float l_t27 = l_t6 *l_t11;
	float l_t28 = l_t9 *l_t11;
	float l_t30 = 0.1e1 / (-l_t24 +l_t21 -l_t25 +l_t26 -l_t27 +l_t28);
	float l_t32 = l_t1 *l_t15;
	float l_t35 = l_t14 *l_t11;
	float l_t41 = l_t4 *l_t1;
	float l_t42 = l_t6 *l_t41;
	float l_t43 = l_t14 *l_t2;
	float l_t46 = l_t16 *l_t9;
	float l_t48 = l_t14 *l_t9 *l_t11;
	float l_t51 = l_t4 *l_t6 *l_t2;
	float l_t55 = l_t6 *l_t14;
	ap_cgret[0] = -(-l_t5 +l_t8 +l_t10 *l_t11 -l_t11 *l_t7 -l_t16 *l_t2 +l_t18 -l_t20 +l_t21 *l_t2) *l_t30;
	ap_cgret[1] = (l_t5 -l_t8 -l_t32 *l_t4 +l_t32 *l_t9 +l_t18 -l_t2 *l_t35 +l_t27 *l_t2 -l_t20) *l_t30;
	ap_cgret[2] = l_t1;
	ap_cgret[3] = (-l_t9 *l_t7 +l_t42 +l_t43 *l_t4 -l_t16 *l_t4 +l_t46 -l_t48 +l_t27 *l_t9 -l_t51) *l_t30;
	ap_cgret[4] = (-l_t42 +l_t41 *l_t9 -l_t55 *l_t2 +l_t46 -l_t48 +l_t55 *l_t11 +l_t51 -l_t21 *l_t9) *l_t30;
	ap_cgret[5] = l_t14;
	ap_cgret[6] = (-l_t10 +l_t41 +l_t43 -l_t35 +l_t24 -l_t21 -l_t26 +l_t27) *l_t30;
	ap_cgret[7] = (-l_t7 +l_t10 +l_t16 -l_t43 +l_t27 -l_t28 -l_t21 +l_t25) *l_t30;
}

CvMat * cv_homography::compute_p4_closed(	CvMat * ap_dst_points,
											CvMat * ap_src_points )
{
	CvMat * lp_hom = cvCreateMat(3,3,CV_32FC1);

	const float l_a[] = {CV_MAT_ELEM(*ap_src_points,float,0,0), CV_MAT_ELEM(*ap_src_points,float,1,0)};
	const float l_b[] = {CV_MAT_ELEM(*ap_src_points,float,0,1), CV_MAT_ELEM(*ap_src_points,float,1,1)};
	const float l_c[] = {CV_MAT_ELEM(*ap_src_points,float,0,2), CV_MAT_ELEM(*ap_src_points,float,1,2)};
	const float l_d[] = {CV_MAT_ELEM(*ap_src_points,float,0,3), CV_MAT_ELEM(*ap_src_points,float,1,3)};
	const float l_x[] = {CV_MAT_ELEM(*ap_dst_points,float,0,0), CV_MAT_ELEM(*ap_dst_points,float,1,0)};
	const float l_y[] = {CV_MAT_ELEM(*ap_dst_points,float,0,1), CV_MAT_ELEM(*ap_dst_points,float,1,1)};
	const float l_z[] = {CV_MAT_ELEM(*ap_dst_points,float,0,2), CV_MAT_ELEM(*ap_dst_points,float,1,2)};
	const float l_w[] = {CV_MAT_ELEM(*ap_dst_points,float,0,3), CV_MAT_ELEM(*ap_dst_points,float,1,3)};
	
	float l_hr[3][3], l_hl[3][3];

	homography_from_4pt(l_a,l_b,l_c,l_d,&l_hr[0][0]);
	homography_from_4pt(l_x,l_y,l_z,l_w,&l_hl[0][0]);

	// the following code computes R = l_hl * inverse l_hr
	float t2 = l_hr[1][1]-l_hr[2][1]*l_hr[1][2];
	float t4 = l_hr[0][0]*l_hr[1][1];
	float t5 = l_hr[0][0]*l_hr[1][2];
	float t7 = l_hr[1][0]*l_hr[0][1];
	float t8 = l_hr[0][2]*l_hr[1][0];
	float t10 = l_hr[0][1]*l_hr[2][0];
	float t12 = l_hr[0][2]*l_hr[2][0];
	float t15 = 1/(t4-t5*l_hr[2][1]-t7+t8*l_hr[2][1]+t10*l_hr[1][2]-t12*l_hr[1][1]);
	float t18 = -l_hr[1][0]+l_hr[1][2]*l_hr[2][0];
	float t23 = -l_hr[1][0]*l_hr[2][1]+l_hr[1][1]*l_hr[2][0];
	float t28 = -l_hr[0][1]+l_hr[0][2]*l_hr[2][1];
	float t31 = l_hr[0][0]-t12;
	float t35 = l_hr[0][0]*l_hr[2][1]-t10;
	float t41 = -l_hr[0][1]*l_hr[1][2]+l_hr[0][2]*l_hr[1][1];
	float t44 = t5-t8;
	float t47 = t4-t7;
	float t48 = t2*t15;
	float t49 = t28*t15;
	float t50 = t41*t15;
	CV_MAT_ELEM(*lp_hom,float,0,0) = l_hl[0][0]*t48+l_hl[0][1]*(t18*t15)-l_hl[0][2]*(t23*t15);
	CV_MAT_ELEM(*lp_hom,float,0,1) = l_hl[0][0]*t49+l_hl[0][1]*(t31*t15)-l_hl[0][2]*(t35*t15);
	CV_MAT_ELEM(*lp_hom,float,0,2) = -l_hl[0][0]*t50-l_hl[0][1]*(t44*t15)+l_hl[0][2]*(t47*t15);
	CV_MAT_ELEM(*lp_hom,float,1,0) = l_hl[1][0]*t48+l_hl[1][1]*(t18*t15)-l_hl[1][2]*(t23*t15);
	CV_MAT_ELEM(*lp_hom,float,1,1) = l_hl[1][0]*t49+l_hl[1][1]*(t31*t15)-l_hl[1][2]*(t35*t15);
	CV_MAT_ELEM(*lp_hom,float,1,2) = -l_hl[1][0]*t50-l_hl[1][1]*(t44*t15)+l_hl[1][2]*(t47*t15);
	CV_MAT_ELEM(*lp_hom,float,2,0) = l_hl[2][0]*t48+l_hl[2][1]*(t18*t15)-t23*t15;
	CV_MAT_ELEM(*lp_hom,float,2,1) = l_hl[2][0]*t49+l_hl[2][1]*(t31*t15)-t35*t15;
	CV_MAT_ELEM(*lp_hom,float,2,2) = -l_hl[2][0]*t50-l_hl[2][1]*(t44*t15)+t47*t15;

	return lp_hom;
}

/****************************************************************************/

CvMat * cv_homography::compute_pn( CvMat * ap_dst_points,
                                   CvMat * ap_src_points )
{
	//Note: Points at infinity are not supported (it's not checked whether some
	//of the points are at infinity or not)...
	CvMat * lp_dst_points = cvCreateMat(ap_dst_points->rows,ap_dst_points->cols,ap_dst_points->type);
	CvMat * lp_src_points = cvCreateMat(ap_src_points->rows,ap_src_points->cols,ap_src_points->type);

	//compute mean...
	float l_dst_center_x = 0.0f;
	float l_dst_center_y = 0.0f;
	float l_src_center_x = 0.0f;
	float l_src_center_y = 0.0f;

	for( int l_i=0; l_i<ap_dst_points->cols; ++l_i )
	{
		l_dst_center_x += CV_MAT_ELEM(*ap_dst_points,float,0,l_i)/CV_MAT_ELEM(*ap_dst_points,float,2,l_i);
		l_dst_center_y += CV_MAT_ELEM(*ap_dst_points,float,1,l_i)/CV_MAT_ELEM(*ap_dst_points,float,2,l_i);
		l_src_center_x += CV_MAT_ELEM(*ap_src_points,float,0,l_i)/CV_MAT_ELEM(*ap_src_points,float,2,l_i);
		l_src_center_y += CV_MAT_ELEM(*ap_src_points,float,1,l_i)/CV_MAT_ELEM(*ap_src_points,float,2,l_i);
	}
	l_dst_center_x /= ap_dst_points->cols;
	l_dst_center_y /= ap_dst_points->cols;
	l_src_center_x /= ap_dst_points->cols;
	l_src_center_y /= ap_dst_points->cols;

	//compute scale...
	float l_dst_scale = 0.0f;
	float l_src_scale = 0.0f;

	for( int l_i=0; l_i<ap_dst_points->cols; ++l_i )
	{
		l_dst_scale += sqrt(	SQR(CV_MAT_ELEM(*ap_dst_points,float,0,l_i)/CV_MAT_ELEM(*ap_dst_points,float,2,l_i)-l_dst_center_x)+
  								SQR(CV_MAT_ELEM(*ap_dst_points,float,1,l_i)/CV_MAT_ELEM(*ap_dst_points,float,2,l_i)-l_dst_center_y));
		 l_src_scale += sqrt(	SQR(CV_MAT_ELEM(*ap_src_points,float,0,l_i)/CV_MAT_ELEM(*ap_src_points,float,2,l_i)-l_src_center_x)+
								SQR(CV_MAT_ELEM(*ap_src_points,float,1,l_i)/CV_MAT_ELEM(*ap_src_points,float,2,l_i)-l_src_center_y));
	}
	l_dst_scale /= ap_dst_points->cols;
	l_src_scale /= ap_dst_points->cols;

	l_dst_scale = sqrt(2.0f)/l_dst_scale;
	l_src_scale = sqrt(2.0f)/l_src_scale;

	//create transformations for normalization...
	CvMat * lp_dst_scaling = cvCreateMat(3,3,CV_32FC1);
	CvMat * lp_src_scaling = cvCreateMat(3,3,CV_32FC1);
	CvMat * lp_dst_translation = cvCreateMat(3,3,CV_32FC1);
	CvMat * lp_src_translation = cvCreateMat(3,3,CV_32FC1);
	CvMat * lp_dst_normalization = cvCreateMat(3,3,CV_32FC1);
	CvMat * lp_src_normalization = cvCreateMat(3,3,CV_32FC1);

	cvSetIdentity(lp_src_translation);
	cvSetIdentity(lp_dst_translation);
	cvSetIdentity(lp_src_scaling);
	cvSetIdentity(lp_dst_scaling);

	CV_MAT_ELEM(*lp_dst_translation,float,0,2) = -l_dst_center_x;
	CV_MAT_ELEM(*lp_dst_translation,float,1,2) = -l_dst_center_y;

	CV_MAT_ELEM(*lp_src_translation,float,0,2) = -l_src_center_x;
	CV_MAT_ELEM(*lp_src_translation,float,1,2) = -l_src_center_y;

	CV_MAT_ELEM(*lp_dst_scaling,float,0,0) = l_dst_scale;
	CV_MAT_ELEM(*lp_dst_scaling,float,1,1) = l_dst_scale;

	CV_MAT_ELEM(*lp_src_scaling,float,0,0) = l_src_scale;
	CV_MAT_ELEM(*lp_src_scaling,float,1,1) = l_src_scale;

	cvMatMul(lp_dst_scaling, lp_dst_translation, lp_dst_normalization);
	cvMatMul(lp_src_scaling, lp_src_translation, lp_src_normalization);

	//normalize points...
	cvMatMul(lp_dst_normalization, ap_dst_points, lp_dst_points);
	cvMatMul(lp_src_normalization, ap_src_points, lp_src_points);

	//create matrix for DLT...
	CvMat * lp_mat_A = cvCreateMat(2*lp_dst_points->cols, 9, CV_32FC1);
	for( int l_i=0; l_i<lp_dst_points->cols; ++l_i )
	{
		CV_MAT_ELEM(*lp_mat_A,float,2*l_i,0) = 0.0f;
		CV_MAT_ELEM(*lp_mat_A,float,2*l_i,1) = 0.0f;
		CV_MAT_ELEM(*lp_mat_A,float,2*l_i,2) = 0.0f;
		CV_MAT_ELEM(*lp_mat_A,float,2*l_i,3) = -CV_MAT_ELEM(*lp_dst_points,float,2,l_i)*CV_MAT_ELEM(*lp_src_points,float,0,l_i);
		CV_MAT_ELEM(*lp_mat_A,float,2*l_i,4) = -CV_MAT_ELEM(*lp_dst_points,float,2,l_i)*CV_MAT_ELEM(*lp_src_points,float,1,l_i);
		CV_MAT_ELEM(*lp_mat_A,float,2*l_i,5) = -CV_MAT_ELEM(*lp_dst_points,float,2,l_i)*CV_MAT_ELEM(*lp_src_points,float,2,l_i);
		CV_MAT_ELEM(*lp_mat_A,float,2*l_i,6) = CV_MAT_ELEM(*lp_dst_points,float,1,l_i)*CV_MAT_ELEM(*lp_src_points,float,0,l_i);
		CV_MAT_ELEM(*lp_mat_A,float,2*l_i,7) = CV_MAT_ELEM(*lp_dst_points,float,1,l_i)*CV_MAT_ELEM(*lp_src_points,float,1,l_i);
		CV_MAT_ELEM(*lp_mat_A,float,2*l_i,8) = CV_MAT_ELEM(*lp_dst_points,float,1,l_i)*CV_MAT_ELEM(*lp_src_points,float,2,l_i);

		CV_MAT_ELEM(*lp_mat_A,float,2*l_i+1,0) = CV_MAT_ELEM(*lp_dst_points,float,2,l_i)*CV_MAT_ELEM(*lp_src_points,float,0,l_i);
		CV_MAT_ELEM(*lp_mat_A,float,2*l_i+1,1) = CV_MAT_ELEM(*lp_dst_points,float,2,l_i)*CV_MAT_ELEM(*lp_src_points,float,1,l_i);
		CV_MAT_ELEM(*lp_mat_A,float,2*l_i+1,2) = CV_MAT_ELEM(*lp_dst_points,float,2,l_i)*CV_MAT_ELEM(*lp_src_points,float,2,l_i);
		CV_MAT_ELEM(*lp_mat_A,float,2*l_i+1,3) = 0.0f;
		CV_MAT_ELEM(*lp_mat_A,float,2*l_i+1,4) = 0.0f;
		CV_MAT_ELEM(*lp_mat_A,float,2*l_i+1,5) = 0.0f;
		CV_MAT_ELEM(*lp_mat_A,float,2*l_i+1,6) = -CV_MAT_ELEM(*lp_dst_points,float,0,l_i)*CV_MAT_ELEM(*lp_src_points,float,0,l_i);
		CV_MAT_ELEM(*lp_mat_A,float,2*l_i+1,7) = -CV_MAT_ELEM(*lp_dst_points,float,0,l_i)*CV_MAT_ELEM(*lp_src_points,float,1,l_i);
		CV_MAT_ELEM(*lp_mat_A,float,2*l_i+1,8) = -CV_MAT_ELEM(*lp_dst_points,float,0,l_i)*CV_MAT_ELEM(*lp_src_points,float,2,l_i);
	}
	//compute SVD...
	CvMat * lp_mat_V = cvCreateMat(9, 9, CV_32FC1);
	CvMat * lp_mat_W = cvCreateMat(9, 1, CV_32FC1);

	cvSVD(lp_mat_A, lp_mat_W, NULL, lp_mat_V, CV_SVD_MODIFY_A);

	//get homography from SVD...
	CvMat * lp_tmp_homography = cvCreateMat(3, 3, CV_32FC1);

	CV_MAT_ELEM(*lp_tmp_homography,float,0,0) = CV_MAT_ELEM(*lp_mat_V,float,0,8);
	CV_MAT_ELEM(*lp_tmp_homography,float,0,1) = CV_MAT_ELEM(*lp_mat_V,float,1,8);
	CV_MAT_ELEM(*lp_tmp_homography,float,0,2) = CV_MAT_ELEM(*lp_mat_V,float,2,8);
	CV_MAT_ELEM(*lp_tmp_homography,float,1,0) = CV_MAT_ELEM(*lp_mat_V,float,3,8);
	CV_MAT_ELEM(*lp_tmp_homography,float,1,1) = CV_MAT_ELEM(*lp_mat_V,float,4,8);
	CV_MAT_ELEM(*lp_tmp_homography,float,1,2) = CV_MAT_ELEM(*lp_mat_V,float,5,8);
	CV_MAT_ELEM(*lp_tmp_homography,float,2,0) = CV_MAT_ELEM(*lp_mat_V,float,6,8);
	CV_MAT_ELEM(*lp_tmp_homography,float,2,1) = CV_MAT_ELEM(*lp_mat_V,float,7,8);
	CV_MAT_ELEM(*lp_tmp_homography,float,2,2) = CV_MAT_ELEM(*lp_mat_V,float,8,8);

	//denormalize result...
	CvMat * lp_inv_dst_normalization = cvCreateMat(3, 3, CV_32FC1);
	CvMat * lp_tmp_mat = cvCreateMat(3, 3, CV_32FC1);
	CvMat * lp_homography = cvCreateMat(3, 3, CV_32FC1);

	cvInvert(lp_dst_normalization, lp_inv_dst_normalization);
	cvMatMul(lp_inv_dst_normalization, lp_tmp_homography, lp_tmp_mat);
	cvMatMul(lp_tmp_mat, lp_src_normalization, lp_homography);

	cvReleaseMat(&lp_inv_dst_normalization);
	cvReleaseMat(&lp_tmp_mat);
	cvReleaseMat(&lp_tmp_homography);
	cvReleaseMat(&lp_mat_V);
	cvReleaseMat(&lp_mat_W);
	cvReleaseMat(&lp_mat_A);
	cvReleaseMat(&lp_dst_normalization);
	cvReleaseMat(&lp_src_normalization);
	cvReleaseMat(&lp_dst_translation);
	cvReleaseMat(&lp_dst_scaling);
	cvReleaseMat(&lp_src_translation);
	cvReleaseMat(&lp_src_scaling);
	cvReleaseMat(&lp_dst_points);
	cvReleaseMat(&lp_src_points);

	return lp_homography;
}

/******************************* END OF FILE ********************************/
