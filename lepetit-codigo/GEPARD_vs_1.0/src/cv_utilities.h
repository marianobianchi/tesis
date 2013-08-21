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
// cv_define: cv_define.h													//
//																			//
// Authors: Stefan Hinterstoisser 2010										//
// Version: 1.0																//
//																			//
//////////////////////////////////////////////////////////////////////////////

#ifndef CV_UTILITIES_H
#define CV_UTILITIES_H

/********************************* flags ************************************/

#define IPP_INCLUDED

/******************************* includes ***********************************/

#include "cv.h"
#include "ipp.h"
#include "cxcore.h"
#include "highgui.h"

#include "emmintrin.h"

#include <string>
#include <iostream>
#include <fstream>

/******************************** typedefs **********************************/

/******************************** defines ***********************************/

#define SQR(a) ((a)*(a))

/******************************* namespaces *********************************/
/**
 *\namespace Namespace ma.
 *\brief Basic namepace for a c++ utility library.
 */
namespace cv
{
/***************************** forward declarations ************************/

/***************************** constant variables **************************/

const double cv_pi = 3.141592653589;

/***************************** helping functions ***************************/

CvMat * cv_copy_from_image( IplImage * ap_image,
							int a_height,
							int a_width,
							int a_row,
							int a_col );

/***************************************************************************/

void print_bit(Ipp8u);	
void print_bit(Ipp16u);
void print_bit(Ipp32s);
void print_bit(Ipp64u);
void print_bit(__m128i);

/***************************************************************************/

void cv_copy( float * ap_src, float * ap_dst, int a_length );

/***************************************************************************/

void cv_add( float * ap_src, float * ap_dst, int a_length );

/***************************************************************************/

void cv_set( float * ap_dst, float a_scalar, int a_length );

/***************************************************************************/

void cv_sqr( float * ap_src, float * ap_dst, int a_length );

/***************************************************************************/

void cv_sqrt( float * ap_src, float * ap_dst, int a_length );

/***************************************************************************/

std::pair<int,float> cv_find_max( float * ap_src, int a_length );

/***************************************************************************/

float cv_dot_product( float * ap_src1, float * ap_src2, int a_length );

/***************************************************************************/

void cv_lin_combination(	float * ap_src1, float a_scalar1, 
							float * ap_src2, float a_scalar2,
							float * ap_dest, int a_length );

/***************************************************************************/

void cv_normalize_mean_std( float * ap_src1, int a_length );

/***************************************************************************/

void cv_normalize_mean( float * ap_src1, int a_length );

/***************************************************************************/

void cv_normalize_std( float * ap_src1, int a_length );

/***************************************************************************/

void cv_mat_mul( CvMat * ap_src1, CvMat * ap_src2, CvMat * ap_src3 );

/***************************************************************************/

void cv_mat_mul_transposed( CvMat * ap_src1, CvMat * ap_src2, CvMat * ap_src3 );

/***************************************************************************/

IplImage * cv_convert_color_to_gray( IplImage * ap_image );

/***************************************************************************/

IplImage * cv_convert_gray_to_color( IplImage * ap_image );

/***************************************************************************/

IplImage * cv_roi_mask(	IplImage * ap_image,
						CvMat * ap_points );

/***************************************************************************/

IplImage * cv_smooth(	IplImage * ap_image,
						int a_block_size );

/***************************************************************************/

CvMat * cv_right_pseudo_inverse( CvMat * ap_mat ); 

/***************************************************************************/

CvMat * cv_left_pseudo_inverse( CvMat * ap_mat ); 

/***************************************************************************/

CvMat * cv_pseudo_inverse( CvMat * ap_mat ); 

/***************************************************************************/

void cv_add_noise(	IplImage * ap_image,
					float a_var );

/***************************************************************************/

void cv_add_illumination(	IplImage * ap_image,
							float a_factor,
							float a_shift );

/***************************************************************************/

void cv_homogenize( CvMat * ap_points ); 

/***************************************************************************/

void cv_print( CvMat * ap_mat );

/***************************************************************************/

void cv_print( IplImage * ap_img );

/***************************************************************************/

void cv_print_real( IplImage * ap_mat, int a_borderx=-1, int a_bordery=-1 );

/***************************************************************************/

void cv_print_complex( IplImage * ap_mat, int a_borderx=-1, int a_bordery=-1 );

/***************************************************************************/

void cv_draw_points(	IplImage * ap_image,
						CvMat * ap_points,
						int a_radius,
						int a_r, int a_g, int a_b,
						int a_i=-1 );

/***************************************************************************/

void cv_draw_poly(	IplImage * ap_image,
					CvMat * ap_points,
					int a_thickness,
					int a_r,
					int a_g,
					int a_b );

/***************************************************************************/

void cv_create_window( std::string a_window_name );

/***************************************************************************/

void cv_show_image( IplImage * ap_image );

/***************************************************************************/

void cv_show_image(	IplImage * ap_image, 
    				std::string a_window_name );

/***************************************************************************/

void cv_show_points(	IplImage * ap_image,
						CvMat * ap_points,
						int a_radius,
						int a_r,
						int a_g,
						int a_b );

/***************************************************************************/

IplImage * cv_load_image( std::string a_name );

/***************************************************************************/

IplImage * cv_load_image_uchar( std::string a_name );

/***************************************************************************/

void cv_save_image( std::string a_name, IplImage * ap_image);

/***************************************************************************/

class cv_mouse
{
public:
	static void cv_on_mouse( int a_event, int a_x, int a_y, int a_flags, void * a_params ){ m_event = a_event; m_x = a_x; m_y = a_y; }
	static void start( std::string a_img_name ){ cvSetMouseCallback( a_img_name.c_str(), cv_mouse::cv_on_mouse, 0 ); }
	static int get_event( void ){ int l_event=m_event; m_event=-1; return l_event;}
	static int get_x( void ){ int l_x=m_x; m_x=-1; return l_x;}
	static int get_y( void ){ int l_y=m_y; m_y=-1; return l_y;}

private:
	static int m_event;
	static int m_x;
	static int m_y;
};

/***************************************************************************/

class cv_timer
{
public:

#ifdef WIN32
	cv_timer() : m_time(0)
	{LARGE_INTEGER l_tmp; ::QueryPerformanceFrequency(&l_tmp); m_freq=l_tmp.QuadPart;}

	void start( bool a_reset=true)
	{LARGE_INTEGER l_tmp; ::QueryPerformanceCounter(&l_tmp); if (a_reset) m_time=0; m_start=l_tmp.QuadPart;}

	void stop(void)
	{LARGE_INTEGER l_tmp; ::QueryPerformanceCounter(&l_tmp); m_stop=l_tmp.QuadPart; m_time+=m_stop-m_start;}

#else
	cv_timer() : m_time(0)
	{  m_freq = 1e6 * cvGetTickFrequency(); }

	void start( bool a_reset=true )
	{ Ipp64u l_tmp = cvGetTickCount(); if(a_reset) m_time=0; m_start=l_tmp; }

	void stop(void)
	{ Ipp64u l_tmp = cvGetTickCount(); m_stop=l_tmp; m_time+=m_stop-m_start; }

#endif
	
	double get_time()
	{return(double)m_time/(double)m_freq;};

	double get_fps() 
	{return(double)m_freq/(double)m_time;};

	long get_clocks()
	{return m_time;};

private:

#ifdef WIN32
	__int64 m_freq;
	__int64 m_start, m_stop, m_time;
#else
	Ipp64u m_freq;
	Ipp64u m_start, m_stop, m_time;
#endif
};

/***************************************************************************/

#define TIME(text,expr,l_timer) l_timer.start(); expr; l_timer.stop(); printf("%s %.7f sec\n",text,l_timer.get_time());
#define FPS(text,expr,l_timer) l_timer.start(); expr; l_timer.stop(); printf("%s %.2f fps\n",text,l_timer.get_fps());

/***************************************************************************/

std::ofstream & cv_write( std::ofstream & a_os, CvMat * ap_mat );

/***************************************************************************/

std::ofstream & cv_write( std::ofstream & a_os, IplImage * ap_img );

/***************************************************************************/

CvMat * cv_read( std::ifstream & a_is );

/***************************************************************************/

void cv_save_mat( std::string a_name, CvMat * ap_mat );

/***************************************************************************/

CvMat * cv_load_mat( std::string a_name );

/***************************************************************************/

IplImage * cv_read_img( std::ifstream & a_is );

/***************************************************************************/

bool cv_homography_heuristic( CvMat * ap_quad, float a_fac, float a_pix );

/***************************************************************************/

} // end of namespace
#endif

/****************************** END OF FILE  ********************************/

