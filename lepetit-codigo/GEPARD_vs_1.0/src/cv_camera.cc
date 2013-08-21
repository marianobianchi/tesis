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
// cv_camera: cv_camera.cc													//
//																			//
// Authors: Stefan Hinterstoisser 2010										//
// Version: 1.0																//
//																			//
//////////////////////////////////////////////////////////////////////////////

/******************************* includes **********************************/

#include "cv_camera.h"

#include <string>
#include <sstream>

/******************************** defines **********************************/

/******************************* namespaces ********************************/

using namespace cv;

/****************************** constructors *******************************/

cv_camera_usb::cv_camera_usb( void ) 
{
	m_fps = -1;
	m_width = -1;
	m_height = -1;
	mp_buffer = NULL;
	m_run = false;

#ifndef WIN32
	mp_cam = NULL;
#endif
}

/******************************* destructor ********************************/

cv_camera_usb::~cv_camera_usb()
{
	std::cerr << "********************************************" << std::endl;
	if( mp_buffer != NULL )
		cvReleaseImage(&mp_buffer);
}

/***************************************************************************/

bool cv_camera_usb::set_capture_data(	int a_width,
										int a_height,
										double a_fps )
{
#ifdef WIN32

	std::cerr << "********************************************" << std::endl;

	int l_device = 0;

	if( a_width < 0 || a_height < 0 || a_fps < 0 ) return false;

	m_cam.setIdealFramerate(l_device,a_fps);
	m_run = m_cam.setupDevice(l_device,a_width,a_height);

	m_width 	= m_cam.getWidth(l_device);
	m_height 	= m_cam.getHeight(l_device);
	m_fps		= a_fps;

	mp_buffer = cvCreateImage(cvSize(m_width,m_height),8,3); 

	std::cerr << "********************************************" << std::endl;

	return m_run;
#else
	mp_cam = cvCaptureFromCAM(0);

	if( mp_cam != NULL ) 
		m_run = true;
	else
		m_run = false;

	return m_run;
#endif
}

/***************************************************************************/


void cv_camera_usb::run( void )
{
//you do not need to do something here...
}

/***************************************************************************/

IplImage * cv_camera_usb::get_image( void )
{
#ifdef WIN32		

	int l_device = 0;

	m_cam.getPixels(l_device,(unsigned char*)mp_buffer->imageData,false,true);

	IplImage * lp_color = cvCreateImage(cvSize(m_width,m_height),IPL_DEPTH_32F,3);
	cvConvert(mp_buffer,lp_color);
	
	return lp_color;
#else
	mp_buffer = cvQueryFrame(mp_cam);

	IplImage * lp_color = cvCreateImage(cvGetSize(mp_buffer),IPL_DEPTH_32F,3);
	cvConvert(mp_buffer,lp_color);

	return lp_color;
#endif
}

/***************************************************************************/

IplImage * cv_camera_usb::get_byte_image( void )
{
#ifdef WIN32		

	int l_device = 0;

	m_cam.getPixels(l_device,(unsigned char*)mp_buffer->imageData,false,true);

	IplImage * lp_color = cvCreateImage(cvSize(m_width,m_height),IPL_DEPTH_8U,3);
	cvConvert(mp_buffer,lp_color);

	return lp_color;
#else
	std::cerr << "get_byte_image is NOT implemented yet..." << std::endl;
	return NULL;
#endif
}

/***************************************************************************/

bool cv_camera_usb::stop( void )
{
	m_run = false;

#ifdef WIN32		
	
	std::cerr << "********************************************" << std::endl;
	
	int l_device = 0;

	m_cam.stopDevice(l_device);
	
	std::cerr << "********************************************" << std::endl;
#else
	cvReleaseCapture(&mp_cam);
#endif
	return true;
}

/***************************************************************************/

/******************************* includes **********************************/

/******************************** defines **********************************/

/******************************* namespaces ********************************/



/******************************* includes **********************************/

/******************************** defines **********************************/

/******************************* namespaces ********************************/

/****************************** constructors *******************************/

cv_camera::cv_camera( void ):
mp_capture_thread_usb(NULL),
mp_buffer(NULL)
{
}

/******************************* destructor ********************************/

cv_camera::~cv_camera()
{
	this->stop_capture_from_cam();
	
	if( mp_buffer != NULL )
		cvReleaseImageHeader(&mp_buffer);
	
	if( mp_capture_thread_usb )
		delete mp_capture_thread_usb;
}

/***************************************************************************/

bool cv_camera::start_capture_from_cam( int a_width,
										int a_height,
										double a_fps )
{
	delete mp_capture_thread_usb;

	cvReleaseImageHeader(&mp_buffer);

	bool l_start_flag=false;

	if( m_capture_source == usb )
	{
		if( a_width == -1 ) a_width = 640;
		if( a_height == -1 ) a_height = 480;
		if( a_fps == -1 ) a_fps = 30;

		mp_capture_thread_usb = new cv_camera_usb();
		l_start_flag = mp_capture_thread_usb->set_capture_data( a_width, a_height, a_fps );
		if( l_start_flag == false )
		{
			delete mp_capture_thread_usb;
			return false;
		}
	}
	else
	{
		printf("cv_camera: capture source is not defined!");
		return false;
	}
	return l_start_flag;
}
/***************************************************************************/

bool cv_camera::stop_capture_from_cam( void )
{
	bool l_stop_flag = false;

	if( m_capture_source == usb )
	{
		if( mp_capture_thread_usb != NULL )
		{
			l_stop_flag = mp_capture_thread_usb->stop();
		}
	}
	else
	{
		printf("cv_camera: stopping not defined!");
		return false;
	}
	return l_stop_flag;
}

/***************************************************************************/

void cv_camera::convert_flip( IplImage * ap_src, IplImage * ap_dst )
{
	int l_width		= (ap_src->width*ap_src->nChannels);
	int l_width_mod = (ap_src->width*ap_src->nChannels/8)*8;
	int l_height    = (ap_src->height)-1;

	for( int l_r=0; l_r<ap_src->height; ++l_r )
	{
		int l_c=0;

		float * lp_float_pointer = &CV_IMAGE_ELEM(ap_dst,float,l_r,0);
		unsigned char * lp_uchar_pointer = &CV_IMAGE_ELEM(ap_src,unsigned char,l_height-l_r,0);

		for(;l_c<l_width_mod;)
		{
			lp_float_pointer[0] = lp_uchar_pointer[0];
			lp_float_pointer[1] = lp_uchar_pointer[1];
			lp_float_pointer[2] = lp_uchar_pointer[2];
			lp_float_pointer[3] = lp_uchar_pointer[3];
			lp_float_pointer[4] = lp_uchar_pointer[4];
			lp_float_pointer[5] = lp_uchar_pointer[5];
			lp_float_pointer[6] = lp_uchar_pointer[6];
			lp_float_pointer[7] = lp_uchar_pointer[7];

			lp_float_pointer+=8;
			lp_uchar_pointer+=8;
			l_c+=8;
		}
		for(;l_c<l_width;)
		{
			lp_float_pointer[0] = lp_uchar_pointer[0];
			++lp_float_pointer;
			++lp_uchar_pointer;
			++l_c;
		}
	}
}

/***************************************************************************/

IplImage * cv_camera::get_image( void )
{
	if( m_capture_source == usb )
	{
		return mp_capture_thread_usb->get_image();
	}
	return NULL;
}

/***************************************************************************/

IplImage * cv_camera::get_byte_image( void )
{
	if( m_capture_source == usb )
	{
		return mp_capture_thread_usb->get_byte_image();
	}
	else
	{
		return NULL;	
	}
}


/************************ END OF FILE **************************************/
