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
// cv_camera: cv_camera.h													//
//																			//
// Authors: Stefan Hinterstoisser 2010										//
// Version: 1.0																//
//																			//
//////////////////////////////////////////////////////////////////////////////

#ifndef CV_CAMERA_H
#define CV_CAMERA_H

/******************************* includes ***********************************/

#include "cv.h"
#include "cxcore.h"
#include "highgui.h"

#ifdef WIN32
#include "videoInput.h"
#endif

#include "cv_utilities.h"

/******************************** defines ***********************************/

/******************************* namespaces *********************************/
/**
 *\namespace Namespace cv for Computer Vision.
 *\brief Basic namepace for a generic c++ vision library.
 */
namespace cv
{
/****************************** structs / types *****************************/

enum cv_capture_source
{
	usb
};

/**************************** forward declarations **************************/

/********************************** classes *********************************/
/** 
 * \class cv_camera_usb
 * \brief This class provides basic functionallity for capturing images
 * \from cameras running in a separated thread.
 * \author Stefan Hinterstoisser
 * \version 1.0
 * \date 2009
 * @author Stefan Hinterstoisser Copyright 2009 Munich
 */
class cv_camera_usb
{
public:

	cv_camera_usb( void );
	~cv_camera_usb();

	bool set_capture_data(	int a_width=-1,
							int a_height=-1,
							double a_fps=-1 );

	IplImage * get_image( void );
	IplImage * get_byte_image( void );

	bool stop( void );
	void run( void );

	int get_width( void ){ return static_cast<int>( m_width ); }
	int get_height( void ){ return static_cast<int>( m_height ); }

protected:
	
	bool m_run;

	long m_width;
	long m_height;

	double m_fps;

#ifdef WIN32
	videoInput m_cam;
#else
	CvCapture * mp_cam;
#endif

	IplImage * mp_buffer;
};

/****************************************************************************/
/** 
 * \class cv_camera 
 * \brief This class provides basic functionallity for capturing
 * \images from cameras (WEBCAM).
 * \author Stefan Hinterstoisser
 * \version 1.0
 * \date 2008
 * @author Stefan Hinterstoisser Copyright 2008 Munich
 */
class cv_camera
{
public:

	cv_camera( void );
	~cv_camera( void );

	void set_cam( cv_capture_source a_capture_source )
	{m_capture_source = a_capture_source;}

	cv_capture_source get_cam( void )
	{return m_capture_source;}

	bool stop_capture_from_cam( void );
	bool start_capture_from_cam(	int a_width=-1,
									int a_height=-1,
									double a_fps=-1 );
	IplImage * get_image( void ); 
	IplImage * get_byte_image( void ); 
	
protected:

	void convert_flip( IplImage * ap_src, IplImage * ap_dst );

protected:

	IplImage * mp_buffer;

	cv_capture_source m_capture_source;
	cv_camera_usb  * mp_capture_thread_usb;
};

}//End of namespace...
#endif

/****************************** END OF FILE  ********************************/
