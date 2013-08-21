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
// cv_esm: cv_esm.h															//
//																			//
// Authors: Stefan Hinterstoisser 2010										//
// Version: 1.0																//
//																			//
//////////////////////////////////////////////////////////////////////////////

#ifndef CV_ESM_H
#define CV_ESM_H

/******************************* includes ***********************************/

#include "ESMlibry.h"

#include "cv.h"
#include "cxcore.h"
#include "highgui.h"

#include "cv_utilities.h"

/******************************** defines ***********************************/

/******************************* namespaces *********************************/
/**
 *\namespace Namespace CV.
 *\brief Basic namepace for a computer vision c++ library.
 */
namespace cv
{

/**************************** forward declarations **************************/

/*************************** structs / typedefs *****************************/

/********************************** classes *********************************/
/** 
 * \class cv_esm
 * \brief esm is an intensity based tracker (by Selim Benhimane).
 * \author Stefan Hinterstoisser
 * \version 1.0
 * \date 2008
 * @author Stefan Hinterstoisser Copyright 2008 Munich
 */ 
class cv_esm
{
public:

	cv_esm( void );
	~cv_esm();

	bool learn(	IplImage * ap_image,
				int a_ulrow, 
				int a_ulcol,
				int a_height,
				int a_width );

	CvMat * track(	IplImage * ap_image, 
					CvMat * ap_rec,
					int a_max_iter,
					int a_prec );
	
	IplImage * get_cur_pattern( void );
	IplImage * get_ref_pattern( void );

	int get_height( void ){ return m_height; }
	int get_width( void ){ return m_width; }

	float get_ncc( void );
	CvMat * get_H( void );
	
protected:

	int m_width;
	int m_height;

	bool m_mask_flag;
	bool m_delete_flag;

	trackStruct m_T; 

	CvMat * mp_rec;
	IplImage * mp_ref;
};

}//end of namespace

#endif

/******************************* END OF FILE ********************************/
