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
// cv_homography: cv_homography.h											//
//																			//
// Authors: Stefan Hinterstoisser 2007										//
// Version: 1.0																//
//																			//
//////////////////////////////////////////////////////////////////////////////

#ifndef CV_HOMOGRAPHY_H
#define CV_HOMOGRAPHY_H

/******************************* includes ***********************************/

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
 * \class cv_homography
 * \brief homography is a class to compute the homography with 4 or more 
 * \point correspondences.
 * \author Stefan Hinterstoisser
 * \version 1.0
 * \date 2007
 * @author Stefan Hinterstoisser Copyright 2007 Munich
 */ 
class cv_homography
{
public:

	static CvMat * compute(	CvMat * ap_dst_points,
							CvMat * ap_src_points );

protected:

	static CvMat * compute_p4(	CvMat * ap_dst_points, 
								CvMat * ap_src_points );

	static CvMat * compute_p4_closed(	CvMat * ap_dst_points,
										CvMat * ap_src_points );

	static CvMat * compute_pn(	CvMat * ap_dst_points, 
								CvMat * ap_src_points );
};

}//end of namespace

#endif

/******************************* END OF FILE ********************************/
