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
// cv_harris: cv_harris.h													//
//																			//
// Authors: Stefan Hinterstoisser 2007										//
// Version: 1.0																//
//																			//
//////////////////////////////////////////////////////////////////////////////

#ifndef CV_HARRIS_H
#define CV_HARRIS_H

/******************************* includes ***********************************/

#include <vector>

#include "cv.h"
#include "cxcore.h"
#include "highgui.h"

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

/**************************** forward declarations **************************/

/********************************** classes *********************************/
/** 
 * \class cv_harris 
 * \brief This class provides basic functionallity for 
 * \key/anchor point extraction.
 * \author Stefan Hinterstoisser
 * \version 1.0
 * \date 2007
 * @author Stefan Hinterstoisser Copyright 2007 Munich
 */
class cv_harris
{
public:

	static void set_radius( double a_radius=4 )
	{m_radius=a_radius;}

	static void set_threshold( double a_threshold=10000 )
	{m_threshold = a_threshold;}

	static void set_num_of_points( int a_num_of_points )
	{m_num_of_points = a_num_of_points;}

	static void set_distance( double a_distance=4 )
	{m_distance = a_distance;}

	static void set_iterations( int a_num_of_iterations=50 )
	{m_num_of_iterations=a_num_of_iterations;}


	static CvMat * get_points(	IplImage * ap_image );

	static CvMat * get_most_robust_points(	IplImage * ap_image,
											IplImage * ap_mask=NULL );

protected:

	static CvMat * get_points(	IplImage * ap_image,
								int a_num_of_points,
								double a_threshold,
								double a_radius );

	static CvMat * get_most_robust_points(	IplImage * ap_image,
											IplImage * ap_mask,
											int a_num_of_points,
											double a_threshold,
											double a_radius );
	
	static std::pair<CvMat*,CvMat*> get_random_transformation(	int a_row,
																int a_col );
	
	static void backproject_and_match(	std::vector<std::pair<int,CvPoint2D32f> > & a_cont,
										CvMat * ap_transformation,
										CvMat * ap_points );

	static IplImage * get_masked_image( IplImage * ap_image, 
										IplImage * ap_mask );

protected:

	static double m_radius;
	static double m_distance;
	static double m_threshold;
	static int m_num_of_points;
	static int m_num_of_iterations;
};

}//End of namespace...
#endif

/****************************** END OF FILE  ********************************/

