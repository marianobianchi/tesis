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
// cv_gepard: cv_gepard.h													//
//																			//
// Authors: Stefan Hinterstoisser 2009										//
// Version: 1.0																//
//																			//
//////////////////////////////////////////////////////////////////////////////

#ifndef CV_GEPARD_H
#define CV_GEPARD_H

/******************************* includes ***********************************/

#include <string>
#include <fstream>
#include <vector>

#include "cv.h"
#include "cxcore.h"
#include "highgui.h"

#include "cv_utilities.h"

/******************************** defines ***********************************/

/******************************* namespaces *********************************/
/**
 *\namespace Namespace cv.
 *\brief Basic namepace for a computer vision c++ library.
 */
namespace cv
{
/**************************** forward declarations **************************/

class cv_pcabase;

/*************************** structs / typedefs *****************************/

/********************************** classes *********************************/
/** 
 * \class cv_gepard
 * \brief gepard is a method to compute a rough homography based on the 
 * \surrounding patch information with mean patches.
 * \author Stefan Hinterstoisser
 * \version 1.0
 * \date 2009
 * @author Stefan Hinterstoisser Copyright 2009 Munich
 */ 
class cv_gepard 
{
	friend cv_pcabase;

public:

	cv_gepard( void );
	~cv_gepard();

	void set_parameters(	CvMat * ap_k, int a_size,
							int a_num_of_samples,
							int a_nx, int a_ny );

	double get_size( void ){ return m_size; }
	
	bool learn( IplImage * ap_image,
				IplImage * ap_mask,
				int a_row, int a_col );

	CvMat * recognize(	IplImage * ap_image, 
						int a_row, int a_col );

	std::ofstream & write( std::ofstream & a_os );
	std::ifstream & read( std::ifstream & a_is );

	bool save( std::string a_name );
	bool load( std::string a_name );

	std::pair<float,float> compare( cv_gepard * ap_gepard );

//protected:

	std::vector<std::pair<CvPoint3D32f,float> > get_views( void );
	
	CvMat *	get_train_intensity(	IplImage * ap_image,
									IplImage * ap_mask,
									CvMat * ap_pos );

	CvMat *	get_run_intensity(	IplImage * ap_image,
								CvMat * ap_pos );

	CvMat *	create_positions(	int a_nx,
								int a_ny,
								int a_height,
								int a_width );

	IplImage * get_patch(	CvMat * ap_mat,
							int a_sup_size, 
							int a_index );

	int find_max(	CvMat * ap_distributions,
					int a_num_of_poses );

	float get_linear(	IplImage * ap_image, 
						double a_row,
						double a_col );

	CvMat * get_axis(	CvPoint3D32f & a_axis,
						double a_angle,
						double a_orientation );

	CvMat * get_pose(	CvMat * ap_k,
						CvMat * ap_v,
						float a_orientation,
						float a_scale );

	IplImage * warp(	IplImage * ap_image,
						CvMat * ap_hom,
						int a_size,
						int a_row,
						int a_col );	
	
	void push_back(	CvMat * ap_mat,
					CvMat * ap_vec,
					int l_index );

	void clear( void );
	
//protected:

	int m_nx;
	int m_ny;
	int m_size;
	int m_num_of_samples;

	CvMat * mp_k;
	CvMat * mp_pos;

	std::vector<CvMat*> m_means;
	std::vector<CvMat*> m_poses;
};

}//end of namespace

#endif

/******************************* END OF FILE ********************************/
