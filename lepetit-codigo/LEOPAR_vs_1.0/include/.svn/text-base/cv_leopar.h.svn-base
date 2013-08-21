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
// cv_leopar: cv_leopar.h													//
//																			//
// Authors: Stefan Hinterstoisser 2008										//
// Version: 1.0																//
//																			//
//////////////////////////////////////////////////////////////////////////////

#ifndef CV_LEOPAR_H
#define CV_LEOPAR_H

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

/*************************** structs / typedefs *****************************/

/********************************** classes *********************************/
/** 
 * \class cv_leopar
 * \brief leopar is a method to compute a rough homography based on the 
 * \surrounding patch information based on ferns.
 * \author Stefan Hinterstoisser
 * \version 1.0
 * \date 2008
 * @author Stefan Hinterstoisser Copyright 2008 Munich
 */ 
class cv_leopar 
{
public:

	cv_leopar( void );
	~cv_leopar();

	void set_parameters(	int a_size,
							int a_depth,
							int a_num_of_ferns,
							int a_num_of_poses,
							int a_num_of_samples );

	double get_patch_size( void ){ return m_size; }
	
	std::vector<CvMat*> get_closest_pose(	CvMat * ap_dest_quad,
											int a_row,
											int a_col );
	bool learn( IplImage * ap_image,
				IplImage * ap_mask,
				int a_row,
				int a_col );

	CvMat * recognize(	IplImage * ap_image, 
						int a_row, 
						int a_col );

	std::ofstream & write( std::ofstream & a_os );
	std::ifstream & read( std::ifstream & a_is );

	bool save( std::string a_name );
	bool load( std::string a_name );

protected:

	void learn_initialize( void );

	void learn_finalize( void );
	
	bool learn_incremental(	IplImage * ap_image,
							IplImage * ap_mask,
							CvMat * ap_dest_quad,
							int a_num_of_samples,
							int a_row,
							int a_col );
private:

	void add_distributions(	CvMat * ap_src_distribution,
							CvMat * ap_dst_distribution,
							int * ap_leaves_index,
							int a_num_of_ferns,
							int a_num_of_leaves,
							int a_num_of_poses );

	int * get_closest_bins(	CvMat * ap_dest_quad,
							int a_row,
							int a_col );

	int * drop_patch(	IplImage * ap_image,
						CvMat * ap_points,
						int a_row,
						int a_col,
						int a_num_of_ferns,
						int a_depth );
	
	CvMat * gen_rand_aff_trans(	float a_row,
								float a_col,
								float a_phi,
								float a_lam );

	int find_max(	CvMat * ap_distribution, 
					int a_num_of_poses );

	float get_linear(	IplImage * ap_image, 
						double a_row,
						double a_col );

	void fix_tests(	CvMat * ap_points, 
					int a_num_of_ferns,
					int a_depth,
					int a_size );

	void fix_poses(	CvMat * ap_poses,
					int a_num_of_poses,
					int a_size );	
	
	void set_aff(	IplImage * ap_image,
					IplImage * ap_mask,
					IplImage * ap_patch,
					CvMat * ap_pos,
					CvMat * ap_aff,
					float a_row,
					float a_col,
					float a_var );
protected:

	int m_depth;
	int m_size;

	int m_num_of_leaves;
	int m_num_of_ferns;
	int m_num_of_points;
	int m_num_of_poses;
	int m_num_of_samples;

	float m_magic_factor;

	CvMat * mp_poses;
	CvMat * mp_points;

	CvMat * mp_pose_distributions1;
	CvMat * mp_pose_distributions2;
	CvMat * mp_pose_distributions3;
	CvMat * mp_pose_distributions4;

	CvMat * mp_pose_result;
};

}//end of namespace

#endif

/******************************* END OF FILE ********************************/
