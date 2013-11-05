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
// cv_hyper: cv_hyper.h														//
//																			//
// Authors: Stefan Hinterstoisser 2009										//
// Version: 1.0																//
//																			//
//////////////////////////////////////////////////////////////////////////////

#ifndef CV_HYPER_H
#define CV_HYPER_H

/******************************* includes ***********************************/

#include <string>
#include <fstream>

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

//class cv_pcabase;

/*************************** structs / typedefs *****************************/

/********************************** classes *********************************/
/** 
 * \class cv_hyper
 * \brief cv_hyper is an intensity based tracker with hyperplane approximation.
 * \author Stefan Hinterstoisser
 * \version 1.0
 * \date 2009
 * @author Stefan Hinterstoisser Copyright 2009 Munich
 */ 
class cv_hyper
{
//	friend cv_pcabase;

public:
	
	cv_hyper(void);
	~cv_hyper();
 
	void set_parameters(	int a_nx, int a_ny, 
							int a_num_of_levels,
							int a_max_motion,
							int a_num_of_samples,
							int a_border );
  
	void learn( IplImage * ap_image,
				CvMat * ap_rec );
	     
	CvMat * track(	IplImage * ap_image,
					CvMat * ap_rec,
					int a_num_of_iters );

	float get_ncc( void ){ return m_ncc; }

	std::ofstream & write( std::ofstream & a_os );
	std::ifstream & read( std::ifstream & a_is );

	bool save( std::string a_name );
	bool load( std::string a_name );	

//private:
	 
	void find_2d_points(IplImage * ap_image );

	void compute_as_matrices(	IplImage * ap_image,
							    CvMat ** ap_as,
								CvMat * ap_rec,
								CvMat * ap_pos,
								CvMat * ap_int,
								int a_nx, int a_ny,
								int a_max_motion,
								int a_num_of_levels,
								int a_num_of_samples );

	void compute_as_level(	IplImage * ap_image,
							CvMat * ap_rec,
							CvMat * ap_pos,
							CvMat * ap_int,
							CvMat ** ap_as,
							int a_num_of_samples,
							int a_num_of_levels,
							int a_max_motion,
							int a_level,
							int a_nx, int a_ny );

	void get_local_maximum(	IplImage * ap_image,
							float a_row1,   float a_col1,
							float a_height, float a_width,
							float & a_row2, float & a_col2 );

	IplImage * compute_gradient(IplImage * ap_image,
								int a_ulr,
								int a_ulc,
								int a_height,
								int a_width );

	void init_ncc(	IplImage * ap_image,
					CvMat * ap_rec,
					CvMat * &ap_ncc_pos,
					CvMat * &ap_ncc_int );

	float compute_ncc(	IplImage * ap_image,
						CvMat * ap_pos,
						CvMat * ap_int,
						CvMat * ap_hom );

	void move(	float a_row1,
				float a_col1,
				float & a_row2,
				float & a_col2,
				float a_amp);

	void add_noise(CvMat * ap_vec);
		
	void clear( void );

//private:

	int m_nx;
	int m_ny;

	float m_ncc;

	int m_border;
	int m_max_motion;
	int m_num_of_levels;
	int m_num_of_samples;

	CvMat * lp_pix;

	CvMat * mp_pos;
	CvMat * mp_rec;
	CvMat * mp_int;

	CvMat ** mp_as;
};

}//end of namespace

#endif

/******************************* END OF FILE ********************************/
