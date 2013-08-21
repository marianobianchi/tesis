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
// cv_pcabase: cv_pcabase.h													//
//																			//
// Authors: Stefan Hinterstoisser 2009										//
// Version: 1.0																//
//																			//
//////////////////////////////////////////////////////////////////////////////

#ifndef CV_PCABASE_H
#define CV_PCABASE_H

/******************************* includes ***********************************/

#include <string>
#include <fstream>
#include <vector>

#include "cv.h"
#include "cxcore.h"
#include "highgui.h"

#include "cv_utilities.h"
#include "cv_gepard.h"

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
 * \class cv_pcabase
 * \brief cv_pcabase is base for fast cv_gepard tracker learning.
 * \author Stefan Hinterstoisser
 * \version 1.0
 * \date 2009
 * @author Stefan Hinterstoisser Copyright 2009 Munich
 */ 
class cv_pcabase
{
public:

	cv_pcabase( void );
	~cv_pcabase();

	int get_support_size( void ){ return m_sup_size; }
	int get_patch_size( void ){ return m_pat_size; }
	
	bool set_parameters(	CvMat * ap_k,
							int a_pat_size,
							int a_sup_size,
							int a_nx, int a_ny,
							int a_num_of_samples );
	
	bool add(	IplImage * ap_image, 
				IplImage * ap_mask,
				int a_row, 
				int a_col );	
		
	bool learn_base( int a_num_of_pcas );

	cv_gepard * get_tracker(	IplImage * ap_image,
								int a_num_of_pcas,
								int a_row,
								int a_col );
		
	std::ofstream & write( std::ofstream & a_os );
	std::ifstream & read( std::ifstream & a_is );

	bool save( std::string a_name );
	bool load( std::string a_name );

protected:

	CvMat * compute_pca_base(	std::vector<CvMat*> & a_samples,
								CvMat * ap_mean_patch,
								int a_num_of_pcas );

	CvMat * get_lin_combination( CvMat * ap_pcas,
								 CvMat * ap_alphas,
		  						 int a_num_of_pcas_to_use,
								 bool a_pca_plus_mean );

	CvMat * get_pca_projection( CvMat * ap_pcas,
								CvMat * ap_patch,
		  						int a_num_of_pcas_to_use );

	CvMat * get_back_gep_mean(	CvMat * ap_lin_comb,
								int a_view_index );

	void push_back_gep_mean(	CvMat * ap_all_base,
								CvMat * ap_gep_mean,
								int a_view_index,
								int a_pca_index );

	CvMat * learn_gepard_base(	CvMat * ap_pca_base,
								CvMat * ap_pca_mean,
								CvMat * ap_pos,
								CvMat * ap_k,
								std::vector<CvMat*> & a_gep_poses,
								int a_num_of_gep_trains,
								int a_sup_size,
								int a_size );
		
	cv_gepard * get_gepard( std::vector<CvMat*> & a_gep_poses,
							CvMat * ap_gep_base,
							CvMat * ap_pos,
							CvMat * ap_alphas,
							CvMat * ap_k,
							int a_num_of_pcas,
							int a_num_of_samples,
						    int a_nx, int a_ny,
							int a_sup_size,
							int a_pat_size );

	void clear_positions( void );
	void clear_samples( void );
	void clear_pose( void );
	void clear_pca( void );
	void clear( void );

private:

	std::vector<std::pair<CvPoint3D32f,float> > get_views( void );

	CvMat *	get_train_intensity(	IplImage * ap_image,
									CvMat * ap_position,
									int a_row,
									int a_col );

	CvMat *	create_positions(	int a_nx,
								int a_ny,
								int a_size );

	IplImage * get_patch(	CvMat * ap_mat,
							int a_sup_size,
							int a_index );

	CvMat * get_patch(	IplImage * ap_img,
						int a_sup_size,
						int a_row,
						int a_col );

	CvMat * get_axis(	CvPoint3D32f & a_axis,
						double a_angle,
						double a_orien );

	CvMat * get_pose(	CvMat * ap_k,
						CvMat * ap_v,
						float a_orien,
						float a_scale );

	float get_linear(	IplImage * ap_image, 
						double a_row,
						double a_col );
	
	IplImage * warp(	IplImage * ap_image,
						CvMat * ap_hom,
						int a_size,
						int a_row,
						int a_col );
private:

	int m_nx;
	int m_ny;

	int m_pat_size;
	int m_sup_size;
	
	int m_num_of_samples;
	
	CvMat * mp_k;
	CvMat * mp_pos;
	CvMat * mp_pca_mean;
	CvMat * mp_pca_base;
	CvMat * mp_gep_base;    

	std::vector<CvMat*> m_gep_poses;
	std::vector<CvMat*> m_pca_samples;
};

}//end of namespace

#endif

/******************************* END OF FILE ********************************/
