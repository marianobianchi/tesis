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
// cv_dot_template: cv_dot_template.h										//
//																			//
// Authors: Stefan Hinterstoisser 2010										//
// Version: 1.0																//
//																			//
//////////////////////////////////////////////////////////////////////////////

#ifndef CV_DOT_TEMPLATE
#define CV_DOT_TEMPLATE

/******************************* includes ***********************************/

#include <algorithm>
#include <iostream>
#include <sstream>
#include <vector>
#include <list>

#include "emmintrin.h"
#include "ipp.h"

#include "cv_utilities.h"

/******************************** defines ***********************************/

//#define NO_CLUSTERING			//this flag is mainly for debugging
#define MAX_DOT_TEM_NUM 10000	//maximum number of templates allowed

/******************************* namespaces *********************************/
/**
 *\namespace Namespace cv for Computer Vision.
 *\brief Basic namepace for a generic c++ vision library.
 */
namespace cv
{
static char bits_set_in_16bit [0x1u<<16];
static char bits_unset_in_16bit [0x1u<<16];

#ifndef CV_CANDIDATE_STRUCT
#define CV_CANDIDATE_STRUCT

/*********************** structs / types / functions ************************/

struct cv_candidate
{
	int m_ind;
	int m_clu;
	int m_cla;
	int m_val;

	int m_row;
	int m_col;
};

struct cv_candidate_ptr_cmp
{
	bool operator()( cv_candidate * a_lhs, cv_candidate * a_rhs )
	{
		return a_lhs->m_val > a_rhs->m_val;
	}
};

void empty_ptr_list( std::list<cv_candidate*> & a_list )
{
	for( std::list<cv_candidate*>::iterator l_i=a_list.begin(); l_i!=a_list.end(); ++l_i )
	{
		delete (*l_i);
	}
	a_list.clear();
}

#endif

template <int I>
Ipp16u get_16bit_eng_bitcount( __m128i * ap_pix1, __m128i * ap_pix2, const __m128i & a_zero )
{
	return	get_16bit_eng_bitcount<I-1>(ap_pix1,ap_pix2,a_zero)+
			bits_unset_in_16bit[_mm_movemask_epi8(_mm_cmpeq_epi8(_mm_and_si128(ap_pix1[I-1],ap_pix2[I-1]),a_zero))];
}

template <>
Ipp16u get_16bit_eng_bitcount<1>( __m128i * ap_pix1, __m128i * ap_pix2, const __m128i & a_zero )
{
	return	bits_unset_in_16bit[_mm_movemask_epi8(_mm_cmpeq_epi8(_mm_and_si128(*ap_pix1,*ap_pix2),a_zero))];
}

template <int I>
Ipp16u energy_bitcount( Ipp8u * ap_pix1, Ipp8u * ap_pix2, const __m128i & a_zero )
{
	return get_16bit_eng_bitcount<I>((__m128i*)ap_pix1,(__m128i*)ap_pix2,a_zero);
}

struct cv_dot_list
{
	Ipp16u m_ind;
	Ipp16u m_clu;
	Ipp16u m_cla;

	Ipp8u * mp_sta;
	Ipp8u * mp_bit;

	cv_dot_list * mp_nxt;
	cv_dot_list * mp_flw;
};

/**************************** forward declarations **************************/

/********************************** classes *********************************/
/**
 * \class cv_dot_template
 * \brief This class provides functionallity for very fast gradient based template matching.
 * \It is published in CVPR 2010, San Francisco:
 * \"Dominant Orientation Templates for Real-Time Detection of Texture-Less Objects"
 * \author Stefan Hinterstoisser
 * \version 1.0
 * \date 2010
 * @author Stefan Hinterstoisser Copyright 2010 Munich
 */
template <int M, int N, int S, int G>
class cv_dot_template
{
public:
	cv_dot_template( float a_min_gradient_norm );
	~cv_dot_template();

	int get_width( void ){ return  S*M; }
	int get_height( void ){ return  S*N; }
	int get_sampling( void ){ return S; }
	int get_classes( void ){ return m_classes; }
	int get_templates( void ){ return m_templates; }
	bool is_clustered( void ){ return m_clustered; }

	void add_class( void ){ ++m_classes; }
	std::vector<CvMat*> & get_rec( void ){ return m_rec; }
	std::vector<IplImage*> & get_cnt( void ){ return m_cnt; }

	std::pair<Ipp8u*,Ipp32f*> compute_gradients( IplImage * ap_img, int a_num_of_gradients=1 );

	std::list<cv_candidate*> * online_process( Ipp8u * ap_img, int a_thres, int a_width, int a_height );
	std::list<cv_candidate*> * process( Ipp8u * ap_img, int a_thres, int a_width, int a_height );

	cv_candidate * online_comp( Ipp8u * ap_pix, int a_thres );
	cv_candidate * comp( Ipp8u * ap_pix, int a_thres );

	void calibrate( IplImage * ap_image );

	void create_bit_list_fast( IplImage * ap_img, int a_row, int a_col, int a_grad_num, float a_thres );
	void online_create_bit_list_fast( IplImage * ap_img, CvMat * ap_rec, int a_class, int a_grad_num, float a_thres );

	void cluster_heu( int a_max );

	void clear_clu_list( void );
	void clear_bit_list( void );
	void clear_rec_list( void );
	void clear_cnt_list( void );

	void render(	IplImage * ap_img, IplImage * ap_msk, int a_row,
					int a_col, int a_r=0, int a_b=255, int a_c=0 );

	bool append( std::string a_name );
	bool save( std::string a_name );
	bool load( std::string a_name );

protected:

	Ipp8u * create_template_fast( IplImage * ap_img, int a_grad_num, float a_thres );

	IplImage * get_contour( IplImage * ap_image );

	Ipp16u cluster_bitcount( Ipp8u * ap_pix1, Ipp8u * ap_pix2 );
	Ipp16u bitsets_bitcount( Ipp8u * ap_pix1 );

	void oring( Ipp8u * ap_pix1, Ipp8u * ap_pix2, Ipp8u * ap_res );
	void add_bit_list( Ipp8u * ap_bit, int a_class );
	void transfer_bit_list( Ipp8u * ap_mem );
	void compute_threshold( void );

	void shift_init( Ipp8u * ap_img, Ipp8u * ap_pix, int a_width, int a_height );
	void shift_right( Ipp8u * ap_img, Ipp8u * ap_list, int a_width, int a_row, int a_col );
	void shift_down( Ipp8u * ap_img, Ipp8u * ap_list, int a_width, int a_row, int a_col );
	void shift_copy( Ipp8u * ap_pix_old, Ipp8u * ap_pix_new );

	std::ofstream & write( std::ofstream & a_os );
	std::ifstream & read( std::ifstream & a_is );
	std::ifstream & append( std::ifstream & a_is );

protected:

	int iterated_bitcount( Ipp16u a_n )
	{
		int l_count=0;

		while(a_n)
		{
			l_count += a_n & 0x1u;
			a_n >>= 1;
		}
		return l_count;
	}

	void compute_bits_set_in_16bit( void )
	{
		for( unsigned int l_i=0; l_i<(0x1u<<16); ++l_i )
		{
			bits_set_in_16bit[l_i] = iterated_bitcount(l_i);
		}
		for( unsigned int l_i=0; l_i<(0x1u<<16); ++l_i )
		{
			bits_unset_in_16bit[l_i] = 16-iterated_bitcount(l_i);
		}
		return;
	}

protected:

	int m_width;
	int m_height;
	int m_classes;
	int m_templates;

	bool m_clustered;

	const int m_elem;
	const int m_bins;
	const float	m_min_norm;

	Ipp8u	*	mp_mem;

	std::vector<CvMat*> m_rec;
	std::vector<IplImage*> m_cnt;

	cv_dot_list * mp_start;
	cv_dot_list * mp_cluster;
};

/********************************** methods *********************************/

template <int M, int N, int S, int G>
cv_dot_template<M,N,S,G>::cv_dot_template( float a_min_gradient_norm ):
m_elem(max(16,((M*N-1)/16+1)*16)),
m_min_norm(a_min_gradient_norm),
m_clustered(false),
m_templates(0),
m_classes(0),
m_bins(8)
{
	mp_cluster = NULL;
	mp_start = NULL;

	//std::cerr << m_elem << "," << ((M*N-1)/16+1) << std::endl;

	mp_mem = ippsMalloc_8u(m_elem*MAX_DOT_TEM_NUM);
	ippsSet_8u(0,mp_mem,m_elem*MAX_DOT_TEM_NUM);

	compute_bits_set_in_16bit();
}

template <int M, int N, int S, int G>
cv_dot_template<M,N,S,G>::~cv_dot_template()
{
	this->clear_clu_list();
	this->clear_bit_list();
	this->clear_rec_list();
	this->clear_cnt_list();

	ippsFree(mp_mem);
}

template <int M, int N, int S, int G>
void cv_dot_template<M,N,S,G>::add_bit_list( Ipp8u * ap_bit, int a_class )
{
	Ipp8u * lp_bit = ippsMalloc_8u(m_elem);

	m_clustered = false;

	if( mp_start == NULL )
	{
		mp_start = new cv_dot_list;

		mp_start->m_ind  = 1;
		mp_start->m_clu  = 0;
		mp_start->m_cla  = a_class;
		mp_start->mp_nxt = NULL;
		mp_start->mp_bit = lp_bit;

		ippsCopy_8u(ap_bit,mp_start->mp_bit,m_elem);

		++m_templates;
	}
	else
	{
		cv_dot_list * lp_bit_list = new cv_dot_list;
		cv_dot_list * lp_cur = mp_start;

		while( lp_cur->mp_nxt != NULL )
		{
			lp_cur = lp_cur->mp_nxt;
		}
		lp_bit_list->m_ind	= lp_cur->m_ind+1;
		lp_bit_list->m_clu	= 0;
		lp_bit_list->m_cla	= a_class;
		lp_bit_list->mp_nxt = NULL;
		lp_bit_list->mp_bit = lp_bit;

		ippsCopy_8u(ap_bit,lp_bit_list->mp_bit,m_elem);

		lp_cur->mp_nxt = lp_bit_list;
		++m_templates;
	}
}

template <int M, int N, int S, int G>
void cv_dot_template<M,N,S,G>::transfer_bit_list( Ipp8u * ap_mem )
{
	Ipp8u * lp_end = ap_mem+m_elem*MAX_DOT_TEM_NUM;

#ifndef NO_CLUSTERING

	Ipp8u * lp_mem = ap_mem;

	cv_dot_list * lp_clu = mp_cluster;

	while( lp_clu != NULL )
	{
		if( lp_mem == lp_end )
		{
			std::cerr << "error: out of memory..." << std::endl;
			return;
		}
		ippsCopy_8u(lp_clu->mp_bit,lp_mem,m_elem);
		lp_clu->mp_sta=lp_mem;
		lp_mem+=m_elem;

		cv_dot_list * lp_cur = lp_clu->mp_flw;

		while( lp_cur != NULL )
		{
			if( lp_mem == lp_end )
			{
				std::cerr << "error: out of memory..." << std::endl;
				return;
			}
			ippsCopy_8u(lp_cur->mp_bit,lp_mem,m_elem);
			lp_cur->mp_sta=lp_mem;
			lp_mem+=m_elem;

			lp_cur = lp_cur->mp_flw;
		}
		lp_clu = lp_clu->mp_nxt;
	}

#else
	Ipp8u * lp_mem = ap_mem;

	cv_dot_list * lp_cur = mp_start;

	while( lp_cur != NULL )
	{
		if( lp_mem == lp_end )
		{
			std::cerr << "error: out of memory..." << std::endl;
			return;
		}
		ippsCopy_8u(lp_cur->mp_bit,lp_mem,m_elem);
		lp_cur->mp_sta=lp_mem;
		lp_mem+=m_elem;

		lp_cur = lp_cur->mp_nxt;
	}
#endif
	return;
}

template <int M, int N, int S, int G>
void cv_dot_template<M,N,S,G>::clear_clu_list( void )
{
	cv_dot_list * lp_cur = mp_cluster;

	while( lp_cur != NULL )
	{
		cv_dot_list * lp_del = lp_cur;
		lp_cur = lp_cur->mp_nxt;
		ippsFree(lp_del->mp_bit);
		delete lp_del;
	}
	mp_cluster = NULL;
}

template <int M, int N, int S, int G>
void cv_dot_template<M,N,S,G>::clear_bit_list( void )
{
	cv_dot_list * lp_cur = mp_start;

	while( lp_cur != NULL )
	{
		cv_dot_list * lp_del = lp_cur;
		lp_cur = lp_cur->mp_nxt;
		ippsFree(lp_del->mp_bit);
		delete lp_del;
	}
	mp_start = NULL;

	m_templates = 0;
	m_classes = 0;
}

template <int M, int N, int S, int G>
void cv_dot_template<M,N,S,G>::clear_rec_list( void )
{
	for( int l_i=0; l_i<m_rec.size(); ++l_i )
	{
		cvReleaseMat(&m_rec[l_i]);
	}
	m_rec.clear();
}

template <int M, int N, int S, int G>
void cv_dot_template<M,N,S,G>::clear_cnt_list( void )
{
	for( int l_i=0; l_i<m_cnt.size(); ++l_i )
	{
		cvReleaseImage(&m_cnt[l_i]);
	}
	m_cnt.clear();
}

template <int M, int N, int S, int G>
void cv_dot_template<M,N,S,G>::cluster_heu( int a_max )
{
	if( m_templates == 0 ) return;

	this->clear_clu_list();

	mp_cluster = NULL;

	cv_dot_list * lp_cur = NULL;

	lp_cur = mp_start;

	while( lp_cur != NULL )
	{
		lp_cur->m_clu = 0;
		lp_cur->mp_flw = NULL;
		lp_cur = lp_cur->mp_nxt;
	}
	lp_cur = mp_cluster;

	int  l_cluster_number = 1;
	bool l_flag = true;

	while( l_flag )
	{
		int l_max = -1;
		l_flag = false;

		cv_dot_list * lp_cur0 = mp_start;
		cv_dot_list * lp_cur1 = mp_start;

		while( lp_cur0 != NULL )
		{
			if( lp_cur0->m_clu == 0 )
			{
				Ipp16u l_val = bitsets_bitcount(lp_cur0->mp_bit);

				if( l_val > l_max )
				{
					l_max	= l_val;
					lp_cur1 = lp_cur0;
				}
				l_flag = true;
			}
			lp_cur0 = lp_cur0->mp_nxt;
		}
		if( lp_cur1->m_clu == 0 )
		{
			bool l_end = false;
			int l_counter = 1;

			cv_dot_list * lp_cluster = new cv_dot_list;

			lp_cur1->m_clu		= l_cluster_number;
			lp_cluster->m_clu	= l_cluster_number;
			lp_cluster->m_ind	= l_cluster_number;
			lp_cluster->mp_bit  = ippsMalloc_8u(m_elem);
			lp_cluster->mp_flw	= lp_cur1;
			lp_cluster->mp_nxt	= NULL;

			ippsCopy_8u(lp_cur1->mp_bit,lp_cluster->mp_bit,m_elem);

			cv_dot_list * lp_ind1 = lp_cur1;

			while( l_counter < a_max && l_end == false )
			{
				cv_dot_list * lp_cur2 = mp_start;
				cv_dot_list * lp_ind2 = NULL;

				int l_min = m_elem*m_bins;

				l_end = true;

				while( lp_cur2 != NULL )
				{
					if( lp_cur2->m_clu == 0 )
					{
						Ipp16u l_val = cluster_bitcount(lp_cluster->mp_bit,lp_cur2->mp_bit);

						if( l_val < l_min )
						{
							l_min	= l_val;
							l_end	= false;
							lp_ind2	= lp_cur2;
						}
					}
					lp_cur2 = lp_cur2->mp_nxt;
				}
				if( lp_ind2 != NULL )
				{
					ippsOr_8u_I(lp_ind2->mp_bit,lp_cluster->mp_bit,m_elem);

					lp_ind2->m_clu	= l_cluster_number;
					lp_ind1->mp_flw	= lp_ind2;
					lp_ind1			= lp_ind2;

					++l_counter;
				}
			}
			if( mp_cluster == NULL )
			{
				mp_cluster = lp_cluster;
			}
			else
			{
				cv_dot_list * lp_cur = mp_cluster;

				while( lp_cur->mp_nxt != NULL )
				{
					lp_cur = lp_cur->mp_nxt;
				}
				lp_cur->mp_nxt = lp_cluster;
			}
			++l_cluster_number;
		}
	}
	std::cerr << "*******************************************" << std::endl;
	std::cerr << std::endl << "clusters: " << l_cluster_number << std::endl;
	std::cerr << "*******************************************" << std::endl;

	m_clustered = true;

	this->transfer_bit_list(mp_mem);

	return;
}

template <int M, int N, int S, int G>
void cv_dot_template<M,N,S,G>::calibrate( IplImage * ap_image )
{
	float l_max;

	IppiSize l_size;

	l_size.height = 50;
	l_size.width  = 50;

	IplImage * lp_sobel_dx = cvCreateImage(cvGetSize(ap_image),IPL_DEPTH_32F,1);
	IplImage * lp_sobel_dy = cvCreateImage(cvGetSize(ap_image),IPL_DEPTH_32F,1);
	IplImage * lp_sobel_mg = cvCreateImage(cvGetSize(ap_image),IPL_DEPTH_32F,1);

	cvSobel(ap_image,lp_sobel_dx,1,0,3);
	cvSobel(ap_image,lp_sobel_dy,0,1,3);

	cvCartToPolar(lp_sobel_dx,lp_sobel_dy,lp_sobel_mg);

	ippiMax_32f_C1R(((float*)lp_sobel_mg->imageData)+
					(lp_sobel_mg->height/2-l_size.height/2)*lp_sobel_mg->width+
					lp_sobel_mg->width/2-l_size.width/2,
					lp_sobel_mg->widthStep,l_size,&l_max);

	m_min_norm = l_max*2;

	std::cerr << "calibration: max magnitude: " << l_max;
	std::cerr << " -> " << m_min_norm << "       " << std::endl;

	cvReleaseImage(&lp_sobel_dx);
	cvReleaseImage(&lp_sobel_dy);
	cvReleaseImage(&lp_sobel_mg);
}

template <int M, int N, int S, int G>
void cv_dot_template<M,N,S,G>::shift_right(	Ipp8u * ap_img, Ipp8u * ap_list,
											int a_width, int a_row, int a_col )
{
	int l_off2 = a_col+N+a_width*a_row;
	int l_off1 = N-1;

	ippsCopy_8u(ap_list+1,ap_list,m_elem-1);

	for( int l_i=0; l_i<M; ++l_i )
	{
		ap_list[l_off1] = ap_img[l_off2];

		l_off2+=a_width;
		l_off1+=N;
	}
}

template <int M, int N, int S, int G>
void cv_dot_template<M,N,S,G>::shift_down(	Ipp8u * ap_img, Ipp8u * ap_list,
											int a_width, int a_row, int a_col )
{
	int l_off = a_col+a_width*(a_row+M);
	int l_sta = N*(M-1);
	int l_end = l_sta+N;

	ippsCopy_8u(ap_list+N,ap_list,m_elem-N);

	for( int l_i=l_sta; l_i<l_end; ++l_i )
	{
		ap_list[l_i] = ap_img[l_off];

		++l_off;
	}
}

template <int M, int N, int S, int G>
void cv_dot_template<M,N,S,G>::shift_copy( Ipp8u * ap_pix_old, Ipp8u * ap_pix_new )
{
	ippsCopy_8u(ap_pix_old,ap_pix_new,N*M);
}

template <int M, int N, int S, int G>
void cv_dot_template<M,N,S,G>::shift_init(	Ipp8u * ap_img, Ipp8u * ap_pix_new,
											int a_width, int a_height )
{
	IppiSize l_size;

	l_size.width  = N;
	l_size.height = M;

	ippiCopy_8u_C1R(ap_img,a_width,ap_pix_new,N,l_size);
}

template <int M, int N, int S, int G>
cv_candidate * cv_dot_template<M,N,S,G>::comp( Ipp8u * ap_pix, int a_thres )
{
	register __m128i l_zero = _mm_setzero_si128();

#ifndef NO_CLUSTERING

	cv_candidate * lp_candidate = new cv_candidate;

	lp_candidate->m_ind = 0;
	lp_candidate->m_clu = 0;
	lp_candidate->m_cla = 0;
	lp_candidate->m_val = 0;

	Ipp16u l_max = 0;

	cv_dot_list * lp_clu = mp_cluster;

	while( lp_clu != NULL )
	{
		int l_clu_val = energy_bitcount<(M*N-1)/16+1>(lp_clu->mp_sta,ap_pix,l_zero);

		if( l_clu_val >= a_thres && l_clu_val > l_max )
		{
			cv_dot_list * lp_cur = lp_clu->mp_flw;

			while( lp_cur != NULL )
			{
				int l_cur_val = energy_bitcount<(M*N-1)/16+1>(lp_cur->mp_sta,ap_pix,l_zero);

				if( l_cur_val > l_max && l_cur_val >= a_thres )
				{
					l_max  = l_cur_val;

					lp_candidate->m_ind = lp_cur->m_ind;
					lp_candidate->m_clu = lp_cur->m_clu;
					lp_candidate->m_cla = lp_cur->m_cla;
					lp_candidate->m_val = l_cur_val;
				}
				lp_cur = lp_cur->mp_flw;
			}
		}
		lp_clu = lp_clu->mp_nxt;
	}
#else
	cv_candidate * lp_candidate = new cv_candidate;

	lp_candidate->m_ind = 0;
	lp_candidate->m_clu = 0;
	lp_candidate->m_cla = 0;
	lp_candidate->m_val = 0;

	Ipp16u l_max = 0;

	cv_dot_list * lp_cur = mp_start;

	while( lp_cur != NULL )
	{
		Ipp16u l_cur_val = energy_bitcount<(M*N-1)/16+1>(lp_cur->mp_sta,ap_pix,l_zero);

		if( l_cur_val >= a_thres && l_cur_val > l_max )
		{
			bool l_flag = false;

			l_max  = l_cur_val;

			lp_candidate->m_ind = lp_cur->m_ind;
			lp_candidate->m_clu = lp_cur->m_clu;
			lp_candidate->m_cla = lp_cur->m_cla;
			lp_candidate->m_val = l_cur_val;
		}
		lp_cur = lp_cur->mp_nxt;
	}
#endif
	return lp_candidate;
}

template <int M, int N, int S, int G>
cv_candidate * cv_dot_template<M,N,S,G>::online_comp( Ipp8u * ap_pix, int a_thres )
{
	register __m128i l_zero = _mm_setzero_si128();

	cv_candidate * lp_candidate = new cv_candidate;

	lp_candidate->m_ind = 0;
	lp_candidate->m_clu = 0;
	lp_candidate->m_cla = 0;
	lp_candidate->m_val = 0;

	Ipp16u l_max = 0;

	cv_dot_list * lp_cur = mp_start;

	while( lp_cur != NULL )
	{
		Ipp16u l_cur_val = energy_bitcount<(M*N-1)/16+1>(lp_cur->mp_bit,ap_pix,l_zero);

		if( l_cur_val >= a_thres && l_cur_val > l_max )
		{
			bool l_flag = false;

			lp_candidate->m_ind = lp_cur->m_ind;
			lp_candidate->m_clu = lp_cur->m_clu;
			lp_candidate->m_cla = lp_cur->m_cla;
			lp_candidate->m_val = l_max = l_cur_val;
		}
		lp_cur = lp_cur->mp_nxt;
	}
	return lp_candidate;
}

template <int M, int N, int S, int G>
void cv_dot_template<M,N,S,G>::oring( Ipp8u * ap_pix1, Ipp8u * ap_pix2, Ipp8u * ap_res )
{
	__m128i * lp_val0 = (__m128i*)ap_res;
	__m128i * lp_pix1 = (__m128i*)ap_pix1;
	__m128i * lp_pix2 = (__m128i*)ap_pix2;

	for( int l_i=0; l_i<m_elem/16; ++l_i )
	{
		lp_val0[l_i] = _mm_or_si128(lp_pix1[l_i],lp_pix2[l_i]);
	}
}

template <int M, int N, int S, int G>
Ipp16u cv_dot_template<M,N,S,G>::cluster_bitcount(Ipp8u * ap_pix1, Ipp8u * ap_pix2 )
{
	int l_val1=0;
	int l_val2=0;
	int l_val3=0;

	Ipp8u * lp_res = ippsMalloc_8u(m_elem);

	for( int l_i=0; l_i<m_elem; ++l_i )
	{
		l_val1+=bits_set_in_16bit[ap_pix1[l_i]];
		l_val2+=bits_set_in_16bit[ap_pix2[l_i]];
	}
	oring(ap_pix1,ap_pix2,lp_res);

	for( int l_i=0; l_i<m_elem; ++l_i )
	{
		l_val3+=bits_set_in_16bit[lp_res[l_i]];
	}
	ippsFree(lp_res);

	return max(l_val3-l_val1,l_val3-l_val2);
}

template <int M, int N, int S, int G>
Ipp16u cv_dot_template<M,N,S,G>::bitsets_bitcount( Ipp8u * ap_pix1 )
{
	int l_val1=0;

	for( int l_i=0; l_i<m_elem; ++l_i )
	{
		l_val1+=bits_set_in_16bit[ap_pix1[l_i]];
	}
	return l_val1;
}

template <int M, int N, int S, int G>
std::list<cv_candidate*> * cv_dot_template<M,N,S,G>::process(	Ipp8u * ap_img,
																int a_thres,
																int a_width,
																int a_height )
{
	std::list<cv_candidate*> * lp_list = new std::list<cv_candidate*>[m_classes];

	Ipp8u * lp_pix_row = ippsMalloc_8u(m_elem);
	Ipp8u * lp_pix_col = ippsMalloc_8u(m_elem);

	ippsSet_8u(0,lp_pix_row,m_elem);
	ippsSet_8u(0,lp_pix_col,m_elem);

	this->shift_init(ap_img,lp_pix_row,a_width,a_height);
	this->shift_copy(lp_pix_row,lp_pix_col);

	for( int l_r=0; l_r<a_height-M; ++l_r )
	{
		for( int l_c=0; l_c<a_width-N; ++l_c )
		{
			cv_candidate * lp_candidate = this->comp(lp_pix_col,a_thres);

			if( lp_candidate->m_ind != 0 )
			{
				lp_candidate->m_col = l_c*S;
				lp_candidate->m_row = l_r*S;

				lp_list[lp_candidate->m_cla].push_back(lp_candidate);
			}
			else
			{
				delete lp_candidate;
			}
			this->shift_right(ap_img,lp_pix_col,a_width,l_r,l_c);
		}
		this->shift_down(ap_img,lp_pix_row,a_width,l_r,0);
		this->shift_copy(lp_pix_row,lp_pix_col);
	}
	ippsFree(lp_pix_row);
	ippsFree(lp_pix_col);

	return lp_list;
}

template <int M, int N, int S, int G>
std::list<cv_candidate*> * cv_dot_template<M,N,S,G>::online_process(	Ipp8u * ap_img,
																		int a_thres,
																		int a_width,
																		int a_height )
{
	std::list<cv_candidate*> * lp_list = new std::list<cv_candidate*>[m_classes];

	Ipp8u * lp_pix_row = ippsMalloc_8u(m_elem);
	Ipp8u * lp_pix_col = ippsMalloc_8u(m_elem);

	ippsSet_8u(0,lp_pix_row,m_elem);
	ippsSet_8u(0,lp_pix_col,m_elem);

	this->shift_init(ap_img,lp_pix_row,a_width,a_height);
	this->shift_copy(lp_pix_row,lp_pix_col);

	for( int l_r=0; l_r<a_height-M; ++l_r )
	{
		for( int l_c=0; l_c<a_width-N; ++l_c )
		{
			cv_candidate * lp_candidate = this->online_comp(lp_pix_col,a_thres);

			if( lp_candidate->m_ind != 0 )
			{
				lp_candidate->m_col = l_c*S;
				lp_candidate->m_row = l_r*S;

				lp_list[lp_candidate->m_cla].push_back(lp_candidate);
			}
			else
			{
				delete lp_candidate;
			}
			this->shift_right(ap_img,lp_pix_col,a_width,l_r,l_c);
		}
		this->shift_down(ap_img,lp_pix_row,a_width,l_r,0);
		this->shift_copy(lp_pix_row,lp_pix_col);
	}
	ippsFree(lp_pix_row);
	ippsFree(lp_pix_col);

	return lp_list;
}

template <int M, int N, int S, int G>
Ipp8u * cv_dot_template<M,N,S,G>::create_template_fast(	IplImage * ap_img,
														int a_grad_num,
														float a_thres )
{
	//num_x+1 because we have to make it translation invariant...
	if( ap_img->width != S*(N+1)+2 || ap_img->height != S*(M+1)+2 )
	{
		printf("create_template_fast: error since the width and height are not correct!" );
		return NULL;
	}
	IplImage * lp_sobel_dx = cvCreateImage(cvGetSize(ap_img),IPL_DEPTH_32F,1);
	IplImage * lp_sobel_dy = cvCreateImage(cvGetSize(ap_img),IPL_DEPTH_32F,1);
	IplImage * lp_sobel_mg = cvCreateImage(cvGetSize(ap_img),IPL_DEPTH_32F,1);
	IplImage * lp_sobel_ag = cvCreateImage(cvGetSize(ap_img),IPL_DEPTH_32F,1);

	cvSobel(ap_img,lp_sobel_dx,1,0,3);
	cvSobel(ap_img,lp_sobel_dy,0,1,3);

	cvCartToPolar(lp_sobel_dx,lp_sobel_dy,lp_sobel_mg,lp_sobel_ag,1);

	Ipp32f * lp_strn_tmp	= ippsMalloc_32f(m_elem);
	Ipp8u * lp_peak_tmp		= ippsMalloc_8u(m_elem);
	Ipp32f * lp_strn_ptr	= lp_strn_tmp;
	Ipp8u * lp_peak_ptr		= lp_peak_tmp;

	ippsSet_32f(0,lp_strn_tmp,m_elem);
	ippsSet_8u(0,lp_peak_tmp,m_elem);

	//because we start at the midpoint of the bin...
	int l_off_x=1+S/2;
	int l_off_y=1+S/2;

	int l_mstep = lp_sobel_mg->widthStep/sizeof(float);
	int l_istep = ap_img->widthStep/sizeof(float);

	IppiSize l_size;

	l_size.height = S;
	l_size.width  = S;

	float l_divisor = 180.0/((m_bins-1));

	float l_gmax_gra=0;
	float l_lmax_gra=0;

	for( int l_r=0; l_r<M; ++l_r )
	{
		for( int l_c=0; l_c<N; ++l_c )
		{
			std::vector<int> l_xcord;
			std::vector<int> l_ycord;
			std::vector<float> l_val;

			for( int l_y=-S/2; l_y<=S/2; l_y+=S/2 )
			{
				int l_yall = l_off_y+l_y;

				for( int l_x=-S/2; l_x<=S/2; l_x+=S/2 )
				{
					int l_xall		= l_off_x+l_x;
					int l_counter	= 0;

					ippiMax_32f_C1R(((float*)lp_sobel_mg->imageData)+l_yall*l_mstep+l_xall,
									lp_sobel_mg->widthStep,
									l_size,
									&l_lmax_gra);

					*lp_strn_ptr += l_lmax_gra;

					if( l_lmax_gra > l_gmax_gra ) l_gmax_gra=l_lmax_gra;

					while( true )
					{
						float l_max;

						int l_inx;
						int l_iny;

						ippiMaxIndx_32f_C1R(((float*)lp_sobel_mg->imageData)+l_yall*l_mstep+l_xall,
											lp_sobel_mg->widthStep,
											l_size,
											&l_max,
											&l_inx,
											&l_iny);

						if( l_lmax_gra < m_min_norm )
						{
							*lp_peak_ptr |= 1 << (m_bins-1);

							break;
						}
						if( l_lmax_gra*a_thres > l_max || l_counter >= a_grad_num ) break;
						//if( l_lmax_gra-m_min_norm > l_max || l_counter >= a_grad_num ) break;
						//if( l_lmax_gra-m_min_norm > l_max  ) break;
						//if( l_counter >= 1 ) break;

						++l_counter;

						int l_ang1 = (CV_IMAGE_ELEM(lp_sobel_ag,float,l_iny+l_yall,l_inx+l_xall)-0.5);
						int l_bin1 = (l_ang1>=180.0?l_ang1-180.0:l_ang1)/l_divisor;

						*lp_peak_ptr |= 1 << l_bin1;

						l_xcord.push_back(l_inx+l_xall);
						l_ycord.push_back(l_iny+l_yall);
						l_val.push_back(l_max);

						CV_IMAGE_ELEM(lp_sobel_mg,float,l_iny+l_yall,l_inx+l_xall) = -1;
					}
					for( int l_k=0; l_k<l_val.size(); ++l_k )
					{
						CV_IMAGE_ELEM(lp_sobel_mg,float,l_ycord[l_k],l_xcord[l_k]) = l_val[l_k];
					}
					l_xcord.clear();
					l_ycord.clear();
					l_val.clear();
				}
			}
			++lp_strn_ptr;
			++lp_peak_ptr;
			l_off_x += S;
		}
		l_off_y += S;
		l_off_x  = S/2+1;
	}
	IppiSize l_patch_size;

	l_patch_size.width  = N;
	l_patch_size.height = M;

	//IplImage * lp_mask = cvCreateImage(cvSize(N*S,M*S),IPL_DEPTH_32F,1);
	//cvSet(lp_mask,cvRealScalar(255));

	for( int l_i=0; l_i<N*M-G; ++l_i )
	{
		Ipp32f l_min;
		int l_x;
		int l_y;

		ippiMinIndx_32f_C1R(lp_strn_tmp,
							N*sizeof(Ipp32f),
							l_patch_size,
							&l_min,
							&l_x,
							&l_y);

		//for( int l_r=0; l_r<S; ++l_r )
		//{
		//	for( int l_c=0; l_c<S; ++l_c )
		//	{
		//		CV_IMAGE_ELEM(lp_mask,float,l_r+l_y*S,l_c+l_x*S) = 0;
		//	}
		//}
		lp_strn_tmp[l_y*N+l_x] = 10e100;
		lp_peak_tmp[l_y*N+l_x] = 0;
	}
	//cv::cv_show_image(lp_mask);
	//cvReleaseImage(&lp_mask);

	cvReleaseImage(&lp_sobel_dx);
	cvReleaseImage(&lp_sobel_dy);
	cvReleaseImage(&lp_sobel_mg);
	cvReleaseImage(&lp_sobel_ag);
	ippFree(lp_strn_tmp);

	return lp_peak_tmp;
}

template <int M, int N, int S, int G>
std::pair<Ipp8u*,Ipp32f*> cv_dot_template<M,N,S,G>::compute_gradients( IplImage * ap_img, int a_num_of_gradients )
{
	//-2 because there are no valid gradient at the borders...
	m_width  = (ap_img->width-2)/S;
	m_height = (ap_img->height-2)/S;

	IplImage * lp_sobel_dx = cvCreateImage(cvGetSize(ap_img),IPL_DEPTH_32F,1);
	IplImage * lp_sobel_dy = cvCreateImage(cvGetSize(ap_img),IPL_DEPTH_32F,1);
	IplImage * lp_sobel_mg = cvCreateImage(cvGetSize(ap_img),IPL_DEPTH_32F,1);
	IplImage * lp_sobel_ag = cvCreateImage(cvGetSize(ap_img),IPL_DEPTH_32F,1);

	cvSobel(ap_img,lp_sobel_dx,1,0,3);
	cvSobel(ap_img,lp_sobel_dy,0,1,3);

	cvCartToPolar(lp_sobel_dx,lp_sobel_dy,lp_sobel_mg,lp_sobel_ag,1);

	Ipp32f * lp_peak_mag = ippsMalloc_32f(lp_sobel_mg->width*lp_sobel_mg->height);
	Ipp8u * lp_peak_img = ippsMalloc_8u(m_height*m_width);
	Ipp8u * lp_peak_ptr = lp_peak_img;

	ippsCopy_32f((Ipp32f*)lp_sobel_mg->imageData,lp_peak_mag,lp_sobel_mg->width*lp_sobel_mg->height);

	ippsSet_8u(0,lp_peak_img,m_width*m_height);

	int l_off_y=1;
	int l_off_x=1;

	int l_mstep = lp_sobel_mg->widthStep/sizeof(float);
	int l_istep = ap_img->widthStep/sizeof(float);

	IppiSize l_size;

	l_size.height = S;
	l_size.width  = S;

	float l_divisor = 180.0/((m_bins-1));

	for( int l_r=0; l_r<m_height; ++l_r )
	{
		for( int l_c=0; l_c<m_width; ++l_c )
		{
			int l_counter = 0;

			while( l_counter < a_num_of_gradients )
			{
				float l_int_min;
				float l_int_max;
				float l_max;

				int l_inx;
				int l_iny;

				ippiMaxIndx_32f_C1R(((float*)lp_sobel_mg->imageData)+l_off_y*l_mstep+l_off_x,
									lp_sobel_mg->widthStep,
									l_size,
									&l_max,
									&l_inx,
									&l_iny);

				if( l_max < m_min_norm )
				{
					*lp_peak_ptr |= 1 << (m_bins-1);
					break;
				}
				else
				{
					int l_ang = (CV_IMAGE_ELEM(lp_sobel_ag,float,l_iny+l_off_y,l_inx+l_off_x)-0.5);
					int l_bin = (l_ang>=180.0?l_ang-180.0:l_ang)/l_divisor;

					*lp_peak_ptr |= 1 << l_bin;
				}
				CV_IMAGE_ELEM(lp_sobel_mg,float,l_iny+l_off_y,l_inx+l_off_x) = -1;

				++l_counter;
			}
			++lp_peak_ptr;
			l_off_x += S;
		}
		l_off_y += S;
		l_off_x = 1;
	}
	cvReleaseImage(&lp_sobel_dx);
	cvReleaseImage(&lp_sobel_dy);
	cvReleaseImage(&lp_sobel_mg);
	cvReleaseImage(&lp_sobel_ag);

	return std::pair<Ipp8u*,Ipp32f*>(lp_peak_img,lp_peak_mag);
}

template <int M, int N, int S, int G>
void cv_dot_template<M,N,S,G>::create_bit_list_fast(	IplImage * ap_img,
														int a_row,
														int a_col,
														int a_grad_num,
														float a_thres )
{
	CvPoint2D32f l_src[4];
	CvPoint2D32f l_dst[4];

	//num_x+1 because we have to make it translation invariant...
	IplImage * lp_img = cvCreateImage(cvSize((N+1)*S+2,(M+1)*S+2),IPL_DEPTH_32F,1);

	CvMat * lp_mat = cvCreateMat(3,3,CV_32F);
	CvMat * lp_rot = cvCreateMat(2,3,CV_32F);
	CvMat * lp_dst = cvCreateMat(3,3,CV_32F);
	CvMat * lp_tmp = cvCreateMat(3,3,CV_32F);

	int l_sizex = S*(N+1)/2;
	int l_sizey = S*(M+1)/2;

	int l_num = 3;
	int l_sampling = 7;

	for( int l_s=-l_num*l_sampling; l_s<=l_num*l_sampling; l_s+=l_sampling )
	{
		for( int l_r=0; l_r<360; l_r+=10 )
		{
			CvPoint2D32f l_center;
			l_center.x = a_col;
			l_center.y = a_row;
			cv2DRotationMatrix(l_center,l_r,1.0,lp_rot);

			l_dst[0].x = 1-l_s;
			l_dst[0].y = 1-l_s;
			l_dst[1].x = S*(N+1)+1+l_s;
			l_dst[1].y = 1-l_s;
			l_dst[2].x = S*(N+1)+1+l_s;
			l_dst[2].y = S*(M+1)+1+l_s;
			l_dst[3].x = 1-l_s;
			l_dst[3].y = S*(M+1)+1+l_s;

			l_src[0].x = -l_sizex+a_col;
			l_src[0].y = -l_sizey+a_row;
			l_src[1].x = +l_sizex+a_col;
			l_src[1].y = -l_sizey+a_row;
	 		l_src[2].x = +l_sizex+a_col;
			l_src[2].y = +l_sizey+a_row;
			l_src[3].x = -l_sizex+a_col;
			l_src[3].y = +l_sizey+a_row;

			cvGetPerspectiveTransform(l_src,l_dst,lp_mat);

			CV_MAT_ELEM(*lp_tmp,float,0,0) = CV_MAT_ELEM(*lp_rot,float,0,0);
			CV_MAT_ELEM(*lp_tmp,float,0,1) = CV_MAT_ELEM(*lp_rot,float,0,1);
			CV_MAT_ELEM(*lp_tmp,float,0,2) = CV_MAT_ELEM(*lp_rot,float,0,2);
			CV_MAT_ELEM(*lp_tmp,float,1,0) = CV_MAT_ELEM(*lp_rot,float,1,0);
			CV_MAT_ELEM(*lp_tmp,float,1,1) = CV_MAT_ELEM(*lp_rot,float,1,1);
			CV_MAT_ELEM(*lp_tmp,float,1,2) = CV_MAT_ELEM(*lp_rot,float,1,2);
			CV_MAT_ELEM(*lp_tmp,float,2,0) = 0;
			CV_MAT_ELEM(*lp_tmp,float,2,1) = 0;
			CV_MAT_ELEM(*lp_tmp,float,2,2) = 1;

			cvMatMul(lp_mat,lp_tmp,lp_dst);
			cvWarpPerspective(ap_img,lp_img,lp_dst);

			CvMat * lp_war = cvCreateMat(3,4,CV_32F);

			CV_MAT_ELEM(*lp_war,float,0,0) = l_src[0].x;
			CV_MAT_ELEM(*lp_war,float,1,0) = l_src[0].y;
			CV_MAT_ELEM(*lp_war,float,0,1) = l_src[1].x;
			CV_MAT_ELEM(*lp_war,float,1,1) = l_src[1].y;
			CV_MAT_ELEM(*lp_war,float,0,2) = l_src[2].x;
			CV_MAT_ELEM(*lp_war,float,1,2) = l_src[2].y;
			CV_MAT_ELEM(*lp_war,float,0,3) = l_src[3].x;
			CV_MAT_ELEM(*lp_war,float,1,3) = l_src[3].y;
			CV_MAT_ELEM(*lp_war,float,2,0) = 1;
			CV_MAT_ELEM(*lp_war,float,2,1) = 1;
			CV_MAT_ELEM(*lp_war,float,2,2) = 1;
			CV_MAT_ELEM(*lp_war,float,2,3) = 1;

			cvMatMul(lp_dst,lp_war,lp_war);
			cv::cv_homogenize(lp_war);

			//cv::cv_show_image(lp_img);

			Ipp8u * lp_template = create_template_fast(lp_img,a_grad_num,a_thres);

			this->add_bit_list(lp_template,m_classes);
			m_rec.push_back(lp_war);

			IplImage * lp_cnt = get_contour(lp_img);
			m_cnt.push_back(lp_cnt);

			ippsFree(lp_template);
		}
	}
	cvReleaseImage(&lp_img);
	cvReleaseMat(&lp_mat);
	cvReleaseMat(&lp_rot);
	cvReleaseMat(&lp_dst);
	cvReleaseMat(&lp_tmp);

	++m_classes;

	std::cerr << "num of templates: " << get_templates() <<  std::endl;
}

template <int M, int N, int S, int G>
void cv_dot_template<M,N,S,G>::online_create_bit_list_fast(	IplImage * ap_img,
															CvMat * ap_rec,
															int a_class,
															int a_grad_num,
															float a_thres )
{
	CvPoint2D32f l_src[4];
	CvPoint2D32f l_dst[4];
	CvPoint2D32f l_war[4];

	int l_sizex = S*(N+1)/2;
	int l_sizey = S*(M+1)/2;

	//num_x+1 because we have to make it translation invariant...
	IplImage * lp_img = cvCreateImage(cvSize((N+1)*S+2,(M+1)*S+2),IPL_DEPTH_32F,1);

	CvMat * lp_mat = cvCreateMat(3,3,CV_32F);
	CvMat * lp_pts = cvCreateMat(3,4,CV_32F);
	CvMat * lp_wpt = cvCreateMat(3,1,CV_32F);
	CvMat * lp_pt  = cvCreateMat(3,1,CV_32F);

	l_dst[0].x = 1;
	l_dst[0].y = 1;
	l_dst[1].x = S*(N+1)+1;
	l_dst[1].y = 1;
	l_dst[2].x = S*(N+1)+1;
	l_dst[2].y = S*(M+1)+1;
	l_dst[3].x = 1;
	l_dst[3].y = S*(M+1)+1;

	l_war[0].x = CV_MAT_ELEM(*ap_rec,float,0,0);
	l_war[0].y = CV_MAT_ELEM(*ap_rec,float,1,0);
	l_war[1].x = CV_MAT_ELEM(*ap_rec,float,0,1);
	l_war[1].y = CV_MAT_ELEM(*ap_rec,float,1,1);
	l_war[2].x = CV_MAT_ELEM(*ap_rec,float,0,2);
	l_war[2].y = CV_MAT_ELEM(*ap_rec,float,1,2);
	l_war[3].x = CV_MAT_ELEM(*ap_rec,float,0,3);
	l_war[3].y = CV_MAT_ELEM(*ap_rec,float,1,3);

	CV_MAT_ELEM(*lp_pt,float,0,0) = l_sizex+1;
	CV_MAT_ELEM(*lp_pt,float,1,0) = l_sizey+1;
	CV_MAT_ELEM(*lp_pt,float,2,0) = 1;

	cvGetPerspectiveTransform(l_dst,l_war,lp_mat);
	cvMatMul(lp_mat,lp_pt,lp_wpt);
	cv::cv_homogenize(lp_wpt);

	int l_col = CV_MAT_ELEM(*lp_wpt,float,0,0);
	int l_row = CV_MAT_ELEM(*lp_wpt,float,1,0);

	l_src[0].x = -l_sizex+l_col;
	l_src[0].y = -l_sizey+l_row;
	l_src[1].x = +l_sizex+l_col;
	l_src[1].y = -l_sizey+l_row;
	l_src[2].x = +l_sizex+l_col;
	l_src[2].y = +l_sizey+l_row;
	l_src[3].x = -l_sizex+l_col;
	l_src[3].y = +l_sizey+l_row;

	cvGetPerspectiveTransform(l_src,l_dst,lp_mat);
	cvWarpPerspective(ap_img,lp_img,lp_mat);

	cvCopy(ap_rec,lp_pts);
	CV_MAT_ELEM(*lp_pts,float,0,0) -= l_col-l_sizex;
	CV_MAT_ELEM(*lp_pts,float,1,0) -= l_row-l_sizey;
	CV_MAT_ELEM(*lp_pts,float,0,1) -= l_col-l_sizex;
	CV_MAT_ELEM(*lp_pts,float,1,1) -= l_row-l_sizey;
	CV_MAT_ELEM(*lp_pts,float,0,2) -= l_col-l_sizex;
	CV_MAT_ELEM(*lp_pts,float,1,2) -= l_row-l_sizey;
	CV_MAT_ELEM(*lp_pts,float,0,3) -= l_col-l_sizex;
	CV_MAT_ELEM(*lp_pts,float,1,3) -= l_row-l_sizey;

	//cv::cv_show_image(lp_img);

	Ipp8u * lp_template = create_template_fast(	lp_img,
												a_grad_num,
												a_thres );
	this->add_bit_list(lp_template,a_class);
	m_rec.push_back(lp_pts);

	IplImage * lp_cnt = get_contour(lp_img);
	m_cnt.push_back(lp_cnt);

	ippsFree(lp_template);
	cvReleaseImage(&lp_img);
	cvReleaseMat(&lp_mat);
	cvReleaseMat(&lp_wpt);
	cvReleaseMat(&lp_pt);
}

template <int M, int N, int S, int G>
IplImage * cv_dot_template<M,N,S,G>::get_contour( IplImage * ap_img )
{
	IplImage * lp_sobel_dx = cvCreateImage(cvGetSize(ap_img),IPL_DEPTH_32F,1);
	IplImage * lp_sobel_dy = cvCreateImage(cvGetSize(ap_img),IPL_DEPTH_32F,1);
	IplImage * lp_sobel_mg = cvCreateImage(cvGetSize(ap_img),IPL_DEPTH_32F,1);
	IplImage * lp_sobel_mk1 = cvCreateImage(cvGetSize(ap_img),IPL_DEPTH_8U,1);
	IplImage * lp_sobel_mk2 = cvCreateImage(cvGetSize(ap_img),IPL_DEPTH_8U,1);
	IplImage * lp_sobel_mk3 = cvCreateImage(cvGetSize(ap_img),IPL_DEPTH_8U,1);

	cvSobel(ap_img,lp_sobel_dx,1,0,3);
	cvSobel(ap_img,lp_sobel_dy,0,1,3);

	cvCartToPolar(lp_sobel_dx,lp_sobel_dy,lp_sobel_mg);
	cvConvert(lp_sobel_mg,lp_sobel_mk2);

	cvThreshold( lp_sobel_mk2,lp_sobel_mk3,40,255,CV_THRESH_BINARY);

	//cv::cv_show_image(lp_sobel_mk3,"hallo2");
	//cvWaitKey(100);

	cvReleaseImage(&lp_sobel_mk1);
	cvReleaseImage(&lp_sobel_mk2);
	cvReleaseImage(&lp_sobel_dx);
	cvReleaseImage(&lp_sobel_dy);
	cvReleaseImage(&lp_sobel_mg);

	return lp_sobel_mk3;
}

template <int M, int N, int S, int G>
void cv_dot_template<M,N,S,G>::render(	IplImage * ap_img,
										IplImage * ap_msk,
										int a_row,
										int a_col,
										int a_r,
										int a_b,
										int a_g )
{
	for( int l_r=0; l_r<ap_msk->height; ++l_r )
	{
		for( int l_c=0; l_c<ap_msk->width; ++l_c )
		{
			if( CV_IMAGE_ELEM(ap_msk,unsigned char,l_r,l_c) == 255 )
			{
				CV_IMAGE_ELEM(ap_img,float,l_r+a_row,(l_c+a_col)*3+0) = a_g;
				CV_IMAGE_ELEM(ap_img,float,l_r+a_row,(l_c+a_col)*3+1) = a_b;
				CV_IMAGE_ELEM(ap_img,float,l_r+a_row,(l_c+a_col)*3+2) = a_r;
			}
		}
	}
}

template <int M, int N, int S, int G>
std::ofstream & cv_dot_template<M,N,S,G>::write( std::ofstream & a_os )
{
	//these functions do not really work right now....
	a_os.write((char*)&m_width,sizeof(m_width));
	a_os.write((char*)&m_height,sizeof(m_height));
	a_os.write((char*)&m_classes,sizeof(m_classes));
	a_os.write((char*)&m_templates,sizeof(m_templates));

	a_os.write((char*)&m_elem,sizeof(m_elem));
	a_os.write((char*)&m_bins,sizeof(m_bins));

	a_os.write((char*)&m_min_norm,sizeof(m_min_norm));

	for( int l_i=0; l_i<m_templates; ++l_i )
	{
		cv::cv_write(a_os,m_rec[l_i]);
		cv::cv_write(a_os,m_cnt[l_i]);
	}
	cv_dot_list * lp_cur = mp_start;

	while( lp_cur != NULL )
	{
		int l_class = lp_cur->m_cla;

		a_os.write((char*)&l_class,sizeof(l_class));

		for( int l_i=0; l_i<m_elem; ++l_i )
		{
			a_os.write((char*)&lp_cur->mp_bit[l_i],sizeof(lp_cur->mp_bit[l_i]));
		}
		lp_cur = lp_cur->mp_nxt;
	}
	return a_os;
}

template <int M, int N, int S, int G>
std::ifstream & cv_dot_template<M,N,S,G>::read( std::ifstream & a_is )
{
	//these functions do not really work right now....
	this->clear_clu_list();
	this->clear_bit_list();
	this->clear_rec_list();
	this->clear_cnt_list();

	a_is.read((char*)&m_width,sizeof(m_width));
	a_is.read((char*)&m_height,sizeof(m_height));
	a_is.read((char*)&m_classes,sizeof(m_classes));
	a_is.read((char*)&m_templates,sizeof(m_templates));

	a_is.read((char*)&m_elem,sizeof(m_elem));
	a_is.read((char*)&m_bins,sizeof(m_bins));

	a_is.read((char*)&m_min_norm,sizeof(m_min_norm));

	int l_size = m_templates;

	//set it to zero to allow add_bit_list to do its job...
	m_templates = 0;

	for( int l_i=0; l_i<l_size; ++l_i )
	{
		CvMat * lp_mat = cv::cv_read(a_is);
		IplImage * lp_cnt = cv::cv_read_img(a_is);

		m_rec.push_back(lp_mat);
		m_cnt.push_back(lp_cnt);
	}
	for( int l_j=0; l_j<l_size; ++l_j )
	{
		int l_class;

		Ipp8u * lp_bit = ippsMalloc_8u(m_elem);

		a_is.read((char*)&l_class,sizeof(l_class));

		for( int l_i=0; l_i<m_elem; ++l_i )
		{
			a_is.read((char*)&lp_bit[l_i],sizeof(lp_bit[l_i]));
		}
		this->add_bit_list(lp_bit,l_class);

		ippsFree(lp_bit);
	}
	std::cerr << "num of templates: " << this->get_templates() << std::endl;
	std::cerr << "num of classes:   " << this->get_classes() << std::endl;

	return a_is;
}

template <int M, int N, int S, int G>
std::ifstream & cv_dot_template<M,N,S,G>::append( std::ifstream & a_is )
{
	//these functions do not really work right now....
	std::cerr << "append: you still have to change that..." << std::endl;
	this->clear_clu_list();

	int l_classes;
	int l_templates;

	a_is.read((char*)&m_width,sizeof(m_width));
	a_is.read((char*)&m_height,sizeof(m_height));
	a_is.read((char*)&l_classes,sizeof(l_classes));
	a_is.read((char*)&l_templates,sizeof(l_templates));

	//m_classes += l_classes;//if append not class

	a_is.read((char*)&m_elem,sizeof(m_elem));
	a_is.read((char*)&m_bins,sizeof(m_bins));

	a_is.read((char*)&m_min_norm,sizeof(m_min_norm));

	int l_size = l_templates;

	for( int l_i=0; l_i<l_size; ++l_i )
	{
		CvMat * lp_mat = cv::cv_read(a_is);
		IplImage * lp_cnt = cv::cv_read_img(a_is);

		m_rec.push_back(lp_mat);
		m_cnt.push_back(lp_cnt);
	}
	int l_pre_classes = this->get_classes()-1;//if append not class

	for( int l_j=0; l_j<l_size; ++l_j )
	{
		int l_class;

		Ipp8u * lp_bit = ippsMalloc_8u(m_elem);

		a_is.read((char*)&l_class,sizeof(l_class));

		l_class += l_pre_classes;

		for( int l_i=0; l_i<m_elem; ++l_i )
		{
			a_is.read((char*)&lp_bit[l_i],sizeof(lp_bit[l_i]));
		}
		this->add_bit_list(lp_bit,l_class);

		ippsFree(lp_bit);
	}
	std::cerr << "num of templates: " << this->get_templates() << std::endl;
	std::cerr << "num of classes:   " << this->get_classes() << std::endl;

	return a_is;
}

template <int M, int N, int S, int G>
bool cv_dot_template<M,N,S,G>::save( std::string a_name )
{
	std::ofstream l_file(	a_name.c_str(),
							std::ofstream::out |
							std::ofstream::binary );

	if( l_file.fail() == true )
	{
		printf("cv_dot_template: could not open file for writing!\n");
		return false;
	}
	this->write( l_file );

	l_file.close();

	return true;
}

template <int M, int N, int S, int G>
bool cv_dot_template<M,N,S,G>::load( std::string a_name )
{
	std::cerr << "load" << std::endl;

	std::ifstream l_file(	a_name.c_str(),
							std::ifstream::in |
							std::ifstream::binary );

	if( l_file.fail() == true )
	{
		printf("cv_dot_template: could not open file for reading!\n");
		return false;
	}
	this->read( l_file );

	l_file.close();

	return true;
}

template <int M, int N, int S, int G>
bool cv_dot_template<M,N,S,G>::append( std::string a_name )
{
	std::cerr << "load" << std::endl;

	std::ifstream l_file(	a_name.c_str(),
							std::ifstream::in |
							std::ifstream::binary );

	if( l_file.fail() == true )
	{
		printf("cv_dot_template: could not open file for appending!\n");
		return false;
	}
	this->append( l_file );

	l_file.close();

	return true;
}

}//end of namespace...

#endif
