
#include "cv_dot_template.h"
#include "cv_camera.h"
#include "cv_esm.h"


template <int M, int N, int S, int G>
bool get_key_and_act(   bool &l_show_esm,
                        bool &l_learn_onl,
                        cv::cv_dot_template<M,N,S,G> &l_template,
                        std::vector<cv::cv_esm*> &l_esm_vec,
                        std::vector<CvMat*> &l_cur_vec,
                        IplImage *lp_color,
                        IplImage *lp_mean,
                        IplImage *lp_gray)
{

    int l_key = cvWaitKey(1);

    // Si se aprieta E
    if( l_key == 101 )
    {
        l_show_esm = !l_show_esm;
    }
    // Si se aprieta I
    if( l_key == 105 )
    {
        l_learn_onl = !l_learn_onl;

        if( l_learn_onl == false )
        {
            l_template.clear_clu_list();
            l_template.cluster_heu(4);
        }
    }

    // Si se aprieta D
    if( l_key == 100 )
    {
        for( int l_i=0; l_i<l_template.get_classes(); ++l_i )
        {
            delete l_esm_vec[l_i];
            cvReleaseMat(&l_cur_vec[l_i]);
        }
        l_esm_vec.clear();
        l_cur_vec.clear();

        l_template.clear_clu_list();
        l_template.clear_bit_list();
        l_template.clear_rec_list();
        l_template.clear_cnt_list();
    }
    // Si se aprieta ESC
    if( l_key == 27 )
    {
        // User wants to exit
        for( int l_i=0; l_i<l_template.get_classes(); ++l_i )
        {
            delete l_esm_vec[l_i];
            cvReleaseMat(&l_cur_vec[l_i]);
        }
        l_esm_vec.clear();
        l_cur_vec.clear();

        cvReleaseImage(&lp_color);
        cvReleaseImage(&lp_gray);
        cvReleaseImage(&lp_mean);

        l_template.clear_clu_list();
        l_template.clear_bit_list();
        l_template.clear_rec_list();
        l_template.clear_cnt_list();

        return false;
    }

    return true;

}


template <int M, int N, int S, int G>
void draw_rectangle(cv::cv_dot_template<M,N,S,G> &l_template,
                    int l_x,
                    int l_y,
                    IplImage *lp_gray,
                    IplImage *lp_color)
{

    // Si el mouse está dentro de la ventana, dibuja
    // el recuadro
    if( l_x-l_template.get_width()/2 >= 0 ||
        l_y-l_template.get_height()/2 >= 0 ||
        l_x+l_template.get_width()/2 <= lp_gray->width-1 ||
        l_y+l_template.get_height()/2 <= lp_gray->height-1 )
    {
        CvPoint l_pt1;
        CvPoint l_pt2;
        CvPoint l_pt3;
        CvPoint l_pt4;

        // punto de la esquina superior izquierda del recuadro
        l_pt1.x = l_x-l_template.get_width()/2;
        l_pt1.y = l_y-l_template.get_height()/2;

        // punto de la esquina superior derecha del recuadro
        l_pt2.x = l_x+l_template.get_width()/2;
        l_pt2.y = l_y-l_template.get_height()/2;

        // punto de la esquina inferior derecha del recuadro
        l_pt3.x = l_x+l_template.get_width()/2;
        l_pt3.y = l_y+l_template.get_height()/2;

        // punto de la esquina inferior izquierda del recuadro
        l_pt4.x = l_x-l_template.get_width()/2;
        l_pt4.y = l_y+l_template.get_height()/2;

        cvLine(lp_color,l_pt1,l_pt2,CV_RGB(0,0,0),3);
        cvLine(lp_color,l_pt2,l_pt3,CV_RGB(0,0,0),3);
        cvLine(lp_color,l_pt3,l_pt4,CV_RGB(0,0,0),3);
        cvLine(lp_color,l_pt4,l_pt1,CV_RGB(0,0,0),3);

        cvLine(lp_color,l_pt1,l_pt2,CV_RGB(255,255,0),1);
        cvLine(lp_color,l_pt2,l_pt3,CV_RGB(255,255,0),1);
        cvLine(lp_color,l_pt3,l_pt4,CV_RGB(255,255,0),1);
        cvLine(lp_color,l_pt4,l_pt1,CV_RGB(255,255,0),1);
    }

}



template <int M, int N, int S, int G>
void take_template_and_learn(int l_x,
                             int l_y,
                             cv::cv_dot_template<M,N,S,G> &l_template,
                             IplImage *lp_mean,
                             std::vector<CvMat*> &l_cur_vec,
                             std::vector<cv::cv_esm*> &l_esm_vec)
{

    cv::cv_esm * lp_esm = new cv::cv_esm;

    // Matriz de 3 filas x 4 columnas, de tipo float
    CvMat * lp_result = cvCreateMat(3,4,CV_32F);

    // Setea todos los valores en 1 (la fila 2 (lp_result[2,:]) queda así)
    cvSet(lp_result,cvRealScalar(1));

    //Fila 0 y 1, columna 0: son el punto izquierdo de arriba del recuadro
    CV_MAT_ELEM(*lp_result,float,0,0) = l_x-l_template.get_width()/2;
    CV_MAT_ELEM(*lp_result,float,1,0) = l_y-l_template.get_height()/2;

    //Fila 0 y 1, columna 1: son el punto derecho de arriba del recuadro
    CV_MAT_ELEM(*lp_result,float,0,1) = l_x+l_template.get_width()/2;
    CV_MAT_ELEM(*lp_result,float,1,1) = l_y-l_template.get_height()/2;

    //Fila 0 y 1, columna 2: son el punto derecho de abajo del recuadro
    CV_MAT_ELEM(*lp_result,float,0,2) = l_x+l_template.get_width()/2;
    CV_MAT_ELEM(*lp_result,float,1,2) = l_y+l_template.get_height()/2;

    //Fila 0 y 1, columna 3: son el punto izquierdo de abajo del recuadro
    CV_MAT_ELEM(*lp_result,float,0,3) = l_x-l_template.get_width()/2;
    CV_MAT_ELEM(*lp_result,float,1,3) = l_y+l_template.get_height()/2;

    // Toma un template de la imagen del recuadro
    l_template.create_bit_list_fast(lp_mean,l_y,l_x,7,0.9);
    l_template.cluster_heu(4);

    lp_esm->learn(	lp_mean,l_y-l_template.get_height()/2,l_x-l_template.get_width()/2,
                    l_template.get_height(),l_template.get_width());

    l_esm_vec.push_back(lp_esm);
    l_cur_vec.push_back(lp_result);

}




void track_template(std::vector<cv::cv_esm*> &l_esm_vec,
                    int l_j,
                    IplImage *lp_mean,
                    CvMat *lp_rec,
                    bool &l_successful,
                    float *lp_max_val,
                    std::list<cv::cv_candidate*>::iterator l_i,
                    std::vector<CvMat*>l_cur_vec,
                    IplImage *lp_color,
                    bool l_learn_onl)
{
    if( l_esm_vec[l_j] != NULL )
    {
        CvMat * lp_result = l_esm_vec[l_j]->track(lp_mean,lp_rec,10,10);

        if( l_esm_vec[l_j]->get_ncc() > 0.85 )
        {
            l_successful = true;
            lp_max_val[l_j] = (*l_i)->m_val;
            cvCopy(lp_result,l_cur_vec[l_j]);

            //cv::cv_draw_pose(lp_color,lp_result,lp_rec);

            if( l_learn_onl == true )
            {
                cv::cv_draw_poly(lp_color,lp_rec,1,255,255,255);

                cv::cv_draw_poly(lp_color,lp_result,3,0,0,0);
                cv::cv_draw_poly(lp_color,lp_result,1,255,255,255);
            }
            else
            {
                cv::cv_draw_poly(lp_color,lp_result,3,0,255,0);
                cv::cv_draw_poly(lp_color,lp_result,1,0,0,0);
            }
        }
        cvReleaseMat(&lp_result);
    }
}


int main( int argc, char * argv[] )
{
    //the number of pixels the template is invariant to translation (set to 7 - you should leave it as it is)
	const int l_T=7;

	//width of the template in number of regions - if e.g. l_N=10 and l_T=7 the template width is 70 pixels
	const int l_N=77/l_T;

	//height of the template in number of regions - if e.g. l_M=10 and l_T=7 the template height is 70 pixels
	const int l_M=77/l_T;

	// Supongo que es el tamaño de la imagen
	const int l_IN=640;
	const int l_IM=480;

	//number of regions unmasked - for small templates l_G=l_N*l_M. For larger templates tracking arbitrary shapes
    // It might be set to a much lower number (in order to deal with changing background in case of non-rectangular shaped objects)
	const int l_G=121;

	int l_learn_thres=l_G*0.9;


	int l_detect_thres=l_G*0.8;

    // ancho = l_T * l_M
    // alto = l_T * l_N
	cv::cv_dot_template<l_M,l_N,l_T,l_G> l_template(23);

	cv::cv_create_window("hallo1");


	std::vector<cv::cv_esm*> l_esm_vec;
	std::vector<CvMat*> l_cur_vec;

	cv::cv_camera l_camera;
	cv::cv_timer l_timer_computegradients;
	cv::cv_timer l_timer_process_gradients;
	cv::cv_mouse l_mouse;

	l_mouse.start("hallo1");
	l_camera.set_cam(cv::usb);


	int l_x=-1;
	int l_y=-1;

	bool l_show_esm = false;
	bool l_learn_onl = false;

	bool user_wants_to_continue = true;

	if( l_camera.start_capture_from_cam() == false )
	{
		printf("main_dog: the camera could not be initalized!");
		return 0;
	}

	while(user_wants_to_continue)
	{
		IplImage * lp_color = NULL;
		IplImage * lp_gray  = NULL;
		IplImage * lp_mean  = NULL;

		lp_color = l_camera.get_image();
		lp_gray  = cv::cv_convert_color_to_gray(lp_color);
		lp_mean  = cv::cv_smooth(lp_gray,5);


        // Deja en l_x y en l_y el último punto de la ventana
        // en donde se posó el mouse
		int l_xx = l_mouse.get_x();
		int l_yy = l_mouse.get_y();
        bool did_right_click = (l_mouse.get_event() == 2);

		if( l_xx >= 0 && l_yy >= 0 )
		{
			l_x = l_xx;
			l_y = l_yy;
		}
		bool mouse_is_inside_the_window = (l_x != -1 && l_y != -1);


        draw_rectangle(l_template, l_x, l_y, lp_gray, lp_color);


		if( mouse_is_inside_the_window && did_right_click )
		{
            take_template_and_learn(l_x, l_y, l_template, lp_mean, l_cur_vec, l_esm_vec);
		}


		cv::cv_timer l_timer_gradients;
		cv::cv_timer l_timer_computegradients;
		cv::cv_timer l_timer_process_gradients;


		l_timer_gradients.start();
		l_timer_computegradients.start();

        // Computa los gradientes de la imagen suavizada
		std::pair<Ipp8u*,Ipp32f*> l_img = l_template.compute_gradients(lp_mean,1);

		l_timer_computegradients.stop();
		l_timer_process_gradients.start();

        // Crea una lista de listas de cv_candidate
		std::list<cv::cv_candidate*> * lp_list = NULL;

        // Procesa los gradientes
		if( l_learn_onl == true )
		{
			lp_list = l_template.online_process(l_img.first,l_detect_thres,l_IN/l_T,l_IM/l_T);
		}
		else
		{
			lp_list = l_template.process(l_img.first,l_detect_thres,l_IN/l_T,l_IM/l_T);
		}
		l_timer_process_gradients.stop();
		l_timer_gradients.stop();

		ippsFree(l_img.first);
		ippsFree(l_img.second);

        // NO SE QUE ES LP_LIST... La llamo "Lista de candidatos", aunque no se de qué!
		if( lp_list != NULL )
		{
			int count_of_all_candidates_of_all_templates=0;

			float * lp_max_val = new float[l_template.get_classes()];
			float * lp_res_val = new float[l_template.get_classes()];


            // REVISAR DESDE ACA!!!!!!!!!!!!!!!!!!!!!!
            // Entiendo que lo que hace acá es buscar un template que matchee con
            // alguna parte de la imagen
			for( int l_j=0; l_j<l_template.get_classes(); ++l_j )
			{
				int l_counter = 0;
				int l_end_counter=7;
				bool l_successful=false;

				lp_max_val[l_j]=0;

				count_of_all_candidates_of_all_templates += lp_list[l_j].size();

                // Los ordena según "m_val"
				lp_list[l_j].sort(cv::cv_candidate_ptr_cmp());

				for( std::list<cv::cv_candidate*>::iterator l_i=lp_list[l_j].begin(); l_i!=lp_list[l_j].end(); ++l_i )
				{
                    // Sigue hasta consumir lp_list o los primeros l_end_counter de lp_list, lo que ocurra primero
					if( l_counter < l_end_counter )
					{
						CvMat * lp_rec = cvCreateMat(3,4,CV_32F);

                        // TODO:
						// mp_rec se va completando a través del método create_bit_list_fast
						// y entiendo que lo que guarda es la perspectiva del template que
						// se está aprendiendo (¿la perspectiva con respecto a algo?)
						cvCopy(l_template.get_rec()[(*l_i)->m_ind-1],lp_rec);

						CV_MAT_ELEM(*lp_rec,float,0,0) += (*l_i)->m_col;
						CV_MAT_ELEM(*lp_rec,float,1,0) += (*l_i)->m_row;
						CV_MAT_ELEM(*lp_rec,float,0,1) += (*l_i)->m_col;
						CV_MAT_ELEM(*lp_rec,float,1,1) += (*l_i)->m_row;
						CV_MAT_ELEM(*lp_rec,float,0,2) += (*l_i)->m_col;
						CV_MAT_ELEM(*lp_rec,float,1,2) += (*l_i)->m_row;
						CV_MAT_ELEM(*lp_rec,float,0,3) += (*l_i)->m_col;
						CV_MAT_ELEM(*lp_rec,float,1,3) += (*l_i)->m_row;

						if( l_show_esm == true )
						{
                            // Se hace tracking con ESM usando lp_rec
                            track_template(l_esm_vec,
                                           l_j,
                                           lp_mean,
                                           lp_rec,
                                           l_successful,
                                           lp_max_val,
                                           l_i,
                                           l_cur_vec,
                                           lp_color,
                                           l_learn_onl);

                            // Si se encontró, salgo del for
							if (l_successful) break;
						}
						else
						{
							if( l_counter == 0 )
							{
                                // TODO: ver que hace esto. ¿Por que anda si le faltan parámetros?
								l_template.render(
                                    lp_color,
                                    l_template.get_cnt()[(*l_i)->m_ind-1],
                                    (*l_i)->m_row,
                                    (*l_i)->m_col
                                );

								lp_max_val[l_j] = (*l_i)->m_val;
								cv::cv_draw_poly(lp_color,lp_rec,3,255,255,255);
								cv::cv_draw_poly(lp_color,lp_rec,1,0,0,0);
							}
							else
							{
								cv::cv_draw_poly(lp_color,lp_rec,2,0,0,0);
							}
						}
						cvReleaseMat(&lp_rec);
					}
					else
					{
						break;
					}
					++l_counter;
				}


                // Si con el for anterior no se trackeó el objeto deseado, se trata de seguir usando
                // l_cur_vec (creo que tiene los templates en crudo, es decir, lo que sale directo
                // del cuadradito con el que tomamos la muestra al hacer click derecho)
				if( l_learn_onl == true && l_show_esm == true && !l_successful && l_esm_vec[l_j] != NULL)
				{
                    // lp_result es de 3 filas por 4 columnas
                    CvMat * lp_result = l_esm_vec[l_j]->track(lp_gray,l_cur_vec[l_j],10,10);

                    if( l_esm_vec[l_j]->get_ncc() > 0.85 )
                    {
                        l_template.online_create_bit_list_fast(lp_mean,lp_result,l_j,7,0.9);
                        cvCopy(lp_result,l_cur_vec[l_j]);

                        cv::cv_draw_poly(lp_color,lp_result,2,255,255,0);
                    }
                    cvReleaseMat(&lp_result);
				}
			}

			for( int l_i=0; l_i<l_template.get_classes(); ++l_i )
			{
				cv::empty_ptr_list(lp_list[l_i]);
			}
			std::cerr << "tim: " << (int)(l_timer_gradients.get_time()*1000) << "ms; fps: " << (int)l_timer_gradients.get_fps() << "fps ";
			std::cerr << "pre: " << (int)(l_timer_computegradients.get_time()*1000) << "ms; pro: " << (int)(l_timer_process_gradients.get_time()*1000) << "ms; ";

			for( int l_i=0; l_i<l_template.get_classes(); ++l_i )
			{
				std::cerr << (int)(lp_max_val[l_i]) << ";";
			}
			std::cerr << " size: " << l_template.get_templates() << " ," << count_of_all_candidates_of_all_templates << ";" << l_template.get_classes() << "  " << char(13) << std::flush;

			delete[] lp_max_val;
			delete[] lp_list;
		}
		else
		{
			std::cerr << "tim: " << (int)(l_timer_gradients.get_time()*1000) << "ms; fps: " << (int)l_timer_gradients.get_fps() << "fps ";
			std::cerr << "pre: " << (int)(l_timer_computegradients.get_time()*1000) << "ms; pro: " << (int)(l_timer_process_gradients.get_time()*1000) << "ms; ";
			std::cerr << " size: " << l_template.get_templates() << "    " << char(13) << std::flush;
		}
		cv::cv_show_image(lp_color,"hallo1");

		user_wants_to_continue = get_key_and_act(
            l_show_esm,
            l_learn_onl,
            l_template,
            l_esm_vec,
            l_cur_vec,
            lp_color,
            lp_mean,
            lp_gray
        );

		cvReleaseImage(&lp_color);
		cvReleaseImage(&lp_gray);
		cvReleaseImage(&lp_mean);
	}
	return 0;
}
