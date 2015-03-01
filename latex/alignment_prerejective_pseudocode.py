def alignment(obj_data, scene_data):
    # Resultado
    mejor_valor = MAX_INT
    mejor_transf = TRANSFORMACION_IDENTIDAD
    # Calculo de descriptores
    obj_desc = ecv_context_descriptors(obj_data)
    scene_desc = ecv_context_descriptors(scene_data)
    while queden_iteraciones_por_correr:  # RANSAC
        # Paso 1
        rand_obj_desc = tomar_n_puntos_random(obj_desc)
        corr_scene_desc = find_correspondences(scene_desc,
                                               rand_obj_desc)
        # Descarte rapido si no son isometricos (Paso 2)
        if poligonos_son_isometricos(rand_obj_desc,
                                     corr_scene_desc):
            # Paso 3
            transf = estimar_transformacion(rand_obj_desc,
                                            corr_scene_desc)
            # Paso 4
            transf_obj = transformar(obj_data, transf)
            # Paso 5
            pares_de_inliers = vecinos_mas_cercanos(transf_obj,
                                                    scene_data)
            if len(pares_de_inliers) >= umbral_inliers:
                # Paso 6
                sq_dist = dist_cuadratica_media(pares_de_inliers)
                if sq_dist < mejor_valor:
                    mejor_valor = sq_dist; mejor_transf = transf

        descontar_una_iteracion()

    return mejor_transf, mejor_valor
