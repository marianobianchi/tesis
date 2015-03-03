def icp(obj_points, scene_points):
    transformacion = TRANSFORMACION_IDENTIDAD
    obj_points_transf = obj_points
    error_cuadratico = MAX_INT

    while no_se_cumplen_condiciones_de_corte:
        closest_points = vecinos_mas_cercanos(
            obj_points_transf,
            scene_points
        )
        # Calcular la transformacion usando por ejemplo SVD
        transf = calcular_transformacion(closest_points)
        # Aplicar transformacion al objeto
        obj_points_transf = transformar(obj_points_transf,
                                        transf)
        # Calcular error cuadratico medio
        sq_dist = dist_cuadratica_media(obj_points_transf,
                                        scene_points)
        if abs(error_cuadratico, sq_dist) < umbral_de_error:
            break  # Salir del while

        error_cuadratico = sq_dist

    return obj_point_transf, error_cuadratico
