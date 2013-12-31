#!/usr/bin/env python

def moving_circle():
    """
    Crea una tira de imagenes de un circulo moviendose
    """
    # Creo una imagen blanca
    image_height = 600
    image_width = 800
    base_img = np.zeros((image_height, image_width, 1))
    base_img.fill(255)

    img_name = 'videos/moving_circle/mc_{i:03d}.jpg'

    circle_radio = 40

    circle_center = (80,80)

    mov_side = 10
    mov_vertical = 50

    for i in range(100):
        img = base_img.copy() # Copio la imagen en blanco

        cv2.circle(
            img,
            circle_center,
            circle_radio,
            (0,0,0),
            -1,
        )

        # Guardo la imagen
        cv2.imwrite(img_name.format(i=i), img)

        #####################################################
        # Acomodo todas las variables para la próxima imagen
        #####################################################

        # Muevo el circulo
        circle_center = (circle_center[0] + mov_side, circle_center[1] + mov_vertical)

        # Si se pasa de algun borde, cambio las direcciones de movimiento
        # Si se pasa para abajo
        if circle_center[1] + abs(mov_vertical) >= image_height:
            mov_vertical *= -1

        # Si se pasa para la derecha
        if circle_center[0] + abs(mov_side) >= image_width:
            mov_side *= -1

        # Si se pasa para arriba
        if circle_center[1] - abs(mov_vertical) <= 0:
            mov_vertical *= -1

        # Si se pasa para la izquierda
        if circle_center[0] - abs(mov_side) <= 0:
            mov_side *= -1


if __name__ == '__main__':
    moving_circle()