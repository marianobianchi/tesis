#!/usr/bin/env python
#coding=utf-8


import numpy as np
import cv2


###############################
# GETTING STARTED WITH IMAGES
###############################



################
# Abrir imagen
################
def open_image():
    img_color     = cv2.imread('RSCN1018.JPG') # cv2.IMREAD_COLOR
    img_grayscale = cv2.imread('RSCN1018.JPG', cv2.IMREAD_GRAYSCALE)
    img_alpha     = cv2.imread('RSCN1018.JPG', cv2.IMREAD_UNCHANGED) # Loads image as such including alpha channel

    return img_color

#########################################
# Guardar imagen y la convierte a png
#########################################
def save_image():
    img_color = open_image()
    cv2.imwrite('RSCN1018.png', img_color)


############################
# Se puede usar matplotlib
############################
# Ejemplo : http://docs.opencv.org/trunk/doc/py_tutorials/py_gui/py_image_display/py_image_display.html#using-matplotlib










################################
# GETTING STARTED WITH VIDEOS
################################


def watch_video(video_filename=''):
    if not video_filename:
        # Captura de la camara web
        cap = cv2.VideoCapture(0)
    else:
        # Captura desde un archivo
        cap = cv2.VideoCapture(video_filename)

    while(True):
        # Tomo un frame
        ret, frame = cap.read()

        # Se pasa el frame a escala de grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Muestro el frame
        cv2.imshow('frame',gray)

        # Cuando se toque la tecla q, se detiene
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cuando se termine, se libera la captura del video
    cap.release()
    cv2.destroyAllWindows()


def write_video():
    cap = cv2.VideoCapture(0)

    # Tomo un frame para definir el tamaño del video
    ret, frame = cap.read()
    height = len(frame)
    width = len(frame[0])

    # Defino el codec y armo el objeto VideoWriter
    fourcc = cv2.cv.CV_FOURCC(*'XVID')
    out = cv2.VideoWriter(
        'output.avi', # Video file name
        fourcc, # Codec
        20.0, # Frames por segundo
        (width,height), # Tamaño de los frames
        True, # Flag de color. True = color, False = gris
    )

    while(cap.isOpened()):
        ret, frame = cap.read()

        # Le aplicamos un efecto a la imagen
        frame = cv2.flip(frame,0)

        # Guardo el frame con efecto
        out.write(frame)

        # Y lo muestro
        cv2.imshow('image', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            out.release()

    cv2.destroyAllWindows()


################################
# GETTING STARTED WITH DRAWING
################################

def draw_circle():

    # Creo una imagen blanca
    img = np.zeros((512, 512, 1))
    img.fill(255)

    cv2.circle(
        img, # Imagen
        (447,63), # Centro
        63, # Radio
        (0,0,0), # Color
        -1, # Grosor
    )

    cv2.imshow('image', img)

    while(cv2.waitKey(1) & 0xFF != ord('q')):
        pass

    cv2.destroyAllWindows()






if __name__ == '__main__':
    pass
