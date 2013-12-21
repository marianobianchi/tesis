#!/usr/bin/env python

import cv2




################
# Abrir imagen
################

img_color     = cv2.imread('RSCN1018.JPG') # cv2.IMREAD_COLOR
img_grayscale = cv2.imread('RSCN1018.JPG', cv2.IMREAD_GRAYSCALE)
img_alpha     = cv2.imread('RSCN1018.JPG', cv2.IMREAD_UNCHANGED) # Loads image as such including alpha channel


#########################################
# Guardar imagen y la convierte a png
#########################################

cv2.imwrite('RSCN1018.png', img_color)


############################
# Se puede usar matplotlib
############################
# Ejemplo : http://docs.opencv.org/trunk/doc/py_tutorials/py_gui/py_image_display/py_image_display.html#using-matplotlib
