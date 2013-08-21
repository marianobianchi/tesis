#coding=utf-8

from __future__ import unicode_literals

import cv
import pylab

def extract_channel(imsrc, chnum):
    channels = [0.0, 0.0, 0.0]
    imdst = cv.CreateImage(cv.GetSize(imsrc), imsrc.depth, imsrc.channels)
    for nrow in range(imsrc.height):
        for ncol in range(imsrc.width):
            chcopy = channels[:]
            chcopy[chnum] = imsrc[nrow, ncol]
            imdst[nrow, ncol] = tuple(chcopy)
    
    return imdst
    


im1 = cv.LoadImage('../papers/tesis-nakama/implementacion/desk/desk_1_1.png')
im2 = cv.LoadImage('../papers/tesis-nakama/implementacion/desk/desk_1_4.png')

size = cv.GetSize(im1)


#im1r = cv.CreateImage(size, im1.depth, 1)
#im1g = cv.CreateImage(size, im1.depth, 1)
#im1b = cv.CreateImage(size, im1.depth, 1)


#cv.Split(im1, im1r, im1g, im1b, None)



#im2r = cv.CreateImage(size, im2.depth, 1)
#im2g = cv.CreateImage(size, im2.depth, 1)
#im2b = cv.CreateImage(size, im2.depth, 1)


#im1r3 = extract_channel(im1, 0)
#im1g3 = extract_channel(im1, 1)
#im1b3 = extract_channel(im1, 2)


#pylab.figure()
#pylab.imshow(im1r3[:,:])
#pylab.show(block=False)

#pylab.figure()
#pylab.imshow(im1g3[:,:])
#pylab.show(block=False)

#pylab.figure()
#pylab.imshow(im1b3[:,:])
#pylab.show(block=False)
